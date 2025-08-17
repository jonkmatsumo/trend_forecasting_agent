"""
Text Normalization Utilities
Robust Unicode handling and standardization for intent recognition.
Supports dual-view normalization: loose for keywords, strict for regex slots.
"""

import re
import unicodedata
from typing import Optional, Tuple, Dict

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False
    ftfy = None

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    emoji = None

# Character cleanup patterns
_ZERO_WIDTH_RE = re.compile(r'[\u200B-\u200D\u2060\uFEFF]')
_CONTROL_RE = re.compile(r'[\u0000-\u0008\u000B-\u000C\u000E-\u001F\u007F]')  # Exclude \t, \n, \r

# Regex patterns for normalization
_VS_RE = re.compile(r'\b(versus|vs\.?)\b', flags=re.IGNORECASE)
_WS_RE = re.compile(r'\s+')  # Standard whitespace collapse
# Leading/trailing punctuation to trim, but keep internal punctuation (/, %, :, .)
_TRIM_RE = re.compile(r'^[\s\.,;:!?(){}\[\]"\']+|[\s\.,;:!?(){}\[\]"\']+$')

# Link protection patterns for URLs and emails
_LINK_HEAD_RE = re.compile(r'^\s*((?:https?://|www\.|[^\s@\.,;:!?(){}[\]"\']+@[^\s@\.,;:!?(){}[\]"\']+\.[^\s@\.,;:!?(){}[\]"\']+))(?=[\s\.,;:!?(){}\[\]"\']|$)')

# Emoji detection pattern
_EMOJI_RE = re.compile(r'[\U0001F300-\U0001FAFF]')

# Smart punctuation mapping
CHAR_MAP = {
    "\u2018": "'", "\u2019": "'",   # ' '
    "\u201C": '"', "\u201D": '"',   # " "
    "\u2013": "-", "\u2014": "-",   # – —
}




def _map_chars(s: str) -> str:
    """Map smart punctuation to ASCII equivalents."""
    return ''.join(CHAR_MAP.get(ch, ch) for ch in s)


def _process_emojis(s: str, emoji_policy: str = "keep") -> str:
    """
    Process emojis according to the specified policy using the emoji library.
    
    Args:
        s: Input string that may contain emojis
        emoji_policy: Policy for handling emojis ("keep", "strip", "map")
        
    Returns:
        String with emojis processed according to policy
        
    Raises:
        ValueError: If emoji_policy is not one of the allowed values
    """
    if emoji_policy not in ["keep", "strip", "map"]:
        raise ValueError(f"Invalid emoji_policy: {emoji_policy}. Must be one of: keep, strip, map")
    
    if emoji_policy == "keep":
        return s
    elif emoji_policy == "strip":
        # Replace emojis with a space to preserve spacing
        result = _EMOJI_RE.sub(" ", s)
        # Ensure leading space is preserved if emoji was at start
        if s and _EMOJI_RE.match(s[0]):
            result = " " + result.lstrip()
        return result
    elif emoji_policy == "map":
        if EMOJI_AVAILABLE:
            # Use emoji library for comprehensive emoji handling
            # First, demojize to get emoji names like :chart_increasing:
            result = emoji.demojize(s)
            # Convert emoji names to readable text by removing colons and replacing underscores with spaces
            result = re.sub(r':([^:]+):', lambda m: m.group(1).replace('_', ' '), result)

            # Clean up any double spaces that might have been created
            result = " ".join(result.split())
            return result
        else:
            # If emoji library is not available, fall back to stripping emojis
            result = _EMOJI_RE.sub(" ", s)
            # Ensure leading space is preserved if emoji was at start
            if s and _EMOJI_RE.match(s[0]):
                result = " " + result.lstrip()
            return result


def _protect_head_entity(s: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Protect leading URL or email from trimming by temporarily replacing with a placeholder.
    
    Args:
        s: Input string that may contain a leading URL or email
        
    Returns:
        Tuple of (modified_string, original_entity, trailing_punct) or (original_string, None, None)
    """
    match = _LINK_HEAD_RE.match(s)
    if not match:
        return s, None, None
    
    entity = match.group(1)
    # Check if there's trailing punctuation after the entity
    remaining = s[len(entity):].lstrip()
    trailing_punct = None
    if remaining and remaining[0] in '.,;:!?(){}[]"\'':
        trailing_punct = remaining[0]
    
    # Use a unique placeholder that won't conflict with normal text
    placeholder = f"⟦{entity}⟧"
    modified = s.replace(entity, placeholder, 1)
    return modified, entity, trailing_punct


def _restore_head_entity(s: str, entity: Optional[str], trailing_punct: Optional[str] = None) -> str:
    """
    Restore a previously protected entity from its placeholder.
    
    Args:
        s: String containing placeholder
        entity: Original entity to restore, or None if no entity was protected
        trailing_punct: Trailing punctuation that was removed, or None
        
    Returns:
        String with entity restored (trailing punctuation is not restored)
    """
    if entity is None:
        return s
    
    placeholder = f"⟦{entity}⟧"
    return s.replace(placeholder, entity, 1)


def normalize_views(text: str, 
                   *, 
                   trim_punctuation: bool = True, 
                   canonicalize_vs: bool = True, 
                   casefold_strict: bool = True,
                   protect_links: bool = False,
                   emoji_policy: str = "keep") -> Tuple[str, str, Dict]:
    """
    Normalize text with dual-view approach: loose for keywords, strict for regex slots.
    
    Args:
        text: Input text to normalize
        trim_punctuation: Whether to trim leading/trailing punctuation in strict view
        canonicalize_vs: Whether to normalize "versus" variants to "vs" in strict view
        casefold_strict: Whether to apply case folding in strict view
        protect_links: Whether to protect leading URLs/emails from edge trimming
        emoji_policy: Policy for handling emojis ("keep", "strip", "map")
        
    Returns:
        Tuple of (norm_loose, norm_strict, stats)
        - norm_loose: Repaired Unicode, standardized punctuation/whitespace (good for keywords)
        - norm_strict: norm_loose + casefold, versus→vs, optional edge-punct trim (good for regex slots)
        - stats: Normalization statistics and flags
    """
    if not text:
        return text, text, get_normalization_stats(text, text)
    
    # Store original for stats
    original = text
    
    # 1) Character cleanup (applied to both views)
    t = _ZERO_WIDTH_RE.sub('', text)
    t = _CONTROL_RE.sub('', t)
    
    # 2) Base repairs (applied to both views)
    if FTFY_AVAILABLE:
        t = ftfy.fix_text(t)
    t = unicodedata.normalize("NFKC", t)
    t = _map_chars(t)
    
    # Apply emoji processing (applied to both views)
    t = _process_emojis(t, emoji_policy)
    
    # Apply whitespace normalization after emoji processing
    t = _WS_RE.sub(" ", t).strip()
    
    # Loose view: keep case and edge punctuation
    norm_loose = t
    
    # Strict view: apply additional normalizations
    s = t.casefold() if casefold_strict else t
    if canonicalize_vs:
        s = _VS_RE.sub("vs", s)
    if trim_punctuation:
        # Protect leading entities if requested
        head_entity = None
        trailing_punct = None
        if protect_links:
            s, head_entity, trailing_punct = _protect_head_entity(s)
        
        # Apply trimming
        s = _TRIM_RE.sub("", s)
        
        # Restore protected entity
        if head_entity is not None:
            s = _restore_head_entity(s, head_entity, trailing_punct)
            # If there was trailing punctuation, it should be removed in strict view
            # The entity is now protected, so we can safely trim any remaining trailing punctuation
            if trailing_punct:
                s = s.rstrip(trailing_punct)
    
    # Enhanced statistics
    stats = get_normalization_stats(original, s)
    stats.update({
        "zero_width_removed": bool(_ZERO_WIDTH_RE.search(original)),
        "controls_removed": bool(_CONTROL_RE.search(original)),
        "quotes_fixed": any(q in original for q in ("\u2018", "\u2019", "\u201C", "\u201D")),
        "dashes_fixed": any(d in original for d in ("\u2013", "\u2014")),
        "ws_collapsed": "  " in original or "\u00A0" in original or "\t" in original or "\n" in original or "\r" in original,
        "casefold_applied": casefold_strict,
        "trim_punctuation": trim_punctuation,
        "canon_vs": canonicalize_vs,
        "protect_links": protect_links,
        "emoji_policy": emoji_policy,
        "had_emojis": bool(_EMOJI_RE.search(original)),
        "emoji_library_used": EMOJI_AVAILABLE and emoji_policy == "map",
        "loose_length": len(norm_loose),
        "strict_length": len(s),
    })
    
    return norm_loose, s, stats


def normalize_with_ftfy(text: str, 
                       casefold: bool = True,
                       canonicalize_vs: bool = True,
                       trim_punctuation: bool = True,
                       protect_links: bool = False,
                       emoji_policy: str = "keep") -> str:
    """
    Normalize text for robust regex/rule matching while preserving meaning.
    
    Note: This function returns the strict view for backward compatibility.
    For dual-view normalization, use normalize_views() instead.
    
    Args:
        text: Input text to normalize
        casefold: Whether to apply case folding (stronger than lowercasing)
        canonicalize_vs: Whether to normalize "versus" variants to "vs"
        trim_punctuation: Whether to trim leading/trailing punctuation
        protect_links: Whether to protect leading URLs/emails from edge trimming
        emoji_policy: Policy for handling emojis ("keep", "strip", "map")
        
    Returns:
        Normalized text suitable for intent recognition (strict view)
    """
    # Use the new dual-view function and return strict view
    _, strict, _ = normalize_views(
        text,
        trim_punctuation=trim_punctuation,
        canonicalize_vs=canonicalize_vs,
        casefold_strict=casefold,
        protect_links=protect_links,
        emoji_policy=emoji_policy
    )
    return strict


def get_normalization_stats(original: str, normalized: str, max_compare: int = 4096) -> Dict:
    """
    Get statistics about the normalization process.
    
    Args:
        original: Original text
        normalized: Normalized text
        max_compare: Maximum characters to compare for detailed stats
        
    Returns:
        Dictionary with normalization statistics
    """
    if not original and not normalized:
        return {
            'characters_changed': 0,
            'length_change': 0,
            'vs_canonicalized': False,
            'unicode_fixed': False,
            'original_length': 0,
            'normalized_length': 0
        }
    
    # Enhanced character change detection
    n = min(len(original), len(normalized), max_compare)
    replaced = sum(1 for i in range(n) if original[i] != normalized[i])
    replaced += abs(len(original) - len(normalized))
    
    return {
        'characters_changed': replaced,
        'length_change': len(normalized) - len(original),
        'vs_canonicalized': bool(_VS_RE.search(original.lower())) and (" vs " in normalized or normalized.endswith("vs")),
        'unicode_fixed': FTFY_AVAILABLE and (original != normalized),
        'original_length': len(original),
        'normalized_length': len(normalized)
    }


class TextNormalizer:
    """Configurable text normalizer with statistics tracking and dual-view support."""
    
    def __init__(self, 
                 casefold: bool = True,
                 canonicalize_vs: bool = True,
                 trim_punctuation: bool = True,
                 protect_links: bool = False,
                 emoji_policy: str = "keep"):
        """
        Initialize text normalizer.
        
        Args:
            casefold: Whether to apply case folding
            canonicalize_vs: Whether to normalize "versus" variants
            trim_punctuation: Whether to trim leading/trailing punctuation
            protect_links: Whether to protect leading URLs/emails from edge trimming
            emoji_policy: Policy for handling emojis ("keep", "strip", "map")
        """
        self.casefold = casefold
        self.canonicalize_vs = canonicalize_vs
        self.trim_punctuation = trim_punctuation
        self.protect_links = protect_links
        self.emoji_policy = emoji_policy
    
    def normalize(self, text: str) -> Tuple[str, Dict]:
        """
        Normalize text and return statistics (returns strict view for backward compatibility).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Tuple of (normalized_text, statistics) - normalized_text is the strict view
        """
        _, strict, stats = normalize_views(
            text,
            trim_punctuation=self.trim_punctuation,
            canonicalize_vs=self.canonicalize_vs,
            casefold_strict=self.casefold,
            protect_links=self.protect_links,
            emoji_policy=self.emoji_policy
        )
        return strict, stats
    
    def normalize_only(self, text: str) -> str:
        """
        Normalize text without statistics (returns strict view for backward compatibility).
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text (strict view)
        """
        _, strict, _ = normalize_views(
            text,
            trim_punctuation=self.trim_punctuation,
            canonicalize_vs=self.canonicalize_vs,
            casefold_strict=self.casefold,
            protect_links=self.protect_links,
            emoji_policy=self.emoji_policy
        )
        return strict
    
    def normalize_both(self, text: str) -> Tuple[str, str, Dict]:
        """
        Normalize text with both loose and strict views.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Tuple of (norm_loose, norm_strict, stats)
        """
        return normalize_views(
            text,
            trim_punctuation=self.trim_punctuation,
            canonicalize_vs=self.canonicalize_vs,
            casefold_strict=self.casefold,
            protect_links=self.protect_links,
            emoji_policy=self.emoji_policy
        )
    
    def normalize_for_keywords(self, text: str) -> str:
        """
        Normalize text for keyword extraction (loose view).
        Preserves readable quotes/dashes and case inside quoted phrases.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text suitable for keyword extraction
        """
        loose, _, _ = normalize_views(
            text,
            trim_punctuation=False,  # keep edges for keyword quotes
            canonicalize_vs=False,   # keyword text shouldn't be altered semantically
            casefold_strict=False,
            emoji_policy=self.emoji_policy
        )
        return loose
    
    def normalize_for_slots(self, text: str) -> str:
        """
        Normalize text for regex slot extraction (strict view).
        Identical to current normalize() behavior.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text suitable for regex slot extraction
        """
        _, strict, _ = normalize_views(
            text,
            trim_punctuation=self.trim_punctuation,
            canonicalize_vs=self.canonicalize_vs,
            casefold_strict=self.casefold,
            protect_links=self.protect_links,
            emoji_policy=self.emoji_policy
        )
        return strict 
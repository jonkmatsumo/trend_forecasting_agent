"""
Text Normalization Utilities
Robust Unicode handling and standardization for intent recognition.
Supports dual-view normalization: loose for keywords, strict for regex slots.
"""

import re
import unicodedata
import functools
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse, urljoin

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
# String-based trimming - no regex needed
_TRIM_CHARS = set('.,;:!?(){}[]"\' \t\n\r')

# Legacy regex patterns (kept for backward compatibility but no longer used)
# These have been replaced by string-based parsing
_HTTP_URL_RE = re.compile(r'https?://[^\s]+')
_WWW_URL_RE = re.compile(r'www\.[^\s]+')
_EMAIL_RE = re.compile(r'[^\s@]+@[^\s@]+\.[^\s@]+')
_LINK_HEAD_RE = re.compile(r'^\s*((?:https?://[^\s\.,;:!?(){}[\]"\']+|www\.[^\s\.,;:!?(){}[\]"\']+|[^\s@\.,;:!?(){}[\]"\']+@[^\s@\.,;:!?(){}[\]"\']+\.[^\s@\.,;:!?(){}[\]"\']+))(?=[\s\.,;:!?(){}\[\]"\']|$)')

# Emoji detection pattern
_EMOJI_RE = re.compile(r'[\U0001F300-\U0001FAFF]')

# Smart punctuation mapping
CHAR_MAP = {
    "\u2018": "'", "\u2019": "'",   # ' '
    "\u201C": '"', "\u201D": '"',   # " "
    "\u2013": "-", "\u2014": "-",   # – —
}

# Cache configuration
DEFAULT_CACHE_SIZE = 1024


def is_valid_url(text: str) -> bool:
    """
    Validate if a string is a valid URL using urllib.parse.
    
    Args:
        text: String to validate as URL
        
    Returns:
        True if the string is a valid URL, False otherwise
    """
    try:
        result = urlparse(text)
        # Must have scheme and netloc for a valid URL
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def is_url_with_trailing_punctuation(text: str) -> bool:
    """
    Check if a URL has trailing punctuation that should be protected.
    
    Args:
        text: String to check
        
    Returns:
        True if the text is a URL with trailing punctuation that should be protected
    """
    # Common trailing punctuation that should be protected
    trailing_punct = '.,;:!?(){}[]"\''
    
    # Check if the text ends with trailing punctuation
    if not text or text[-1] not in trailing_punct:
        return False
    
    # Try to find a valid URL by progressively removing trailing punctuation
    for i in range(len(text) - 1, 0, -1):
        if text[i] in trailing_punct:
            candidate = text[:i]
            if is_valid_url(candidate):
                return True
        else:
            break
    
    return False


def is_valid_email(text: str) -> bool:
    """
    Basic email validation.
    
    Args:
        text: String to validate as email
        
    Returns:
        True if the string appears to be a valid email, False otherwise
    """
    # Basic email pattern: local@domain.tld
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, text))


def tokenize_text(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into potential link candidates.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of (token, start_pos, end_pos) tuples
    """
    tokens = []
    current_token = ""
    start_pos = 0
    
    for i, char in enumerate(text):
        if char.isspace():
            if current_token:
                tokens.append((current_token, start_pos, i))
                current_token = ""
        else:
            if not current_token:
                start_pos = i
            current_token += char
    
    # Add final token
    if current_token:
        tokens.append((current_token, start_pos, len(text)))
    
    return tokens


def validate_url_progressive(text: str) -> Tuple[str, int]:
    """
    Validate URL progressively, finding the longest valid prefix.
    
    Args:
        text: Text to validate as URL
        
    Returns:
        Tuple of (valid_url, end_position)
    """
    # Try progressively longer prefixes
    for i in range(len(text), 0, -1):
        candidate = text[:i]
        if is_valid_url(candidate):
            # Check if this is a URL with trailing punctuation that should be protected
            if is_url_with_trailing_punctuation(text):
                # For URLs with trailing punctuation, we want to protect the full text
                return text, len(text)
            return candidate, i
        # Special handling for www. URLs - try with http:// prefix
        elif candidate.startswith('www.') and is_valid_url(f"http://{candidate}"):
            return candidate, i
    
    return "", 0


def validate_email_progressive(text: str) -> Tuple[str, int]:
    """
    Validate email progressively, finding the longest valid prefix.
    
    Args:
        text: Text to validate as email
        
    Returns:
        Tuple of (valid_email, end_position)
    """
    # Try progressively longer prefixes
    for i in range(len(text), 0, -1):
        candidate = text[:i]
        if is_valid_email(candidate):
            return candidate, i
    
    return "", 0


def trim_punctuation_string_based(text: str) -> str:
    """
    Trim leading and trailing punctuation using string operations instead of regex.
    
    Args:
        text: Input text to trim
        
    Returns:
        Text with leading and trailing punctuation removed
    """
    if not text:
        return text
    
    # Find the first non-trim character
    start = 0
    while start < len(text) and text[start] in _TRIM_CHARS:
        start += 1
    
    # Find the last non-trim character
    end = len(text)
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    
    return text[start:end]


def extract_and_validate_links(text: str) -> List[Tuple[str, int, int]]:
    """
    Extract and validate URLs and emails from text using string parsing and progressive validation.
    
    Args:
        text: Input text to extract links from
        
    Returns:
        List of tuples (link_text, start_pos, end_pos) for valid links
    """
    links = []
    tokens = tokenize_text(text)
    
    for token, start_pos, end_pos in tokens:
        # Check if token starts with URL indicators
        if token.startswith(('http://', 'https://', 'www.')):
            valid_url, valid_length = validate_url_progressive(token)
            if valid_url:
                links.append((valid_url, start_pos, start_pos + valid_length))
        
        # Check if token contains a URL (for cases like "!!!https://example.com")
        elif any(indicator in token for indicator in ('http://', 'https://', 'www.')):
            # Find the start of the URL within the token
            for indicator in ('https://', 'http://', 'www.'):
                if indicator in token:
                    url_start = token.find(indicator)
                    url_part = token[url_start:]
                    valid_url, valid_length = validate_url_progressive(url_part)
                    if valid_url:
                        # Adjust start position to account for the URL position within the token
                        actual_start = start_pos + url_start
                        links.append((valid_url, actual_start, actual_start + valid_length))
                    break
        
        # Check for email patterns
        elif '@' in token and '.' in token:
            # Use progressive validation for emails too
            valid_email, valid_length = validate_email_progressive(token)
            if valid_email:
                links.append((valid_email, start_pos, start_pos + valid_length))
    
    return links


def find_longest_link_at_start(text: str) -> Optional[Tuple[str, int, int]]:
    """
    Find the longest valid link at the start of the text using string parsing.
    
    Args:
        text: Input text to check for leading link
        
    Returns:
        Tuple of (link_text, start_pos, end_pos) or None if no valid link found
    """
    # Strip leading whitespace for matching
    stripped_text = text.lstrip()
    if not stripped_text:
        return None
    
    # Find all potential links
    links = extract_and_validate_links(stripped_text)
    
    # Filter to only links that start at the beginning (after whitespace)
    start_links = [link for link in links if link[1] == 0]
    
    if not start_links:
        # If no links start at the beginning, check if there's a link after leading punctuation
        # This handles cases like "!!!https://example.com"
        for link in links:
            # Check if the link starts after some leading punctuation
            if link[1] > 0:
                # Verify that all characters before the link are punctuation/whitespace
                prefix = stripped_text[:link[1]]
                if all(c in _TRIM_CHARS for c in prefix):
                    # Adjust start position to account for original whitespace
                    original_start = len(text) - len(stripped_text) + link[1]
                    return (link[0], original_start, original_start + (link[2] - link[1]))
        return None
    
    # Return the longest link (most comprehensive match)
    longest_link = max(start_links, key=lambda x: x[2] - x[1])
    
    # Adjust start position to account for original whitespace
    original_start = len(text) - len(stripped_text)
    return (longest_link[0], original_start, original_start + longest_link[2])




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
    Uses string-based parsing and progressive validation.
    
    Args:
        s: Input string that may contain a leading URL or email
        
    Returns:
        Tuple of (modified_string, original_entity, trailing_punct) or (original_string, None, None)
    """
    # Use the new string-based link detection
    link_info = find_longest_link_at_start(s)
    if not link_info:
        return s, None, None
    
    entity, start_pos, end_pos = link_info
    
    # Check if there's trailing punctuation after the entity
    remaining = s[end_pos:].lstrip()
    trailing_punct = None
    if remaining and remaining[0] in '.,;:!?(){}[]"\'':
        trailing_punct = remaining[0]
    
    # Use a unique placeholder that won't conflict with normal text
    placeholder = f"⟦{entity}⟧"
    # Replace only the first occurrence at the specific position
    modified = s[:start_pos] + placeholder + s[end_pos:]
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


def _normalize_views_impl(text: str, 
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
        
        # Apply trimming using string-based approach
        s = trim_punctuation_string_based(s)
        
        # Restore protected entity
        if head_entity is not None:
            s = _restore_head_entity(s, head_entity, trailing_punct)
            # Note: Trailing punctuation is preserved when link protection is enabled
            # This allows URLs with trailing punctuation to be protected
    
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
        "cache_hit": False,  # This will be updated by the cached wrapper
        "cache_size": DEFAULT_CACHE_SIZE,
        "string_based_link_protection": True,  # Indicates use of new string-based link protection
    })
    
    return norm_loose, s, stats


def _normalize_views_cached(text: str, 
                           trim_punctuation: bool, 
                           canonicalize_vs: bool, 
                           casefold_strict: bool,
                           protect_links: bool,
                           emoji_policy: str) -> Tuple[str, str, Dict]:
    """
    Cached version of normalize_views implementation.
    
    Note: This function uses a tuple of parameters as the cache key.
    """
    result = _normalize_views_impl(
        text,
        trim_punctuation=trim_punctuation,
        canonicalize_vs=canonicalize_vs,
        casefold_strict=casefold_strict,
        protect_links=protect_links,
        emoji_policy=emoji_policy
    )
    
    # Update cache statistics and create a copy to avoid modifying cached objects
    loose, strict, stats = result
    stats_copy = stats.copy()
    stats_copy["cache_info"] = _normalize_views_cached.cache_info()
    
    return loose, strict, stats_copy

# Apply LRU cache decorator
_normalize_views_cached = functools.lru_cache(maxsize=DEFAULT_CACHE_SIZE)(_normalize_views_cached)


def get_cache_info() -> Dict:
    """
    Get cache statistics for the text normalizer.
    
    Returns:
        Dictionary with cache statistics including hits, misses, and current size
    """
    cache_info = _normalize_views_cached.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
        "hit_rate": cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0.0
    }


def clear_cache() -> None:
    """
    Clear the text normalizer cache.
    
    This can be useful for memory management or testing purposes.
    """
    _normalize_views_cached.cache_clear()


def normalize_views(text: str, 
                   *, 
                   trim_punctuation: bool = True, 
                   canonicalize_vs: bool = True, 
                   casefold_strict: bool = True,
                   protect_links: bool = False,
                   emoji_policy: str = "keep") -> Tuple[str, str, Dict]:
    """
    Normalize text with dual-view approach: loose for keywords, strict for regex slots.
    
    This function uses LRU caching for improved performance when the same text
    is normalized multiple times with the same parameters.
    
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
    # Handle empty text case (not cached)
    if not text:
        stats = get_normalization_stats(text, text)
        stats.update({
            "cache_hit": False,
            "cache_size": DEFAULT_CACHE_SIZE,
            "cache_info": _normalize_views_cached.cache_info()
        })
        return text, text, stats
    
    # Get cache info before call to detect cache hit/miss
    cache_info_before = _normalize_views_cached.cache_info()
    misses_before = cache_info_before.misses
    
    # Use cached implementation
    result = _normalize_views_cached(
        text,
        trim_punctuation,
        canonicalize_vs,
        casefold_strict,
        protect_links,
        emoji_policy
    )
    
    # Check if this was a cache miss by comparing miss count
    cache_info_after = _normalize_views_cached.cache_info()
    was_cache_miss = cache_info_after.misses > misses_before
    
    # Update the cache_hit flag based on whether it was a miss
    loose, strict, stats = result
    # Create a copy to avoid modifying the cached stats object
    stats_copy = stats.copy()
    stats_copy["cache_hit"] = not was_cache_miss
    
    return loose, strict, stats_copy


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
    
    def get_cache_info(self) -> Dict:
        """
        Get cache statistics for this normalizer instance.
        
        Returns:
            Dictionary with cache statistics
        """
        return get_cache_info()
    
    def clear_cache(self) -> None:
        """
        Clear the text normalizer cache.
        
        This can be useful for memory management or testing purposes.
        """
        clear_cache()
    
    def extract_links(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Extract and validate all links from text using robust parsing.
        
        Args:
            text: Input text to extract links from
            
        Returns:
            List of tuples (link_text, start_pos, end_pos) for valid links
        """
        return extract_and_validate_links(text)
    
    def find_link_at_start(self, text: str) -> Optional[Tuple[str, int, int]]:
        """
        Find the longest valid link at the start of the text.
        
        Args:
            text: Input text to check for leading link
            
        Returns:
            Tuple of (link_text, start_pos, end_pos) or None if no valid link found
        """
        return find_longest_link_at_start(text)
    
    def is_valid_url(self, text: str) -> bool:
        """
        Validate if a string is a valid URL using urllib.parse.
        
        Args:
            text: String to validate as URL
            
        Returns:
            True if the string is a valid URL, False otherwise
        """
        return is_valid_url(text)
    
    def is_valid_email(self, text: str) -> bool:
        """
        Basic email validation.
        
        Args:
            text: String to validate as email
            
        Returns:
            True if the string appears to be a valid email, False otherwise
        """
        return is_valid_email(text)
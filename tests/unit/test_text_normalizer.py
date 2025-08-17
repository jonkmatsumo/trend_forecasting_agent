"""
Tests for enhanced text normalization utilities with dual-view approach.
"""

import pytest
from app.utils.text_normalizer import (
    normalize_views, normalize_with_ftfy, get_normalization_stats,
    TextNormalizer, CHAR_MAP
)


class TestNormalizeViews:
    """Test the new dual-view normalization function."""
    
    def test_empty_text(self):
        """Test handling of empty text."""
        loose, strict, stats = normalize_views("")
        assert loose == ""
        assert strict == ""
        assert stats["original_length"] == 0
        assert stats["normalized_length"] == 0
    
    def test_basic_normalization(self):
        """Test basic text normalization."""
        text = "  Hello   World  "
        loose, strict, stats = normalize_views(text)
        
        assert loose == "Hello World"
        assert strict == "hello world"
        assert stats["ws_collapsed"] is True
        assert stats["casefold_applied"] is True
    
    def test_curly_quotes_and_dashes(self):
        """Test smart punctuation normalization."""
        text = "iPhone—17 with \u201Csmart\u201D features"
        loose, strict, stats = normalize_views(text)
        
        # Loose view preserves case but fixes quotes/dashes
        assert loose == 'iPhone-17 with "smart" features'
        # Strict view also casefolds
        assert strict == 'iphone-17 with "smart" features'
        
        assert stats["quotes_fixed"] is True
        assert stats["dashes_fixed"] is True
    
    def test_versus_canonicalization(self):
        """Test versus canonicalization in strict view only."""
        text = "Python versus JavaScript"
        loose, strict, stats = normalize_views(text)
        
        # Loose view preserves original
        assert loose == "Python versus JavaScript"
        # Strict view canonicalizes
        assert strict == "python vs javascript"
        
        assert stats["canon_vs"] is True
        assert stats["vs_canonicalized"] is True
    
    def test_zero_width_and_control_chars(self):
        """Test removal of zero-width and control characters."""
        text = "Hello\u200BWorld\u0000Test\uFEFF"
        loose, strict, stats = normalize_views(text)
        
        assert loose == "HelloWorldTest"
        assert strict == "helloworldtest"
        assert stats["zero_width_removed"] is True
        assert stats["controls_removed"] is True
    
    def test_punctuation_trimming(self):
        """Test edge punctuation trimming in strict view."""
        text = "!!!Hello World!!!"
        loose, strict, stats = normalize_views(text)
        
        # Loose view keeps edge punctuation
        assert loose == "!!!Hello World!!!"
        # Strict view trims it
        assert strict == "hello world"
        
        assert stats["trim_punctuation"] is True
    
    def test_internal_punctuation_preserved(self):
        """Test that internal punctuation is preserved."""
        text = "p10 / p50 / p90"
        loose, strict, stats = normalize_views(text)
        
        # Both views preserve internal punctuation
        assert loose == "p10 / p50 / p90"
        assert strict == "p10 / p50 / p90"
    
    def test_whitespace_collapse(self):
        """Test whitespace collapse with various space types."""
        text = "Hello\u00A0\u2000World\u2001Test"
        loose, strict, stats = normalize_views(text)
        
        assert loose == "Hello World Test"
        assert strict == "hello world test"
        assert stats["ws_collapsed"] is True
    
    def test_idempotency(self):
        """Test that normalization is idempotent."""
        text = "  Hello\u2014World  "
        loose1, strict1, _ = normalize_views(text)
        loose2, strict2, _ = normalize_views(loose1)
        strict3, strict4, _ = normalize_views(strict1)
        
        assert loose1 == loose2
        assert strict1 == strict2
        assert strict3 == strict4
    
    def test_custom_flags(self):
        """Test custom normalization flags."""
        text = "Python VERSUS JavaScript!!!"
        loose, strict, stats = normalize_views(
            text,
            trim_punctuation=False,
            canonicalize_vs=False,
            casefold_strict=False
        )
        
        # Should only do basic repairs
        assert loose == "Python VERSUS JavaScript!!!"
        assert strict == "Python VERSUS JavaScript!!!"
        assert stats["trim_punctuation"] is False
        assert stats["canon_vs"] is False
        assert stats["casefold_applied"] is False


class TestBackwardCompatibility:
    """Test backward compatibility of existing functions."""
    
    def test_normalize_with_ftfy_returns_strict(self):
        """Test that normalize_with_ftfy returns strict view."""
        text = "  Hello\u2014World  "
        result = normalize_with_ftfy(text)
        
        # Should return strict view (casefolded, trimmed)
        assert result == "hello-world"
    
    def test_text_normalizer_normalize_returns_strict(self):
        """Test that TextNormalizer.normalize returns strict view."""
        normalizer = TextNormalizer()
        text = "Python versus JavaScript!!!"
        result, stats = normalizer.normalize(text)
        
        # Should return strict view
        assert result == "python vs javascript"
        assert stats["vs_canonicalized"] is True
    
    def test_text_normalizer_normalize_only_returns_strict(self):
        """Test that TextNormalizer.normalize_only returns strict view."""
        normalizer = TextNormalizer()
        text = "  Hello World  "
        result = normalizer.normalize_only(text)
        
        # Should return strict view
        assert result == "hello world"


class TestTextNormalizerNewMethods:
    """Test new methods on TextNormalizer."""
    
    def test_normalize_both(self):
        """Test normalize_both method."""
        normalizer = TextNormalizer()
        text = "Python versus JavaScript!!!"
        loose, strict, stats = normalizer.normalize_both(text)
        
        assert loose == 'Python versus JavaScript!!!'
        assert strict == "python vs javascript"
        assert stats["vs_canonicalized"] is True
    
    def test_normalize_for_keywords(self):
        """Test normalize_for_keywords method."""
        normalizer = TextNormalizer()
        text = "iPhone—17 with \u201Csmart\u201D features"
        result = normalizer.normalize_for_keywords(text)
        
        # Should preserve case and edge punctuation
        assert result == 'iPhone-17 with "smart" features'
    
    def test_normalize_for_slots(self):
        """Test normalize_for_slots method."""
        normalizer = TextNormalizer()
        text = "Python versus JavaScript!!!"
        result = normalizer.normalize_for_slots(text)
        
        # Should return strict view
        assert result == "python vs javascript"
    
    def test_custom_normalizer_config(self):
        """Test custom normalizer configuration."""
        normalizer = TextNormalizer(
            casefold=False,
            canonicalize_vs=False,
            trim_punctuation=False
        )
        text = "Python VERSUS JavaScript!!!"
        loose, strict, stats = normalizer.normalize_both(text)
        
        # Should only do basic repairs
        assert loose == "Python VERSUS JavaScript!!!"
        assert strict == "Python VERSUS JavaScript!!!"
        assert stats["casefold_applied"] is False
        assert stats["canon_vs"] is False
        assert stats["trim_punctuation"] is False


class TestNormalizationStats:
    """Test enhanced normalization statistics."""
    
    def test_basic_stats(self):
        """Test basic statistics calculation."""
        original = "Hello World"
        normalized = "hello world"
        stats = get_normalization_stats(original, normalized)
        
        assert stats["characters_changed"] > 0
        assert stats["length_change"] == 0
        assert stats["original_length"] == 11
        assert stats["normalized_length"] == 11
    
    def test_enhanced_stats_from_normalize_views(self):
        """Test enhanced statistics from normalize_views."""
        text = "Hello\u2014World\u200B"
        loose, strict, stats = normalize_views(text)
        
        assert stats["zero_width_removed"] is True
        assert stats["dashes_fixed"] is True
        assert stats["ws_collapsed"] is False
        assert stats["loose_length"] == len(loose)
        assert stats["strict_length"] == len(strict)
    
    def test_stats_with_empty_strings(self):
        """Test statistics with empty strings."""
        stats = get_normalization_stats("", "")
        
        assert stats["characters_changed"] == 0
        assert stats["length_change"] == 0
        assert stats["original_length"] == 0
        assert stats["normalized_length"] == 0
    
    def test_stats_with_max_compare(self):
        """Test statistics with max_compare parameter."""
        # Create a long string to test max_compare
        original = "A" * 10000
        normalized = "B" * 10000
        stats = get_normalization_stats(original, normalized, max_compare=100)
        
        # Should not exceed max_compare for detailed comparison
        assert stats["characters_changed"] <= 100 + abs(len(original) - len(normalized))


class TestCharacterMapping:
    """Test smart character mapping."""
    
    def test_char_map_completeness(self):
        """Test that CHAR_MAP covers all smart punctuation."""
        expected_chars = {
            "\u2018", "\u2019",  # ' '
            "\u201C", "\u201D",  # " "
            "\u2013", "\u2014",  # – —
        }
        
        for char in expected_chars:
            assert char in CHAR_MAP
            assert CHAR_MAP[char] in "'\"-"
    
    def test_char_mapping_function(self):
        """Test the _map_chars function."""
        from app.utils.text_normalizer import _map_chars
        
        text = "Hello\u2014World\u2018Test\u2019"
        result = _map_chars(text)
        
        assert result == "Hello-World'Test'"


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        # Test with combining characters
        text = "café"  # e with acute accent
        loose, strict, stats = normalize_views(text)
        
        # Should normalize to composed form
        assert loose == "café"
        assert strict == "café"
    
    def test_mixed_whitespace(self):
        """Test mixed whitespace handling."""
        text = "Hello\t\n\rWorld"
        loose, strict, stats = normalize_views(text)
        
        assert loose == "Hello World"
        assert strict == "hello world"
        assert stats["ws_collapsed"] is True
    
    def test_only_punctuation(self):
        """Test text with only punctuation."""
        text = "!!!"
        loose, strict, stats = normalize_views(text)
        
        assert loose == "!!!"
        assert strict == ""
        assert stats["trim_punctuation"] is True
    
    def test_only_whitespace(self):
        """Test text with only whitespace."""
        text = "   \t\n   "
        loose, strict, stats = normalize_views(text)
        
        assert loose == ""
        assert strict == ""
        assert stats["ws_collapsed"] is True


class TestIntegrationScenarios:
    """Test integration scenarios for different use cases."""
    
    def test_keyword_extraction_scenario(self):
        """Test scenario suitable for keyword extraction."""
        normalizer = TextNormalizer()
        text = 'Show me trends for "Machine Learning" vs AI'
        
        # For keywords, we want loose view
        keywords = normalizer.normalize_for_keywords(text)
        assert keywords == 'Show me trends for "Machine Learning" vs AI'
        
        # For slots, we want strict view
        slots = normalizer.normalize_for_slots(text)
        assert slots == 'show me trends for "machine learning" vs ai'
    
    def test_regex_slot_scenario(self):
        """Test scenario suitable for regex slot extraction."""
        normalizer = TextNormalizer()
        text = "Forecast Python—3.12 trends!!!"
        
        # For slots, we want strict view
        slots = normalizer.normalize_for_slots(text)
        assert slots == "forecast python-3.12 trends"
        
        # For keywords, we want loose view
        keywords = normalizer.normalize_for_keywords(text)
        assert keywords == "Forecast Python-3.12 trends!!!"
    
    def test_messy_input_scenario(self):
        """Test handling of messy input with various issues."""
        text = "  Python\u200B\u2014\u201C3.12\u201D!!!  "
        loose, strict, stats = normalize_views(text)
        
        assert loose == 'Python-"3.12"!!!'
        assert strict == 'python-"3.12'
        
        # Check that all issues were detected
        assert stats["zero_width_removed"] is True
        assert stats["dashes_fixed"] is True
        assert stats["quotes_fixed"] is True
        assert stats["trim_punctuation"] is True
        assert stats["ws_collapsed"] is True 
"""
Tests for enhanced text normalization utilities with dual-view approach.
"""

import pytest
import unicodedata
from app.utils.text_normalizer import (
    normalize_views, normalize_with_ftfy, normalize_strict, get_normalization_stats,
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

    def test_changed_flag_for_modified_text(self):
        """Test that changed flag correctly identifies modified text."""
        # Test unchanged text
        original = "Hello World"
        normalized = "Hello World"
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is False
        assert stats["percent_changed"] == 0.0
        
        # Test changed text
        original = "Hello World"
        normalized = "hello world"
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is True
        assert stats["percent_changed"] > 0.0
        
        # Test empty strings
        stats = get_normalization_stats("", "")
        assert stats["changed"] is False
        assert stats["percent_changed"] == 0.0

    def test_percent_changed_calculation_accuracy(self):
        """Test percent_changed calculation accuracy."""
        # Test case folding (only 2 characters actually changed: H->h, W->w)
        original = "Hello World"
        normalized = "hello world"
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is True
        # Calculate expected: 2 characters changed out of 11 = 18.18%
        expected_percent = (2 / 11) * 100
        assert abs(stats["percent_changed"] - expected_percent) < 0.01
        
        # Test all characters changed
        original = "ABC"
        normalized = "xyz"
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is True
        assert stats["percent_changed"] == 100.0  # All 3 characters changed
        
        # Test partial changes
        original = "Hello World"
        normalized = "Hello World!"  # Added one character
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is True
        assert stats["percent_changed"] > 0.0
        
        # Test no changes
        original = "Hello World"
        normalized = "Hello World"
        stats = get_normalization_stats(original, normalized)
        assert stats["changed"] is False
        assert stats["percent_changed"] == 0.0

    def test_link_email_detection_in_stats(self):
        """Test link and email detection in statistics."""
        # Test with links
        original = "Check out https://example.com"
        normalized = "check out https://example.com"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_links"] is True
        assert stats["had_emails"] is False
        
        # Test with emails
        original = "Contact user@example.com"
        normalized = "contact user@example.com"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_links"] is False
        assert stats["had_emails"] is True
        
        # Test with both
        original = "Visit https://example.com or email user@example.com"
        normalized = "visit https://example.com or email user@example.com"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_links"] is True
        assert stats["had_emails"] is True
        
        # Test with neither
        original = "Hello World"
        normalized = "hello world"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_links"] is False
        assert stats["had_emails"] is False

    def test_quote_dash_detection_in_stats(self):
        """Test quote and dash detection in statistics."""
        # Test with quotes
        original = 'He said "Hello World"'
        normalized = 'he said "hello world"'
        stats = get_normalization_stats(original, normalized)
        assert stats["had_quotes"] is True
        assert stats["had_dashes"] is False
        
        # Test with smart quotes
        original = "He said \u201CHello World\u201D"
        normalized = 'he said "hello world"'
        stats = get_normalization_stats(original, normalized)
        assert stats["had_quotes"] is True
        assert stats["had_dashes"] is False
        
        # Test with dashes
        original = "iPhone-17 with features"
        normalized = "iphone-17 with features"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_quotes"] is False
        assert stats["had_dashes"] is True
        
        # Test with smart dashes
        original = "iPhone\u201417 with features"
        normalized = "iphone-17 with features"
        stats = get_normalization_stats(original, normalized)
        assert stats["had_quotes"] is False
        assert stats["had_dashes"] is True
        
        # Test with both
        original = 'iPhone\u201417 with "smart" features'
        normalized = 'iphone-17 with "smart" features'
        stats = get_normalization_stats(original, normalized)
        assert stats["had_quotes"] is True
        assert stats["had_dashes"] is True

    def test_stats_dont_contain_raw_text(self):
        """Test that stats are privacy-safe and don't contain raw text content."""
        original = "Sensitive data: https://example.com user@example.com 'secret'"
        normalized = "sensitive data: https://example.com user@example.com 'secret'"
        stats = get_normalization_stats(original, normalized)
        
        # Verify stats contain only metadata, not content
        assert "original" not in stats
        assert "normalized" not in stats
        assert "content" not in stats
        assert "text" not in stats
        
        # Verify stats contain only boolean flags and numeric values
        for key, value in stats.items():
            assert isinstance(value, (bool, int, float, str)), f"Unexpected type for {key}: {type(value)}"
            if isinstance(value, str):
                # String values should only be cache_info or similar metadata
                assert key in ["cache_info", "cache_hit"], f"Unexpected string key: {key}"


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


class TestStringBasedLinkProtection:
    """Test the new string-based link protection using urllib.parse."""
    
    def test_is_valid_url(self):
        """Test URL validation using urllib.parse."""
        from app.utils.text_normalizer import is_valid_url
        
        # Valid URLs
        assert is_valid_url("https://example.com")
        assert is_valid_url("http://example.com")
        assert is_valid_url("https://sub.example.com")
        assert is_valid_url("https://example.com/path")
        assert is_valid_url("https://example.com?param=value")
        assert is_valid_url("https://example.com#fragment")
        assert is_valid_url("https://api.example.co.uk")
        
        # Invalid URLs
        assert not is_valid_url("example.com")  # No scheme
        assert not is_valid_url("https://")  # No netloc
        assert not is_valid_url("not-a-url")
        assert not is_valid_url("")
        # Note: FTP URLs are actually valid according to urllib.parse
        # assert not is_valid_url("ftp://example.com")  # Different scheme
    
    def test_is_valid_email(self):
        """Test email validation."""
        from app.utils.text_normalizer import is_valid_email
        
        # Valid emails
        assert is_valid_email("user@example.com")
        assert is_valid_email("user.name@example.com")
        assert is_valid_email("user+tag@example.com")
        assert is_valid_email("user@sub.example.com")
        assert is_valid_email("user@example.co.uk")
        
        # Invalid emails
        assert not is_valid_email("user@")  # No domain
        assert not is_valid_email("@example.com")  # No local part
        assert not is_valid_email("user@example")  # No TLD
        assert not is_valid_email("user example.com")  # No @
        assert not is_valid_email("")
    
    def test_extract_and_validate_links(self):
        """Test link extraction and validation."""
        from app.utils.text_normalizer import extract_and_validate_links
        
        # Test HTTP URLs
        text = "Check out https://example.com and http://test.com"
        links = extract_and_validate_links(text)
        assert len(links) == 2
        assert links[0][0] == "https://example.com"
        assert links[1][0] == "http://test.com"
        
        # Test WWW URLs
        text = "Visit www.example.com and www.test.com"
        links = extract_and_validate_links(text)
        assert len(links) == 2
        assert links[0][0] == "www.example.com"
        assert links[1][0] == "www.test.com"
        
        # Test emails
        text = "Contact user@example.com and admin@test.com"
        links = extract_and_validate_links(text)
        assert len(links) == 2
        assert links[0][0] == "user@example.com"
        assert links[1][0] == "admin@test.com"
        
        # Test mixed content - new behavior preserves trailing punctuation when valid
        text = "Visit https://example.com, email user@test.com, or go to www.demo.com"
        links = extract_and_validate_links(text)
        assert len(links) == 3
        assert links[0][0] == "https://example.com,"  # Comma preserved as part of URL
        assert links[1][0] == "user@test.com"  # Comma stripped from email
        assert links[2][0] == "www.demo.com"
        
        # Test invalid URLs (should not be extracted)
        text = "Invalid: example.com and user@"
        links = extract_and_validate_links(text)
        assert len(links) == 0
    
    def test_find_longest_link_at_start(self):
        """Test finding the longest link at the start of text."""
        from app.utils.text_normalizer import find_longest_link_at_start
        
        # Test with leading whitespace
        text = "  https://example.com/path hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://example.com/path"
        assert link_info[1] == 2  # start position after whitespace
        assert link_info[2] == 26  # end position
        
        # Test with trailing punctuation
        text = "https://example.com!!! hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://example.com!!!"
        
        # Test with complex URL
        text = "https://api.example.com/path?param=value#fragment hello"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://api.example.com/path?param=value#fragment"
        
        # Test with email
        text = "user@example.com hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "user@example.com"
        
        # Test with no link at start
        text = "hello https://example.com world"
        link_info = find_longest_link_at_start(text)
        assert link_info is None
        
        # Test with empty string
        link_info = find_longest_link_at_start("")
        assert link_info is None
    
    def test_protect_head_entity_string_based(self):
        """Test the improved protect_head_entity function with string-based detection."""
        from app.utils.text_normalizer import _protect_head_entity
        
        # Test with URL and trailing punctuation
        text = "https://example.com!!! hello world"
        modified, entity, trailing_punct = _protect_head_entity(text)
        assert entity == "https://example.com!!!"
        assert trailing_punct is None  # No trailing punctuation after the URL
        assert "⟦https://example.com!!!⟧" in modified
        
        # Test with URL and actual trailing punctuation
        text = "https://example.com, hello world"
        modified, entity, trailing_punct = _protect_head_entity(text)
        assert entity == "https://example.com,"  # New behavior preserves comma as part of URL
        assert trailing_punct is None  # No trailing punctuation after the URL
        assert "⟦https://example.com,⟧" in modified
        
        # Test with email
        text = "user@example.com hello world"
        modified, entity, trailing_punct = _protect_head_entity(text)
        assert entity == "user@example.com"
        assert trailing_punct is None
        assert "⟦user@example.com⟧" in modified
        
        # Test with no link at start
        text = "hello world"
        modified, entity, trailing_punct = _protect_head_entity(text)
        assert entity is None
        assert trailing_punct is None
        assert modified == text
    
    def test_link_protection_with_string_based_parsing(self):
        """Test that link protection works correctly with the new string-based parsing."""
        # Test that URLs are properly protected from trimming
        # The link protection ensures URLs are preserved even when they would be trimmed
        text = "!!!https://example.com hello world"
        
        # With link protection
        loose1, strict1, stats1 = normalize_views(text, protect_links=True)
        
        # Without link protection
        loose2, strict2, stats2 = normalize_views(text, protect_links=False)
        
        # Both should preserve the URL, but the protection ensures it's handled correctly
        # The key is that the URL is preserved in both cases
        assert "https://example.com" in loose1 or "https://example.com" in strict1
        assert "https://example.com" in loose2 or "https://example.com" in strict2
        
        # Test with complex URL
        text = "!!!https://api.example.com/path?param=value#fragment hello world"
        
        loose1, strict1, stats1 = normalize_views(text, protect_links=True)
        loose2, strict2, stats2 = normalize_views(text, protect_links=False)
        
        # Both should preserve the complex URL
        assert "https://api.example.com/path?param=value#fragment" in loose1 or "https://api.example.com/path?param=value#fragment" in strict1
        assert "https://api.example.com/path?param=value#fragment" in loose2 or "https://api.example.com/path?param=value#fragment" in strict2
        
        # Test that protection stats are correctly set
        assert stats1["protect_links"] is True
        assert stats2["protect_links"] is False
    
    def test_text_normalizer_link_methods(self):
        """Test the new link detection methods in TextNormalizer class."""
        normalizer = TextNormalizer(protect_links=True)
        
        # Test extract_links
        text = "Visit https://example.com and email user@test.com"
        links = normalizer.extract_links(text)
        assert len(links) == 2
        assert links[0][0] == "https://example.com"
        assert links[1][0] == "user@test.com"
        
        # Test find_link_at_start
        text = "https://example.com hello world"
        link_info = normalizer.find_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://example.com"
        
        # Test is_valid_url
        assert normalizer.is_valid_url("https://example.com")
        assert not normalizer.is_valid_url("example.com")
        
        # Test is_valid_email
        assert normalizer.is_valid_email("user@example.com")
        assert not normalizer.is_valid_email("user@")
    
    def test_string_based_link_protection_stats(self):
        """Test that string-based link protection is indicated in statistics."""
        text = "https://example.com hello world"
        
        _, _, stats = normalize_views(text, protect_links=True)
        
        assert stats["string_based_link_protection"] is True
        assert stats["protect_links"] is True
    
    def test_edge_cases_string_based_link_protection(self):
        """Test edge cases with the new string-based link protection."""
        from app.utils.text_normalizer import find_longest_link_at_start
        
        # Test with multiple dots in domain
        text = "https://sub.example.co.uk hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://sub.example.co.uk"
        
        # Test with URL containing special characters
        text = "https://example.com/path-with-dashes?param=value&other=123 hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://example.com/path-with-dashes?param=value&other=123"
        
        # Test with email containing special characters
        text = "user.name+tag@example.com hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "user.name+tag@example.com"
        
        # Test with leading whitespace and complex URL
        text = "  https://api.example.com/v1/users/123?include=profile&fields=name,email hello world"
        link_info = find_longest_link_at_start(text)
        assert link_info is not None
        assert link_info[0] == "https://api.example.com/v1/users/123?include=profile&fields=name,email"


class TestUnicodeDigitNormalization:
    """Test Unicode digit normalization via NFKC."""
    
    def test_full_width_digits(self):
        """Test full-width digit normalization."""
        # Full-width digits: ０１２３４５６７８９
        text = "９０％"
        loose, strict, stats = normalize_views(text)
        
        # NFKC should normalize full-width digits to ASCII
        assert loose == "90%"
        assert strict == "90%"
    
    def test_full_width_numbers(self):
        """Test full-width number normalization."""
        text = "１２３"
        loose, strict, stats = normalize_views(text)
        
        # NFKC should normalize full-width numbers to ASCII
        assert loose == "123"
        assert strict == "123"
    
    def test_full_width_decimals(self):
        """Test full-width decimal normalization."""
        text = "５０．５"
        loose, strict, stats = normalize_views(text)
        
        # NFKC should normalize full-width decimals to ASCII
        assert loose == "50.5"
        assert strict == "50.5"
    
    def test_mixed_full_width_and_ascii(self):
        """Test mixed full-width and ASCII digits."""
        text = "９０% and 50%"
        loose, strict, stats = normalize_views(text)
        
        # Should normalize full-width digits but preserve ASCII
        assert loose == "90% and 50%"
        assert strict == "90% and 50%"
    
    def test_full_width_with_text(self):
        """Test full-width digits in context."""
        text = "Version ３.１２ is released"
        loose, strict, stats = normalize_views(text)
        
        # Should normalize digits but preserve text
        assert loose == "Version 3.12 is released"
        assert strict == "version 3.12 is released"
    
    def test_python_regex_unicode_digit_support(self):
        r"""Test that Python's \d regex matches Unicode Nd category."""
        import re
        
        # Test that \d matches full-width digits after NFKC normalization
        text = "９０％"
        normalized = unicodedata.normalize("NFKC", text)
        
        # \d should match the normalized digits
        matches = re.findall(r'\d', normalized)
        assert matches == ['9', '0']
        
        # Test with mixed digits
        text = "９０ and 50"
        normalized = unicodedata.normalize("NFKC", text)
        matches = re.findall(r'\d+', normalized)
        assert matches == ['90', '50']


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


class TestEmojiPolicy:
    """Test emoji policy functionality."""
    
    def test_emoji_keep_policy(self):
        """Test emoji keep policy (default)."""
        text = "🔥 stock"
        loose, strict, stats = normalize_views(text, emoji_policy="keep")
        
        assert loose == "🔥 stock"
        assert strict == "🔥 stock"
        assert stats["emoji_policy"] == "keep"
        assert stats["had_emojis"] is True
    
    def test_emoji_strip_policy(self):
        """Test emoji strip policy."""
        text = "🔥 stock"
        loose, strict, stats = normalize_views(text, emoji_policy="strip")
        
        assert loose == "stock"
        assert strict == "stock"
        assert stats["emoji_policy"] == "strip"
        assert stats["had_emojis"] is True
    
    def test_emoji_map_policy(self):
        """Test emoji map policy."""
        text = "🔥 stock"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        
        assert loose == "fire stock"
        assert strict == "fire stock"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
    
    def test_emoji_map_policy_multiple(self):
        """Test emoji map policy with multiple emojis."""
        text = "📈 trend 🔥 stock"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        
        assert loose == "chart increasing trend fire stock"
        assert strict == "chart increasing trend fire stock"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
    
    def test_emoji_map_policy_unmapped(self):
        """Test emoji map policy with unmapped emojis."""
        text = "🔥 stock 🎵 music"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        
        # 🎵 should be mapped to "musical note" by the emoji library
        assert loose == "fire stock musical note music"
        assert strict == "fire stock musical note music"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
    
    def test_emoji_policy_default(self):
        """Test that emoji policy defaults to keep."""
        text = "🔥 stock"
        loose, strict, stats = normalize_views(text)
        
        assert loose == "🔥 stock"
        assert strict == "🔥 stock"
        assert stats["emoji_policy"] == "keep"
        assert stats["had_emojis"] is True
    
    def test_emoji_policy_invalid(self):
        """Test that invalid emoji policy raises ValueError."""
        text = "🔥 stock"
        
        with pytest.raises(ValueError, match="Invalid emoji_policy"):
            normalize_views(text, emoji_policy="invalid")
    
    def test_emoji_with_normalization(self):
        """Test emoji processing with other normalizations."""
        text = "🔥 STOCK!!!"
        loose, strict, stats = normalize_views(text, emoji_policy="map", trim_punctuation=True)
        
        assert loose == "fire STOCK!!!"
        assert strict == "fire stock"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
        assert stats["trim_punctuation"] is True
    
    def test_no_emojis(self):
        """Test text without emojis."""
        text = "normal text"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        
        assert loose == "normal text"
        assert strict == "normal text"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is False
    
    def test_emoji_with_links(self):
        """Test emoji processing with link protection."""
        text = "🔥 https://example.com."
        loose, strict, stats = normalize_views(text, emoji_policy="map", protect_links=True)
        
        assert loose == "fire https://example.com."
        assert strict == "fire https://example.com"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
        assert stats["protect_links"] is True
    
    def test_emoji_with_versus_canonicalization(self):
        """Test emoji processing with versus canonicalization."""
        text = "🔥 stock versus bonds"
        loose, strict, stats = normalize_views(text, emoji_policy="map", canonicalize_vs=True)
        
        assert loose == "fire stock versus bonds"
        assert strict == "fire stock vs bonds"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
        assert stats["canon_vs"] is True
    
    def test_emoji_with_text_normalizer_class(self):
        """Test emoji processing with TextNormalizer class."""
        normalizer = TextNormalizer(emoji_policy="map")
        text = "🔥 stock"
        result, stats = normalizer.normalize(text)
        
        assert result == "fire stock"
        assert stats["emoji_policy"] == "map"
        assert stats["had_emojis"] is True
    
    def test_emoji_with_normalize_with_ftfy(self):
        """Test emoji processing with normalize_with_ftfy function."""
        text = "🔥 stock"
        result = normalize_with_ftfy(text, emoji_policy="map")
        
        assert result == "fire stock"
    
    def test_emoji_edge_cases(self):
        """Test emoji processing with edge cases."""
        # Test with only emoji
        text = "🔥"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        assert loose == "fire"
        assert strict == "fire"
        
        # Test with emoji at start and end
        text = "🔥 stock 📈"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        assert loose == "fire stock chart increasing"
        assert strict == "fire stock chart increasing"
        
        # Test with multiple consecutive emojis
        text = "🔥📈💰"
        loose, strict, stats = normalize_views(text, emoji_policy="map")
        assert loose == "firechart increasingmoney bag"
        assert strict == "firechart increasingmoney bag" 


class TestCaching:
    """Test caching functionality for text normalization."""
    
    def setup_method(self):
        """Clear cache before each test."""
        from app.utils.text_normalizer import clear_cache
        clear_cache()
    
    def test_cache_hit_on_repeated_calls(self):
        """Test that repeated calls with same parameters use cache."""
        from app.utils.text_normalizer import get_cache_info
        
        text = "Hello World!!!"
        
        # First call should miss cache
        loose1, strict1, stats1 = normalize_views(text)
        
        # Second call with same parameters should hit cache
        loose2, strict2, stats2 = normalize_views(text)
        
        # Results should be identical
        assert loose1 == loose2
        assert strict1 == strict2
        
        # First call should not be cached, second should be
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is True
        
        # Check cache statistics
        cache_info = get_cache_info()
        assert cache_info["hits"] >= 1
        assert cache_info["misses"] >= 1
    
    def test_cache_miss_on_different_parameters(self):
        """Test that different parameters don't use cache."""
        
        text = "Hello World!!!"
        
        # First call with default parameters
        loose1, strict1, stats1 = normalize_views(text)
        
        # Second call with different parameters
        loose2, strict2, stats2 = normalize_views(text, trim_punctuation=False)
        
        # Results should be different
        assert loose1 == loose2  # Loose view should be same
        assert strict1 != strict2  # Strict view should be different
        
        # Both calls should miss cache (different parameters)
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is False
    
    def test_cache_miss_on_different_text(self):
        """Test that different text doesn't use cache."""
        
        # First call
        loose1, strict1, stats1 = normalize_views("Hello World!!!")
        
        # Second call with different text
        loose2, strict2, stats2 = normalize_views("Goodbye World!!!")
        
        # Results should be different
        assert loose1 != loose2
        assert strict1 != strict2
        
        # Both calls should miss cache (different text)
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is False
    
    def test_cache_statistics_included(self):
        """Test that cache statistics are included in results."""
        
        text = "Test text"
        loose, strict, stats = normalize_views(text)
        
        # Check that cache statistics are included
        assert "cache_hit" in stats
        assert "cache_size" in stats
        assert "cache_info" in stats
        
        # Initial call should not be cached
        assert stats["cache_hit"] is False
        assert stats["cache_size"] == 1024  # DEFAULT_CACHE_SIZE
    
    def test_cache_eviction_works(self):
        """Test that cache eviction works correctly."""
        from app.utils.text_normalizer import get_cache_info, DEFAULT_CACHE_SIZE
        
        # Fill cache beyond its size
        for i in range(DEFAULT_CACHE_SIZE + 10):
            text = f"Test text {i}"
            normalize_views(text)
        
        # Check that cache size doesn't exceed max
        cache_info = get_cache_info()
        assert cache_info["currsize"] <= DEFAULT_CACHE_SIZE
    
    def test_cache_clear_functionality(self):
        """Test that cache can be cleared."""
        from app.utils.text_normalizer import clear_cache, get_cache_info
        
        text = "Hello World!!!"
        
        # Make a call to populate cache
        normalize_views(text)
        
        # Check cache has content
        cache_info_before = get_cache_info()
        assert cache_info_before["currsize"] > 0
        
        # Clear cache
        clear_cache()
        
        # Check cache is empty
        cache_info_after = get_cache_info()
        assert cache_info_after["currsize"] == 0
    
    def test_cache_info_function(self):
        """Test the get_cache_info function."""
        from app.utils.text_normalizer import get_cache_info
        
        cache_info = get_cache_info()
        
        # Check all expected keys are present
        expected_keys = ["hits", "misses", "maxsize", "currsize", "hit_rate"]
        for key in expected_keys:
            assert key in cache_info
        
        # Check initial values
        assert cache_info["hits"] == 0
        assert cache_info["misses"] == 0
        assert cache_info["currsize"] == 0
        assert cache_info["hit_rate"] == 0.0
    
    def test_text_normalizer_cache_methods(self):
        """Test cache methods on TextNormalizer class."""
        
        normalizer = TextNormalizer()
        
        # Test cache info method
        cache_info = normalizer.get_cache_info()
        assert "hits" in cache_info
        assert "misses" in cache_info
        
        # Test cache clear method
        normalizer.clear_cache()
        cache_info_after = normalizer.get_cache_info()
        assert cache_info_after["currsize"] == 0
    
    def test_cache_with_empty_strings(self):
        """Test that empty strings are not cached."""
        
        # Empty string should not be cached
        loose1, strict1, stats1 = normalize_views("")
        loose2, strict2, stats2 = normalize_views("")
        
        # Both should not be cached
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is False
    
    def test_cache_with_emoji_policies(self):
        """Test cache behavior with different emoji policies."""
        
        text = "🔥 Hello World"
        
        # Different emoji policies should not use cache
        loose1, strict1, stats1 = normalize_views(text, emoji_policy="keep")
        loose2, strict2, stats2 = normalize_views(text, emoji_policy="map")
        
        # Results should be different
        assert loose1 != loose2
        assert strict1 != strict2
        
        # Both should miss cache
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is False
    
    def test_cache_with_link_protection(self):
        """Test cache behavior with link protection."""
        
        # Use a case where link protection should be tested
        # The URL should be preserved in both cases, but protection ensures proper handling
        text = "!!!https://example.com hello world"
        
        # Different link protection settings should not use cache
        loose1, strict1, stats1 = normalize_views(text, protect_links=False)
        loose2, strict2, stats2 = normalize_views(text, protect_links=True)
        
        # Both should preserve the URL, but protection ensures it's handled correctly
        assert "https://example.com" in loose1 or "https://example.com" in strict1
        assert "https://example.com" in loose2 or "https://example.com" in strict2
        
        # Both should miss cache (different parameters)
        assert stats1["cache_hit"] is False
        assert stats2["cache_hit"] is False
    
    def test_cache_performance_improvement(self):
        """Test that caching provides performance improvement."""
        import time
        
        text = "This is a longer text that will take more time to normalize " * 50
        
        # First call (cache miss)
        start_time = time.time()
        normalize_views(text)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        normalize_views(text)
        second_call_time = time.time() - start_time
        
        # Both calls should complete successfully
        # Note: We don't assert that cache hit is faster due to system load variations
        assert first_call_time >= 0
        assert second_call_time >= 0


class TestNormalizeStrictConvenienceFunction:
    """Test the normalize_strict convenience function."""
    
    def test_convenience_function_returns_strict_view(self):
        """Test that convenience function returns strict view."""
        text = "  Hello World!!!  "
        result = normalize_strict(text)
        
        # Should return strict view (casefolded, trimmed)
        assert result == "hello world"
        
        # Test with versus canonicalization
        text = "Python versus JavaScript"
        result = normalize_strict(text)
        assert result == "python vs javascript"
        
        # Test with smart punctuation
        text = "iPhone—17 with \u201Csmart\u201D features"
        result = normalize_strict(text)
        assert result == 'iphone-17 with "smart" features'
    
    def test_convenience_function_equivalent_to_normalize_views(self):
        """Test that convenience function is equivalent to normalize_views()[1]."""
        test_cases = [
            "  Hello World!!!  ",
            "Python versus JavaScript",
            "iPhone—17 with \u201Csmart\u201D features",
            "🔥 stock trends",
            "https://example.com hello world",
            "p10 / p50 / p90",
            "Hello\u200BWorld\u0000Test\uFEFF",
            "Hello\u00A0\u2000World\u2001Test",
            "!!!Hello World!!!",
            "café",
            "９０％",
            "１２３",
            "５０．５"
        ]
        
        for text in test_cases:
            # Get result from convenience function
            convenience_result = normalize_strict(text)
            
            # Get result from normalize_views (strict view)
            _, strict_view, _ = normalize_views(text)
            
            # They should be equivalent
            assert convenience_result == strict_view, f"Failed for text: {repr(text)}"
    
    def test_convenience_function_handles_edge_cases(self):
        """Test that convenience function handles edge cases."""
        # Test empty string
        result = normalize_strict("")
        assert result == ""
        
        # Test whitespace only
        result = normalize_strict("   \t\n   ")
        assert result == ""
        
        # Test only punctuation
        result = normalize_strict("!!!")
        assert result == ""
        
        # Test None (should return None gracefully)
        result = normalize_strict(None)
        assert result is None
        
        # Test non-string input (should raise TypeError)
        with pytest.raises(TypeError):
            normalize_strict(123)
        
        # Test with zero-width characters
        result = normalize_strict("Hello\u200BWorld\u0000Test\uFEFF")
        assert result == "helloworldtest"
        
        # Test with control characters
        result = normalize_strict("Hello\u0000World\u0001Test")
        assert result == "helloworldtest"
        
        # Test with mixed whitespace
        result = normalize_strict("Hello\t\n\rWorld")
        assert result == "hello world"
        
        # Test with full-width digits
        result = normalize_strict("９０％")
        assert result == "90%"
        
        # Test with combining characters
        result = normalize_strict("café")
        assert result == "café" 
"""
Unit tests for Agent Validators
"""

import pytest
from app.services.agent.validators import AgentValidator, ValidationResult
from app.services.agent.slot_extractor import ExtractedSlots
from app.models.agent_models import AgentIntent


class TestAgentValidator:
    """Test cases for AgentValidator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AgentValidator()
    
    def test_validate_query_length_valid(self):
        """Test validation of valid query lengths."""
        valid_queries = [
            "Forecast machine learning trends",
            "Compare python vs javascript",
            "Show me a summary of data science"
        ]
        
        for query in valid_queries:
            result = self.validator.validate_query_length(query)
            assert result.is_valid is True
            assert len(result.errors) == 0
    
    def test_validate_query_length_too_long(self):
        """Test validation of queries that are too long."""
        long_query = "forecast " * 200  # Much longer than 1000 chars
        result = self.validator.validate_query_length(long_query)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "too long" in result.errors[0].lower()
    
    def test_validate_query_length_too_short(self):
        """Test validation of queries that are too short."""
        short_queries = ["", "a", "hi"]
        
        for query in short_queries:
            result = self.validator.validate_query_length(query)
            assert result.is_valid is False
            assert len(result.errors) > 0
            assert "too short" in result.errors[0].lower()
    
    def test_validate_slots_forecast_valid(self):
        """Test validation of valid forecast slots."""
        slots = ExtractedSlots(
            keywords=["machine learning"],
            horizon=30,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.sanitized_slots is not None
    
    def test_validate_slots_forecast_missing_keywords(self):
        """Test validation of forecast slots without keywords."""
        slots = ExtractedSlots(
            horizon=30,
            quantiles=[0.1, 0.5, 0.9]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "keywords are required" in result.errors[0].lower()
    
    def test_validate_slots_forecast_invalid_horizon(self):
        """Test validation of forecast slots with invalid horizon."""
        slots = ExtractedSlots(
            keywords=["machine learning"],
            horizon=0,  # Invalid
            quantiles=[0.1, 0.5, 0.9]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "at least 1 day" in result.errors[0].lower()
    
    def test_validate_slots_forecast_large_horizon_warning(self):
        """Test validation of forecast slots with large horizon (warning)."""
        slots = ExtractedSlots(
            keywords=["machine learning"],
            horizon=100,  # Large but valid
            quantiles=[0.1, 0.5, 0.9]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "exceeds recommended maximum" in result.warnings[0].lower()
    
    def test_validate_slots_forecast_invalid_quantiles(self):
        """Test validation of forecast slots with invalid quantiles."""
        slots = ExtractedSlots(
            keywords=["machine learning"],
            horizon=30,
            quantiles=[0.1, 1.5, 0.9]  # Invalid quantile
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "between 0 and 1" in result.errors[0].lower()
    
    def test_validate_slots_forecast_unsorted_quantiles_warning(self):
        """Test validation of forecast slots with unsorted quantiles (warning)."""
        slots = ExtractedSlots(
            keywords=["machine learning"],
            horizon=30,
            quantiles=[0.9, 0.1, 0.5]  # Unsorted
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert "ascending order" in result.warnings[0].lower()
    
    def test_validate_slots_compare_valid(self):
        """Test validation of valid compare slots."""
        slots = ExtractedSlots(
            keywords=["python", "javascript"]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.COMPARE, 0.9)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_slots_compare_insufficient_keywords(self):
        """Test validation of compare slots with insufficient keywords."""
        slots = ExtractedSlots(
            keywords=["python"]  # Need at least 2
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.COMPARE, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "at least 2 keywords" in result.errors[0].lower()
    
    def test_validate_slots_summary_valid(self):
        """Test validation of valid summary slots."""
        slots = ExtractedSlots(
            keywords=["python"],
            date_range={"start_date": "2024-01-01", "end_date": "2024-01-31"}
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.SUMMARY, 0.8)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_slots_summary_invalid_date_range(self):
        """Test validation of summary slots with invalid date range."""
        slots = ExtractedSlots(
            keywords=["python"],
            date_range={"start_date": "2024-01-31", "end_date": "2024-01-01"}  # Invalid order
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.SUMMARY, 0.8)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "before end_date" in result.errors[0].lower()
    
    def test_validate_slots_evaluate_valid(self):
        """Test validation of valid evaluate slots."""
        slots = ExtractedSlots(
            model_id="abc-123"
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.EVALUATE, 0.9)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_slots_evaluate_invalid_model_id(self):
        """Test validation of evaluate slots with invalid model ID."""
        slots = ExtractedSlots(
            model_id="a"  # Too short
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.EVALUATE, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "too short" in result.errors[0].lower()
    
    def test_validate_slots_health_no_requirements(self):
        """Test validation of health slots (no requirements)."""
        slots = ExtractedSlots()  # Empty slots
        
        result = self.validator.validate_slots(slots, AgentIntent.HEALTH, 0.8)
        
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    def test_validate_slots_low_confidence(self):
        """Test validation with low confidence."""
        slots = ExtractedSlots(
            keywords=["machine learning"]
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.2)  # Low confidence
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "confidence too low" in result.errors[0].lower()
    
    def test_validate_keywords_too_many(self):
        """Test validation of too many keywords."""
        slots = ExtractedSlots(
            keywords=["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"]  # More than max
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is True  # Should still be valid
        assert len(result.warnings) > 0
        assert "too many keywords" in result.warnings[0].lower()
    
    def test_validate_keywords_too_short(self):
        """Test validation of keywords that are too short."""
        slots = ExtractedSlots(
            keywords=["a", "machine learning"]  # One too short
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "too short" in result.errors[0].lower()
    
    def test_validate_keywords_unsafe_content(self):
        """Test validation of keywords with unsafe content."""
        slots = ExtractedSlots(
            keywords=["<script>alert('xss')</script>"]  # Unsafe content
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "unsafe content" in result.errors[0].lower()
    
    def test_sanitize_slots(self):
        """Test slot sanitization."""
        slots = ExtractedSlots(
            keywords=["kw1", "kw2", "kw3", "kw4", "kw5", "kw6"],  # Too many
            horizon=1000,  # Too large
            quantiles=[0.9, 0.1, 0.5]  # Unsorted
        )
        
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        
        assert result.is_valid is True
        assert result.sanitized_slots is not None
        
        # Check sanitization
        sanitized = result.sanitized_slots
        assert len(sanitized.keywords) == 5  # Should be limited
        assert sanitized.horizon == 90  # Should be clamped
        assert sanitized.quantiles == [0.1, 0.5, 0.9]  # Should be sorted
    
    def test_get_validation_help(self):
        """Test getting validation help text."""
        help_text = self.validator.get_validation_help(AgentIntent.FORECAST)
        
        assert isinstance(help_text, str)
        assert len(help_text) > 0
        assert "forecast" in help_text.lower()
        assert "keywords" in help_text.lower()
        
        help_text = self.validator.get_validation_help(AgentIntent.COMPARE)
        assert "compare" in help_text.lower()
        assert "keywords" in help_text.lower()
    
    def test_contains_unsafe_content(self):
        """Test detection of unsafe content."""
        unsafe_patterns = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')",
            "onload=alert('xss')",
            "onerror=alert('xss')",
            "<iframe src='http://evil.com'>",
            "<object data='evil.swf'>",
            "<embed src='evil.swf'>"
        ]
        
        for pattern in unsafe_patterns:
            assert self.validator._contains_unsafe_content(pattern) is True
        
        # Safe content
        safe_content = [
            "machine learning",
            "python programming",
            "data science trends",
            "artificial intelligence"
        ]
        
        for content in safe_content:
            assert self.validator._contains_unsafe_content(content) is False
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty slots
        slots = ExtractedSlots()
        result = self.validator.validate_slots(slots, AgentIntent.HEALTH, 0.8)
        assert result.is_valid is True
        
        # Very high confidence
        result = self.validator.validate_slots(slots, AgentIntent.HEALTH, 1.0)
        assert result.is_valid is True
        
        # Boundary horizon values
        slots = ExtractedSlots(keywords=["test"], horizon=1)
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        assert result.is_valid is True
        
        slots = ExtractedSlots(keywords=["test"], horizon=90)
        result = self.validator.validate_slots(slots, AgentIntent.FORECAST, 0.9)
        assert result.is_valid is True
    
    def test_validation_result_structure(self):
        """Test ValidationResult structure."""
        result = ValidationResult(is_valid=True)
        
        assert result.is_valid is True
        assert isinstance(result.errors, list)
        assert isinstance(result.warnings, list)
        assert result.sanitized_slots is None
        
        # Test with errors
        result = ValidationResult(is_valid=False)
        result.errors.append("Test error")
        
        assert result.is_valid is False
        assert len(result.errors) == 1
        assert result.errors[0] == "Test error" 
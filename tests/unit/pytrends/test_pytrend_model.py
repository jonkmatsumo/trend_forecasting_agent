"""
Unit tests for pytrend models.
"""

import pytest
from datetime import datetime, timedelta
from app.models.pytrends.pytrend_model import TrendData, TrendsRequest


class TestTrendData:
    """Test TrendData model."""

    def test_valid_trend_data(self):
        """Test valid trend data creation."""
        data = TrendData(
            keyword="python",
            dates=["2023-01-01", "2023-01-02", "2023-01-03"],
            interest_values=[50, 60, 70]
        )
        assert data.keyword == "python"
        assert len(data.dates) == 3
        assert len(data.interest_values) == 3
        assert data.interest_values == [50, 60, 70]

    def test_invalid_keyword(self):
        """Test invalid keyword validation."""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            TrendData(
                keyword="",
                dates=["2023-01-01"],
                interest_values=[50]
            )

    def test_mismatched_data_lengths(self):
        """Test validation of mismatched data lengths."""
        with pytest.raises(ValueError, match="Dates and interest values must have same length"):
            TrendData(
                keyword="python",
                dates=["2023-01-01", "2023-01-02"],
                interest_values=[50]
            )

    def test_empty_data(self):
        """Test validation of empty data."""
        with pytest.raises(ValueError, match="At least one data point is required"):
            TrendData(
                keyword="python",
                dates=[],
                interest_values=[]
            )

    def test_invalid_interest_values(self):
        """Test validation of interest values."""
        with pytest.raises(ValueError, match="Interest value at index 0 must be between 0 and 100"):
            TrendData(
                keyword="python",
                dates=["2023-01-01"],
                interest_values=[150]
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        data = TrendData(
            keyword="python",
            dates=["2023-01-01", "2023-01-02"],
            interest_values=[50, 60]
        )
        result = data.to_dict()
        assert result["keyword"] == "python"
        assert result["dates"] == ["2023-01-01", "2023-01-02"]
        assert result["interest_values"] == [50, 60]


class TestTrendsRequest:
    """Test TrendsRequest model."""

    def test_valid_trends_request(self):
        """Test valid trends request creation."""
        request = TrendsRequest(
            keywords=["python", "javascript"],
            timeframe="today 12-m"
        )
        assert request.keywords == ["python", "javascript"]
        assert request.timeframe == "today 12-m"

    def test_empty_keywords(self):
        """Test validation of empty keywords."""
        with pytest.raises(ValueError, match="At least one keyword is required"):
            TrendsRequest(
                keywords=[],
                timeframe="today 12-m"
            )

    def test_too_many_keywords(self):
        """Test validation of too many keywords."""
        with pytest.raises(ValueError, match="Maximum 5 keywords allowed"):
            TrendsRequest(
                keywords=["a", "b", "c", "d", "e", "f"],
                timeframe="today 12-m"
            )

    def test_empty_keyword(self):
        """Test validation of empty keyword in list."""
        with pytest.raises(ValueError, match="Keyword at index 1 cannot be empty"):
            TrendsRequest(
                keywords=["python", ""],
                timeframe="today 12-m"
            )

    def test_long_keyword(self):
        """Test validation of keyword length."""
        with pytest.raises(ValueError, match="Keyword at index 0 is too long"):
            TrendsRequest(
                keywords=["a" * 101],
                timeframe="today 12-m"
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        request = TrendsRequest(
            keywords=["python"],
            timeframe="today 12-m"
        )
        result = request.to_dict()
        assert result["keywords"] == ["python"]
        assert result["timeframe"] == "today 12-m" 
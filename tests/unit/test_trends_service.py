"""
Unit tests for trends service
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import time

from app.services.trends_service import TrendsService
from app.models.trend_model import TrendsRequest, TrendData, TrendsResponse
from app.utils.error_handlers import TrendsAPIError, RateLimitError, ValidationError


class TestTrendsService:
    """Test TrendsService class"""
    
    @pytest.fixture
    def trends_service(self):
        """Create a TrendsService instance for testing"""
        return TrendsService()
    
    @pytest.fixture
    def sample_trends_request(self):
        """Create a sample TrendsRequest for testing"""
        return TrendsRequest(
            keywords=["python", "javascript"],
            timeframe="today 12-m",
            geo="US"
        )
    
    @pytest.fixture
    def mock_interest_data(self):
        """Create mock interest over time data"""
        dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
        data = {
            'python': [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
            'javascript': [40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
        }
        return pd.DataFrame(data, index=dates)
    
    def test_trends_service_initialization(self, trends_service):
        """Test TrendsService initialization"""
        assert trends_service.trend is not None
        assert trends_service.validator is not None
        assert trends_service.logger is not None
        assert trends_service.rate_limit_counter == 0
        assert trends_service.max_requests_per_minute == 60
        assert trends_service._cache_ttl == 300
    
    def test_generate_cache_key(self, trends_service, sample_trends_request):
        """Test cache key generation"""
        cache_key = trends_service._generate_cache_key(sample_trends_request)
        expected_key = "python,javascript_today 12-m_US"
        assert cache_key == expected_key
    
    def test_cache_operations(self, trends_service, sample_trends_request):
        """Test cache operations"""
        # Test caching data
        response = TrendsResponse(
            status="success",
            data=[],
            request_info=sample_trends_request.to_dict(),
            timestamp=datetime.utcnow()
        )
        
        cache_key = trends_service._generate_cache_key(sample_trends_request)
        trends_service._cache_data(cache_key, response)
        
        # Test getting cached data
        cached_data = trends_service._get_cached_data(cache_key)
        assert cached_data is not None
        assert cached_data.status == "success"
        
        # Test cache expiration
        trends_service._cache[cache_key]['timestamp'] = time.time() - 400  # Expired
        expired_data = trends_service._get_cached_data(cache_key)
        assert expired_data is None
        assert cache_key not in trends_service._cache
    
    def test_rate_limiting(self, trends_service):
        """Test rate limiting functionality"""
        # First request should pass
        trends_service._check_rate_limit()
        assert trends_service.rate_limit_counter == 1
        
        # Simulate rate limit exceeded
        trends_service.rate_limit_counter = 60
        
        with pytest.raises(RateLimitError) as exc_info:
            trends_service._check_rate_limit()
        
        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.details['retry_after'] == 60
    
    @patch('app.services.trends_service.TrendReq')
    def test_fetch_from_google_trends_success(self, mock_trend_req, trends_service, 
                                            sample_trends_request, mock_interest_data):
        """Test successful Google Trends data fetching"""
        # Mock the TrendReq instance
        mock_trend = Mock()
        mock_trend_req.return_value = mock_trend
        mock_trend.interest_over_time.return_value = mock_interest_data
        trends_service.trend = mock_trend
        
        # Fetch data
        result = trends_service._fetch_from_google_trends(sample_trends_request)
        
        # Verify results
        assert len(result) == 2
        assert result[0].keyword == "python"
        assert result[1].keyword == "javascript"
        assert len(result[0].dates) == 10
        assert len(result[0].interest_values) == 10
        
        # Verify API calls
        mock_trend.build_payload.assert_called_once_with(
            kw_list=["python", "javascript"],
            timeframe="today 12-m",
            geo="US"
        )
        mock_trend.interest_over_time.assert_called_once()
    
    @patch('app.services.trends_service.TrendReq')
    def test_fetch_from_google_trends_empty_data(self, mock_trend_req, trends_service, 
                                                sample_trends_request):
        """Test Google Trends API returning empty data"""
        # Mock empty response
        mock_trend = Mock()
        mock_trend_req.return_value = mock_trend
        mock_trend.interest_over_time.return_value = pd.DataFrame()
        trends_service.trend = mock_trend
        
        with pytest.raises(TrendsAPIError) as exc_info:
            trends_service._fetch_from_google_trends(sample_trends_request)
        
        assert "No data returned from Google Trends API" in str(exc_info.value)
    
    @patch('app.services.trends_service.TrendReq')
    def test_fetch_from_google_trends_api_error(self, mock_trend_req, trends_service, 
                                               sample_trends_request):
        """Test Google Trends API error handling"""
        # Mock API error
        mock_trend = Mock()
        mock_trend_req.return_value = mock_trend
        mock_trend.interest_over_time.side_effect = Exception("API Error")
        trends_service.trend = mock_trend
        
        with pytest.raises(TrendsAPIError) as exc_info:
            trends_service._fetch_from_google_trends(sample_trends_request)
        
        assert "Google Trends API request failed" in str(exc_info.value)
    
    @patch('app.services.trends_service.TrendReq')
    def test_fetch_trends_data_success(self, mock_trend_req, trends_service, 
                                      sample_trends_request, mock_interest_data):
        """Test successful trends data fetching with caching"""
        # Mock the TrendReq instance
        mock_trend = Mock()
        mock_trend_req.return_value = mock_trend
        mock_trend.interest_over_time.return_value = mock_interest_data
        trends_service.trend = mock_trend
        
        # Fetch data
        response = trends_service.fetch_trends_data(sample_trends_request)
        
        # Verify response
        assert response.status == "success"
        assert len(response.data) == 2
        assert response.total_keywords == 2
        
        # Verify cache was populated
        cache_key = trends_service._generate_cache_key(sample_trends_request)
        assert cache_key in trends_service._cache
    
    @patch('app.services.trends_service.TrendReq')
    def test_fetch_trends_data_cached(self, mock_trend_req, trends_service, 
                                     sample_trends_request, mock_interest_data):
        """Test trends data fetching with cache hit"""
        # Mock the TrendReq instance
        mock_trend = Mock()
        mock_trend_req.return_value = mock_trend
        mock_trend.interest_over_time.return_value = mock_interest_data
        trends_service.trend = mock_trend
        
        # Pre-populate cache
        cache_key = trends_service._generate_cache_key(sample_trends_request)
        cached_response = TrendsResponse(
            status="success",
            data=[],
            request_info=sample_trends_request.to_dict(),
            timestamp=datetime.utcnow()
        )
        trends_service._cache_data(cache_key, cached_response)
        
        # Fetch data (should use cache)
        response = trends_service.fetch_trends_data(sample_trends_request)
        
        # Verify cache was used (no API call)
        mock_trend.interest_over_time.assert_not_called()
        assert response.status == "success"
    
    def test_get_trends_for_keyword(self, trends_service):
        """Test convenience method for single keyword"""
        with patch.object(trends_service, 'fetch_trends_data') as mock_fetch:
            # Mock response
            mock_response = Mock()
            mock_response.data = [Mock(keyword="python")]
            mock_fetch.return_value = mock_response
            
            # Call method
            result = trends_service.get_trends_for_keyword("python")
            
            # Verify result
            assert result.keyword == "python"
            mock_fetch.assert_called_once()
    
    def test_get_trends_for_keywords(self, trends_service):
        """Test convenience method for multiple keywords"""
        with patch.object(trends_service, 'fetch_trends_data') as mock_fetch:
            # Mock response
            mock_response = Mock()
            mock_response.data = [Mock(keyword="python"), Mock(keyword="javascript")]
            mock_fetch.return_value = mock_response
            
            # Call method
            result = trends_service.get_trends_for_keywords(["python", "javascript"])
            
            # Verify result
            assert len(result) == 2
            assert result[0].keyword == "python"
            assert result[1].keyword == "javascript"
            mock_fetch.assert_called_once()
    
    def test_clear_cache(self, trends_service, sample_trends_request):
        """Test cache clearing"""
        # Add some data to cache
        cache_key = trends_service._generate_cache_key(sample_trends_request)
        trends_service._cache[cache_key] = {'data': 'test', 'timestamp': time.time()}
        
        # Clear cache
        trends_service.clear_cache()
        
        # Verify cache is empty
        assert len(trends_service._cache) == 0
    
    def test_get_cache_stats(self, trends_service):
        """Test cache statistics"""
        stats = trends_service.get_cache_stats()
        
        assert 'cache_size' in stats
        assert 'cache_ttl' in stats
        assert 'rate_limit_counter' in stats
        assert 'max_requests_per_minute' in stats
        assert stats['cache_size'] == 0
        assert stats['cache_ttl'] == 300
        assert stats['max_requests_per_minute'] == 60
    
    def test_cache_size_management(self, trends_service):
        """Test cache size management"""
        # Add more than 100 cache entries
        for i in range(110):
            trends_service._cache[f"key_{i}"] = {
                'data': f"data_{i}",
                'timestamp': time.time()
            }
        
        # Add one more entry to trigger cleanup
        trends_service._cache_data("new_key", Mock())
        
        # Verify cache size is managed
        assert len(trends_service._cache) <= 100
    
    def test_get_trends_summary(self, trends_service):
        """Test trends summary generation"""
        with patch.object(trends_service, 'get_trends_for_keywords') as mock_get:
            # Mock trend data
            mock_trend1 = Mock()
            mock_trend1.keyword = "python"
            mock_trend1.average_interest = 75.0
            mock_trend1.max_interest = 90.0
            mock_trend1.min_interest = 60.0
            mock_trend1.data_points = 10
            mock_trend1.interest_values = [70, 75, 80, 85, 90]  # Add interest_values
            
            mock_trend2 = Mock()
            mock_trend2.keyword = "javascript"
            mock_trend2.average_interest = 65.0
            mock_trend2.max_interest = 80.0
            mock_trend2.min_interest = 50.0
            mock_trend2.data_points = 10
            mock_trend2.interest_values = [60, 65, 70, 75, 80]  # Add interest_values
            
            mock_get.return_value = [mock_trend1, mock_trend2]
            
            # Call method
            summary = trends_service.get_trends_summary(["python", "javascript"])
            
            # Verify result
            assert summary['keywords'] == ["python", "javascript"]
            assert summary['total_keywords'] == 2
            assert "python" in summary['keyword_stats']
            assert "javascript" in summary['keyword_stats']
            assert summary['keyword_stats']['python']['average_interest'] == 75.0
            assert summary['keyword_stats']['javascript']['average_interest'] == 65.0
    
    def test_calculate_trend_direction(self, trends_service):
        """Test trend direction calculation"""
        # Test increasing trend
        increasing_data = Mock()
        increasing_data.interest_values = [10, 20, 30, 40, 50]
        direction = trends_service._calculate_trend_direction(increasing_data)
        assert direction == "increasing"
        
        # Test decreasing trend
        decreasing_data = Mock()
        decreasing_data.interest_values = [50, 40, 30, 20, 10]
        direction = trends_service._calculate_trend_direction(decreasing_data)
        assert direction == "decreasing"
        
        # Test stable trend
        stable_data = Mock()
        stable_data.interest_values = [30, 31, 29, 30, 31]
        direction = trends_service._calculate_trend_direction(stable_data)
        assert direction == "stable"
    
    def test_compare_keywords(self, trends_service):
        """Test keyword comparison functionality"""
        with patch.object(trends_service, 'get_trends_for_keywords') as mock_get:
            # Mock trend data
            mock_trend1 = Mock()
            mock_trend1.keyword = "python"
            mock_trend1.average_interest = 75.0
            mock_trend1.interest_values = [70, 75, 80]
            
            mock_trend2 = Mock()
            mock_trend2.keyword = "javascript"
            mock_trend2.average_interest = 65.0
            mock_trend2.interest_values = [60, 65, 70]
            
            mock_get.return_value = [mock_trend1, mock_trend2]
            
            # Call method
            comparison = trends_service.compare_keywords(["python", "javascript"])
            
            # Verify result
            assert comparison['keywords'] == ["python", "javascript"]
            assert comparison['highest_average_interest']['keyword'] == "python"
            assert comparison['highest_average_interest']['value'] == 75.0
            assert "python" in comparison['keyword_comparison']
            assert "javascript" in comparison['keyword_comparison']
            assert comparison['keyword_comparison']['python']['rank'] == 1
            assert comparison['keyword_comparison']['javascript']['rank'] == 2
    
    def test_compare_keywords_insufficient_data(self, trends_service):
        """Test keyword comparison with insufficient data"""
        with patch.object(trends_service, 'get_trends_for_keywords') as mock_get:
            mock_get.return_value = [Mock(keyword="python")]
            
            with pytest.raises(ValidationError) as exc_info:
                trends_service.compare_keywords(["python"])
            
            assert "At least 2 keywords required for comparison" in str(exc_info.value)
    
    def test_calculate_volatility(self, trends_service):
        """Test volatility calculation"""
        # Test with varying data
        trend_data = Mock()
        trend_data.interest_values = [10, 20, 30, 40, 50]
        trend_data.average_interest = 30.0
        
        volatility = trends_service._calculate_volatility(trend_data)
        assert volatility > 0  # Should have some volatility
        
        # Test with constant data
        trend_data.interest_values = [30, 30, 30, 30, 30]
        trend_data.average_interest = 30.0
        
        volatility = trends_service._calculate_volatility(trend_data)
        assert volatility == 0.0  # Should have no volatility
    
    def test_calculate_rank(self, trends_service):
        """Test rank calculation"""
        # Create mock trend data
        trend1 = Mock()
        trend1.keyword = "python"
        trend1.average_interest = 75.0
        
        trend2 = Mock()
        trend2.keyword = "javascript"
        trend2.average_interest = 65.0
        
        trend3 = Mock()
        trend3.keyword = "java"
        trend3.average_interest = 85.0
        
        all_trends = [trend1, trend2, trend3]
        
        # Test ranking
        rank1 = trends_service._calculate_rank(trend1, all_trends)
        rank2 = trends_service._calculate_rank(trend2, all_trends)
        rank3 = trends_service._calculate_rank(trend3, all_trends)
        
        assert rank1 == 2  # python is second
        assert rank2 == 3  # javascript is third
        assert rank3 == 1  # java is first


class TestTrendsServiceIntegration:
    """Integration tests for TrendsService"""
    
    @pytest.fixture
    def trends_service(self):
        """Create a TrendsService instance for integration testing"""
        return TrendsService()
    
    def test_full_trends_request_flow(self, trends_service):
        """Test complete trends request flow with validation"""
        # Create request
        request = TrendsRequest(
            keywords=["python"],
            timeframe="today 12-m",
            geo=""
        )
        
        # This would normally make a real API call, but we'll test the structure
        assert request.keywords == ["python"]
        assert request.timeframe == "today 12-m"
        assert request.geo == ""
    
    def test_error_handling_integration(self, trends_service):
        """Test error handling integration"""
        # Test with invalid request (empty keywords)
        with pytest.raises(ValueError) as exc_info:
            TrendsRequest(keywords=[], timeframe="today 12-m")
        
        assert "At least one keyword is required" in str(exc_info.value)
    
    def test_rate_limiting_integration(self, trends_service):
        """Test rate limiting integration"""
        # Reset rate limit counter
        trends_service.rate_limit_counter = 0
        
        # Make multiple requests
        for i in range(5):
            trends_service._check_rate_limit()
        
        assert trends_service.rate_limit_counter == 5
        
        # Test rate limit exceeded
        trends_service.rate_limit_counter = 60
        
        with pytest.raises(RateLimitError):
            trends_service._check_rate_limit() 
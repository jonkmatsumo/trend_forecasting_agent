"""
Integration tests for trends API endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
from datetime import datetime

from app import create_app
from app.models.pytrends.pytrend_model import TrendData, TrendsResponse


class TestTrendsAPI:
    """Integration tests for trends API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create Flask app for testing"""
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def mock_trends_service(self):
        """Mock trends service responses"""
        with patch('app.services.pytrends.trends_service.TrendsService') as mock_service:
            # Mock successful trends response
            mock_trend_data = TrendData(
                keyword="python",
                dates=["2023-01-01", "2023-01-02", "2023-01-03"],
                interest_values=[50, 60, 70],
                timeframe="today 12-m",
                geo="US",
                last_updated=datetime.utcnow()
            )
            
            mock_response = TrendsResponse(
                status="success",
                data=[mock_trend_data],
                request_info={"keywords": ["python"], "timeframe": "today 12-m", "geo": "US"},
                timestamp=datetime.utcnow()
            )
            
            mock_service_instance = Mock()
            mock_service_instance.fetch_trends_data.return_value = mock_response
            mock_service_instance.get_trends_summary.return_value = {
                'keywords': ['python', 'javascript'],
                'total_keywords': 2,
                'keyword_stats': {
                    'python': {'average_interest': 75.0, 'max_interest': 90.0, 'min_interest': 60.0},
                    'javascript': {'average_interest': 65.0, 'max_interest': 80.0, 'min_interest': 50.0}
                }
            }
            mock_service_instance.compare_keywords.return_value = {
                'keywords': ['python', 'javascript'],
                'highest_average_interest': {'keyword': 'python', 'value': 75.0},
                'most_volatile': {'keyword': 'javascript', 'volatility': 15.0},
                'keyword_comparison': {
                    'python': {'average_interest': 75.0, 'rank': 1},
                    'javascript': {'average_interest': 65.0, 'rank': 2}
                }
            }
            
            mock_service.return_value = mock_service_instance
            yield mock_service_instance
    
    def test_get_trends_endpoint_success(self, client, mock_trends_service):
        """Test successful trends data retrieval"""
        # Prepare request data
        request_data = {
            "keywords": ["python"],
            "timeframe": "today 12-m",
            "geo": "US"
        }
        
        # Make request
        response = client.post('/api/trends', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert len(data['data']) == 1
        assert data['data'][0]['keyword'] == 'python'
        
        # Verify service was called
        mock_trends_service.fetch_trends_data.assert_called_once()
    
    def test_get_trends_endpoint_validation_error(self, client):
        """Test trends endpoint with validation error"""
        # Prepare invalid request data
        request_data = {
            "keywords": [],  # Empty keywords should fail validation
            "timeframe": "today 12-m"
        }
        
        # Make request
        response = client.post('/api/trends', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
    
    def test_get_trends_summary_endpoint_success(self, client, mock_trends_service):
        """Test successful trends summary retrieval"""
        # Prepare request data
        request_data = {
            "keywords": ["python", "javascript"],
            "timeframe": "today 12-m",
            "geo": "US"
        }
        
        # Make request
        response = client.post('/api/trends/summary', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'summary' in data
        # The summary contains keyword_stats and other fields, not a simple list
        assert 'keyword_stats' in data['summary']
        assert 'python' in data['summary']['keyword_stats']
        
        # Verify service was called
        mock_trends_service.get_trends_summary.assert_called_once()
    
    def test_get_trends_summary_endpoint_validation_error(self, client):
        """Test trends summary endpoint with validation error"""
        # Prepare invalid request data
        request_data = {
            "keywords": [],  # Empty keywords should fail validation
            "timeframe": "today 12-m"
        }
        
        # Make request
        response = client.post('/api/trends/summary', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
    
    def test_compare_trends_endpoint_success(self, client, mock_trends_service):
        """Test successful trends comparison"""
        # Prepare request data
        request_data = {
            "keywords": ["python", "javascript"],
            "timeframe": "today 12-m",
            "geo": "US"
        }
        
        # Make request
        response = client.post('/api/trends/compare', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'comparison' in data
        assert data['comparison']['highest_average_interest']['keyword'] == 'python'
        assert len(data['comparison']['keyword_comparison']) == 2
        
        # Verify service was called
        mock_trends_service.compare_keywords.assert_called_once()
    
    def test_compare_trends_endpoint_insufficient_keywords(self, client):
        """Test trends compare endpoint with insufficient keywords"""
        # Prepare invalid request data
        request_data = {
            "keywords": ["python"],  # Only one keyword should fail validation
            "timeframe": "today 12-m"
        }
        
        # Make request
        response = client.post('/api/trends/compare', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
    
    def test_clear_trends_cache_endpoint(self, client, mock_trends_service):
        """Test trends cache clearing endpoint"""
        # Make request
        response = client.post('/api/trends/cache/clear')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'cache cleared successfully' in data['message']
        
        # Verify service was called
        mock_trends_service.clear_cache.assert_called_once()
    
    def test_health_endpoint(self, client):
        """Test API health endpoint"""
        # Make request
        response = client.get('/api/health')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'Google Trends Quantile Forecaster API'
        assert 'version' in data
        assert 'timestamp' in data
    
    def test_trends_endpoint_missing_body(self, client):
        """Test trends endpoint with missing request body"""
        # Make request without body
        response = client.post('/api/trends')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'Content-Type must be application/json' in data['message']
    
    def test_trends_endpoint_invalid_json(self, client):
        """Test trends endpoint with invalid JSON"""
        # Make request with invalid JSON
        response = client.post('/api/trends', 
                             data='invalid json',
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR' 
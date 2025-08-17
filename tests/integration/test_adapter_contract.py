"""
Contract Tests for Adapter Consistency
Tests that both adapters produce identical outputs and handle errors consistently.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from app.services.adapters.in_process_adapter import InProcessAdapter
from app.services.adapters.http_adapter import HTTPAdapter
from app.config.adapter_config import AdapterConfig, AdapterType


class TestAdapterContract:
    """Test that both adapters produce identical outputs and handle errors consistently."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock all services for in-process adapter
        with patch('app.services.adapters.in_process_adapter.TrendsService') as mock_trends, \
             patch('app.services.adapters.in_process_adapter.TrainingService') as mock_training, \
             patch('app.services.adapters.in_process_adapter.PredictionService') as mock_prediction, \
             patch('app.services.adapters.in_process_adapter.EvaluationService') as mock_evaluation:
            
            # Create mock service instances
            self.mock_trends_service = Mock()
            self.mock_training_service = Mock()
            self.mock_prediction_service = Mock()
            self.mock_evaluation_service = Mock()
            
            # Configure mocks to return our instances
            mock_trends.return_value = self.mock_trends_service
            mock_training.return_value = self.mock_training_service
            mock_prediction.return_value = self.mock_prediction_service
            mock_evaluation.return_value = self.mock_evaluation_service
            
            # Create adapter instances
            self.in_process_adapter = InProcessAdapter()
            self.http_adapter = HTTPAdapter("http://localhost:5000")
    
    def test_health_endpoint_consistency(self):
        """Test that health endpoint returns consistent results across adapters."""
        # Mock the underlying service
        mock_service = Mock()
        mock_service.health.return_value = {
            "status": "healthy",
            "service": "test_service",
            "version": "1.0.0"
        }
        
        # Mock the HTTP adapter to return the same response
        with patch.object(self.http_adapter, '_make_request') as mock_http_request:
            mock_http_request.return_value = {
                "status": "healthy",
                "service": "Google Trends Quantile Forecaster API",
                "version": "v1"
            }
            
            # Test both adapters
            in_process_result = self.in_process_adapter.health()
            http_result = self.http_adapter.health()
            
            # Both should return the same structure
            assert in_process_result["status"] == http_result["status"]
            assert in_process_result["service"] == http_result["service"]
            assert in_process_result["version"] == http_result["version"]
    
    def test_trends_summary_endpoint_consistency(self):
        """Test that both adapters return consistent trends summary responses."""
        # Mock trends service response for in-process adapter
        mock_trends_response = Mock()
        mock_trends_response.data = [{"keyword": "ai", "values": [1, 2, 3]}]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        # Mock HTTP response for trends summary
        mock_summary_response = {
            "status": "success",
            "summary": {
                "keywords": ["artificial intelligence"],
                "timeframe": "today 12-m",
                "geo": "",
                "statistics": {
                    "total_keywords": 1,
                    "data_points": 1,
                    "timeframe": "today 12-m",
                    "geo": ""
                }
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        with patch.object(self.http_adapter, '_make_request') as mock_http_request:
            mock_http_request.return_value = mock_summary_response
            
            # Get responses from both adapters
            in_process_result = self.in_process_adapter.trends_summary(["artificial intelligence"])
            http_result = self.http_adapter.trends_summary(["artificial intelligence"])
            
            # Verify both return success with same structure
            assert in_process_result["status"] == "success"
            assert http_result["status"] == "success"
            assert in_process_result["summary"]["keywords"] == http_result["summary"]["keywords"]
            assert in_process_result["summary"]["timeframe"] == http_result["summary"]["timeframe"]
            assert in_process_result["summary"]["geo"] == http_result["summary"]["geo"]
    
    def test_compare_endpoint_consistency(self):
        """Test that both adapters return consistent compare responses."""
        # Mock trends service response for in-process adapter
        mock_trends_response = Mock()
        mock_trends_response.data = [
            {"keyword": "chatgpt", "values": [1, 2, 3]},
            {"keyword": "claude", "values": [4, 5, 6]}
        ]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        # Mock HTTP response for compare
        mock_compare_response = {
            "status": "success",
            "comparison": {
                "keywords": ["chatgpt", "claude"],
                "timeframe": "today 12-m",
                "geo": "",
                "comparison_data": {
                    "total_keywords": 2,
                    "data_available": 2,
                    "timeframe": "today 12-m",
                    "geo": ""
                }
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        with patch.object(self.http_adapter, '_make_request') as mock_http_request:
            mock_http_request.return_value = mock_compare_response
            
            # Get responses from both adapters
            in_process_result = self.in_process_adapter.compare(["chatgpt", "claude"])
            http_result = self.http_adapter.compare(["chatgpt", "claude"])
            
            # Verify both return success with same structure
            assert in_process_result["status"] == "success"
            assert http_result["status"] == "success"
            assert in_process_result["comparison"]["keywords"] == http_result["comparison"]["keywords"]
            assert in_process_result["comparison"]["timeframe"] == http_result["comparison"]["timeframe"]
            assert in_process_result["comparison"]["geo"] == http_result["comparison"]["geo"]
    
    def test_list_models_endpoint_consistency(self):
        """Test that both adapters return consistent list models responses."""
        # Mock training service responses for in-process adapter
        self.mock_training_service.list_models.return_value = ["model_1", "model_2"]
        
        mock_metadata = {
            "keyword": "artificial intelligence",
            "model_type": "n_beats",
            "training_date": "2024-01-15T10:30:00Z"
        }
        mock_evaluation_metrics = Mock()
        mock_evaluation_metrics.test_mae = 2.5
        mock_evaluation_metrics.test_rmse = 3.1
        mock_evaluation_metrics.directional_accuracy = 0.85
        
        self.mock_training_service.get_model_metadata.return_value = mock_metadata
        self.mock_training_service.get_evaluation_metrics.return_value = mock_evaluation_metrics
        
        # Mock HTTP response for list models
        mock_models_response = {
            "status": "success",
            "models": [
                {
                    "model_id": "model_1",
                    "keyword": "artificial intelligence",
                    "model_type": "n_beats",
                    "created_at": "2024-01-15T10:30:00Z",
                    "test_mae": 2.5,
                    "test_rmse": 3.1,
                    "directional_accuracy": 0.85
                },
                {
                    "model_id": "model_2",
                    "keyword": "artificial intelligence",
                    "model_type": "n_beats",
                    "created_at": "2024-01-15T10:30:00Z",
                    "test_mae": 2.5,
                    "test_rmse": 3.1,
                    "directional_accuracy": 0.85
                }
            ],
            "total_count": 2,
            "timestamp": "2024-01-15T10:30:00Z"
        }
        
        with patch.object(self.http_adapter, '_make_request') as mock_http_request:
            mock_http_request.return_value = mock_models_response
            
            # Get responses from both adapters
            in_process_result = self.in_process_adapter.list_models(limit=10, offset=0)
            http_result = self.http_adapter.list_models(limit=10, offset=0)
            
            # Verify both return success with same structure
            assert in_process_result["status"] == "success"
            assert http_result["status"] == "success"
            assert in_process_result["total_count"] == http_result["total_count"]
            assert len(in_process_result["models"]) == len(http_result["models"])
    
    def test_error_handling_consistency(self):
        """Test that both adapters handle errors consistently."""
        # Test validation error consistency
        in_process_result = self.in_process_adapter.trends_summary([])
        http_result = self.http_adapter.trends_summary([])
        
        # Both should return validation errors
        assert in_process_result["status"] == "error"
        assert http_result["status"] == "error"
        assert in_process_result["error_code"] in ["VALIDATION_ERROR", "INTERNAL_ERROR"]
        assert http_result["error_code"] in ["VALIDATION_ERROR", "INTERNAL_ERROR"]
    
    def test_timeout_scenarios(self):
        """Test HTTP adapter timeout handling."""
        # Mock timeout exception
        with patch.object(self.http_adapter, '_make_request') as mock_http_request:
            mock_http_request.side_effect = Exception("Connection timeout")
            
            result = self.http_adapter.health()
            
            # Should return error response
            assert result["status"] == "error"
            assert result["error_code"] == "INTERNAL_ERROR"
    
    def test_retry_scenarios(self):
        """Test HTTP adapter retry logic."""
        # Test that the adapter properly handles retries by mocking the session request
        with patch.object(self.http_adapter.session, 'request') as mock_session_request:
            # First call returns 500 error, second call succeeds
            mock_response_500 = Mock()
            mock_response_500.status_code = 500
            mock_response_500.text = "Internal Server Error"
            
            mock_response_200 = Mock()
            mock_response_200.status_code = 200
            mock_response_200.json.return_value = {"status": "healthy", "service": "test"}
            
            mock_session_request.side_effect = [mock_response_500, mock_response_200]
            
            result = self.http_adapter.health()
            
            # Should eventually succeed after retry
            assert result["status"] == "healthy"
            # The session request should be called twice due to retry
            assert mock_session_request.call_count == 2
    
    def test_adapter_configuration(self):
        """Test adapter configuration management."""
        # Test in-process adapter configuration
        config = AdapterConfig()
        
        # Default should be in-process
        assert config.is_in_process_adapter()
        assert not config.is_http_adapter()
        
        # Test HTTP adapter configuration
        with patch.dict(os.environ, {
            "FORECASTER_ADAPTER_TYPE": "http",
            "FORECASTER_HTTP_URL": "http://localhost:5000",
            "FORECASTER_TIMEOUT": "30",
            "FORECASTER_RETRY_ATTEMPTS": "3"
        }):
            config = AdapterConfig()
            assert config.is_http_adapter()
            assert not config.is_in_process_adapter()
            assert config.http_url == "http://localhost:5000"
            assert config.timeout == 30
            assert config.max_retries == 3
    
    def test_adapter_factory(self):
        """Test adapter factory function."""
        from app.config.adapter_config import create_adapter, adapter_config
        
        # Test in-process adapter creation
        with patch.dict(os.environ, {"FORECASTER_ADAPTER_TYPE": "in_process"}):
            # Force re-initialization of config
            adapter_config.__init__()
            adapter = create_adapter()
            assert isinstance(adapter, InProcessAdapter)
        
        # Test HTTP adapter creation
        with patch.dict(os.environ, {
            "FORECASTER_ADAPTER_TYPE": "http",
            "FORECASTER_HTTP_URL": "http://localhost:5000"
        }):
            # Force re-initialization of config
            adapter_config.__init__()
            adapter = create_adapter()
            assert isinstance(adapter, HTTPAdapter)
            assert adapter.base_url == "http://localhost:5000"
    
    def test_invalid_configuration(self):
        """Test handling of invalid configuration."""
        # Test HTTP adapter without URL
        with patch.dict(os.environ, {"FORECASTER_ADAPTER_TYPE": "http"}):
            with pytest.raises(ValueError, match="FORECASTER_HTTP_URL"):
                AdapterConfig()
        
        # Test invalid timeout
        with patch.dict(os.environ, {
            "FORECASTER_ADAPTER_TYPE": "http",
            "FORECASTER_HTTP_URL": "http://localhost:5000",
            "FORECASTER_TIMEOUT": "invalid"
        }):
            config = AdapterConfig()
            assert config.timeout == 30  # Should default to 30
        
        # Test invalid retry attempts
        with patch.dict(os.environ, {
            "FORECASTER_ADAPTER_TYPE": "http",
            "FORECASTER_HTTP_URL": "http://localhost:5000",
            "FORECASTER_RETRY_ATTEMPTS": "invalid"
        }):
            config = AdapterConfig()
            assert config.max_retries == 3  # Should default to 3 
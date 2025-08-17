"""
Integration tests for In-Process Adapter
Tests the adapter with real service integrations.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from app.services.adapters.in_process_adapter import InProcessAdapter
from app.services.pytrends.trends_service import TrendsService
from app.services.darts.prediction_service import PredictionService
from app.services.darts.training_service import TrainingService
from app.services.darts.evaluation_service import EvaluationService
from app.utils.structured_logger import create_structured_logger
from app.models.forecaster_models import (
    create_prediction_response, create_trends_summary_response, create_compare_response,
    create_training_response, create_evaluation_response, create_list_models_response,
    create_health_response, create_cache_stats_response, create_cache_clear_response
)


class TestInProcessAdapter:
    """Test the In-Process Adapter with mocked services."""
    
    def setup_method(self):
        """Set up test fixtures with mocked services."""
        # Mock all services to avoid actual API calls and model training
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
            
            # Create adapter instance
            self.adapter = InProcessAdapter()
    
    def test_health_endpoint(self):
        """Test health endpoint returns correct response."""
        result = self.adapter.health()
        
        assert result["status"] == "healthy"
        assert result["service"] == "Google Trends Quantile Forecaster API"
        assert result["version"] == "v1"
        assert "timestamp" in result
    
    def test_trends_summary_endpoint(self):
        """Test trends summary endpoint with mocked data."""
        # Mock trends service response
        mock_trends_response = Mock()
        mock_trends_response.data = [{"keyword": "ai", "values": [1, 2, 3]}]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        result = self.adapter.trends_summary(["artificial intelligence"])
        
        assert result["status"] == "success"
        assert result["summary"]["keywords"] == ["artificial intelligence"]
        assert result["summary"]["timeframe"] == "today 12-m"
        assert result["summary"]["geo"] == ""
        assert result["summary"]["statistics"]["total_keywords"] == 1
        assert result["summary"]["statistics"]["data_points"] == 1
        assert "timestamp" in result
    
    def test_compare_endpoint(self):
        """Test compare endpoint with mocked data."""
        # Mock trends service response
        mock_trends_response = Mock()
        mock_trends_response.data = [
            {"keyword": "chatgpt", "values": [1, 2, 3]},
            {"keyword": "claude", "values": [4, 5, 6]}
        ]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        result = self.adapter.compare(["chatgpt", "claude"])
        
        assert result["status"] == "success"
        assert result["comparison"]["keywords"] == ["chatgpt", "claude"]
        assert result["comparison"]["timeframe"] == "today 12-m"
        assert result["comparison"]["geo"] == ""
        assert result["comparison"]["comparison_data"]["total_keywords"] == 2
        assert result["comparison"]["comparison_data"]["data_available"] == 2
        assert "timestamp" in result
    
    def test_compare_endpoint_insufficient_keywords(self):
        """Test compare endpoint with insufficient keywords."""
        result = self.adapter.compare(["single_keyword"])
        
        assert result["status"] == "error"
        assert result["error_code"] == "VALIDATION_ERROR"
        assert "At least 2 keywords are required" in result["message"]
        assert "timestamp" in result
    
    def test_list_models_endpoint(self):
        """Test list models endpoint with mocked data."""
        # Mock training service responses
        self.mock_training_service.list_models.return_value = ["model_1", "model_2"]
        
        # Mock metadata and evaluation metrics
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
        
        result = self.adapter.list_models(limit=10, offset=0)
        
        assert result["status"] == "success"
        assert result["total_count"] == 2
        assert len(result["models"]) == 2
        
        # Check first model structure
        model = result["models"][0]
        assert "model_id" in model
        assert "keyword" in model
        assert "model_type" in model
        assert "created_at" in model
        assert "test_mae" in model
        assert "test_rmse" in model
        assert "directional_accuracy" in model
        assert "timestamp" in result
    
    def test_list_models_with_filters(self):
        """Test list models endpoint with keyword and model type filters."""
        # Mock training service responses
        self.mock_training_service.list_models.return_value = ["model_1", "model_2"]
        
        # Mock metadata for filtering
        mock_metadata_1 = {
            "keyword": "artificial intelligence",
            "model_type": "n_beats",
            "training_date": "2024-01-15T10:30:00Z"
        }
        mock_metadata_2 = {
            "keyword": "machine learning",
            "model_type": "tft",
            "training_date": "2024-01-15T11:30:00Z"
        }
        
        mock_evaluation_metrics = Mock()
        mock_evaluation_metrics.test_mae = 2.5
        mock_evaluation_metrics.test_rmse = 3.1
        mock_evaluation_metrics.directional_accuracy = 0.85
        
        # Configure mock to return different metadata for different calls
        self.mock_training_service.get_model_metadata.side_effect = [mock_metadata_1, mock_metadata_2]
        self.mock_training_service.get_evaluation_metrics.return_value = mock_evaluation_metrics
        
        # Test with keyword filter
        result = self.adapter.list_models(keyword="artificial intelligence")
        
        assert result["status"] == "success"
        assert result["total_count"] == 1  # Only one model should match
        assert result["models"][0]["keyword"] == "artificial intelligence"
    
    def test_evaluate_endpoint(self):
        """Test evaluate endpoint with mocked data."""
        # Mock evaluation metrics
        mock_evaluation_metrics = Mock()
        mock_evaluation_metrics.to_dict.return_value = {
            "test_mae": 2.5,
            "test_rmse": 3.1,
            "test_mape": 3.2,
            "directional_accuracy": 0.85
        }
        
        self.mock_training_service.get_evaluation_metrics.return_value = mock_evaluation_metrics
        
        result = self.adapter.evaluate("model_123")
        
        assert result["status"] == "success"
        assert result["model_id"] == "model_123"
        assert "evaluation_metrics" in result
        assert result["evaluation_metrics"]["test_mae"] == 2.5
        assert result["evaluation_metrics"]["test_rmse"] == 3.1
        assert "timestamp" in result
    
    def test_predict_endpoint(self):
        """Test predict endpoint with mocked data."""
        from datetime import datetime
        
        # Mock forecast result
        mock_forecast_result = Mock()
        mock_forecast_result.forecast_values = [85, 87, 89, 91, 93]
        mock_forecast_result.forecast_dates = [
            datetime(2024, 1, 15), datetime(2024, 1, 22), datetime(2024, 1, 29),
            datetime(2024, 2, 5), datetime(2024, 2, 12)
        ]
        mock_forecast_result.confidence_intervals = {
            "95%": {
                "lower": [80, 82, 84, 86, 88],
                "upper": [90, 92, 94, 96, 98]
            }
        }
        
        self.mock_prediction_service.generate_forecast.return_value = mock_forecast_result
        
        # Mock metadata and evaluation metrics
        mock_metadata = {
            "keyword": "artificial intelligence",
            "model_type": "tft",
            "train_samples": 156,
            "test_samples": 39,
            "total_samples": 195
        }
        mock_evaluation_metrics = Mock()
        mock_evaluation_metrics.to_dict.return_value = {
            "test_mae": 2.5,
            "test_rmse": 3.1,
            "test_mape": 3.2,
            "directional_accuracy": 0.85
        }
        
        self.mock_training_service.get_model_metadata.return_value = mock_metadata
        self.mock_training_service.get_evaluation_metrics.return_value = mock_evaluation_metrics
        
        result = self.adapter.predict("model_123", forecast_horizon=25)
        
        # Check if we got an error response and print it for debugging
        if result["status"] == "error":
            print(f"Error response: {result}")
        
        assert result["status"] == "success"
        assert "forecast" in result
        assert "model_performance" in result
        assert "model_info" in result
        
        # Check forecast data
        forecast = result["forecast"]
        assert forecast["values"] == [85, 87, 89, 91, 93]
        assert len(forecast["dates"]) == 5
        assert "confidence_intervals" in forecast
        
        # Check model info
        model_info = result["model_info"]
        assert model_info["model_id"] == "model_123"
        assert model_info["keyword"] == "artificial intelligence"
        assert model_info["model_type"] == "tft"
        assert model_info["forecast_horizon"] == 25
        assert "timestamp" in result
    
    def test_train_endpoint(self):
        """Test train endpoint with mocked data."""
        # Mock training service response
        mock_evaluation_metrics = Mock()
        mock_evaluation_metrics.to_dict.return_value = {
            "test_mae": 2.5,
            "test_rmse": 3.1,
            "test_mape": 3.2,
            "directional_accuracy": 0.85
        }
        
        self.mock_training_service.train_model.return_value = ("model_abc123", mock_evaluation_metrics)
        
        # Test data
        time_series_data = [70.0] * 52  # Minimum required data points
        dates = [f"2023-{i:02d}-01" for i in range(1, 53)]
        
        result = self.adapter.train(
            keyword="artificial intelligence",
            time_series_data=time_series_data,
            dates=dates,
            model_type="n_beats"
        )
        
        assert result["status"] == "success"
        assert result["model_id"] == "model_abc123"
        assert result["keyword"] == "artificial intelligence"
        assert result["model_type"] == "n_beats"
        assert "evaluation_metrics" in result
        assert result["evaluation_metrics"]["test_mae"] == 2.5
        assert "timestamp" in result
    
    def test_validation_errors(self):
        """Test that validation errors are properly handled."""
        # Test invalid model ID
        result = self.adapter.predict("", forecast_horizon=25)
        assert result["status"] == "error"
        # The validation error is caught by the base class validation, so it should be VALIDATION_ERROR
        assert result["error_code"] in ["VALIDATION_ERROR", "INTERNAL_ERROR"]
        
        # Test invalid keywords
        result = self.adapter.trends_summary([])
        assert result["status"] == "error"
        assert result["error_code"] in ["VALIDATION_ERROR", "INTERNAL_ERROR"]
        
        # Test insufficient time series data
        result = self.adapter.train(
            keyword="test",
            time_series_data=[70.0] * 51,  # Less than minimum required
            dates=["2023-01-01"] * 51,
            model_type="n_beats"
        )
        assert result["status"] == "error"
        assert result["error_code"] in ["VALIDATION_ERROR", "INTERNAL_ERROR"]
    
    def test_request_tracking(self):
        """Test that request tracking works correctly."""
        # Enable debug logging to capture request tracking
        logging.getLogger('app.services.adapters.in_process_adapter').setLevel(logging.DEBUG)
        
        # Mock trends service response
        mock_trends_response = Mock()
        mock_trends_response.data = [{"keyword": "ai", "values": [1, 2, 3]}]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        # Call a method that should generate request tracking
        result = self.adapter.trends_summary(["artificial intelligence"])
        
        # Verify request tracking was generated
        assert hasattr(self.adapter, 'request_id')
        assert self.adapter.request_id is not None
        assert result["status"] == "success"
    
    def test_error_mapping(self):
        """Test that different exception types are mapped to correct error responses."""
        # Test that the adapter properly handles exceptions from the underlying services
        # We'll test this by making the training service raise an exception during list_models
        self.mock_training_service.list_models.side_effect = Exception("Service unavailable")
        
        result = self.adapter.list_models()
        # The error should be caught and mapped to an error response
        assert result["status"] == "error"
        assert result["error_code"] == "INTERNAL_ERROR"
        assert "Service unavailable" in result["message"]
    
    def test_default_parameters(self):
        """Test that default parameters are applied correctly."""
        # Mock trends service response
        mock_trends_response = Mock()
        mock_trends_response.data = [{"keyword": "ai", "values": [1, 2, 3]}]
        self.mock_trends_service.fetch_trends_data.return_value = mock_trends_response
        
        # Test with default parameters
        result = self.adapter.trends_summary(["artificial intelligence"])
        
        # Verify default parameters are used
        assert result["summary"]["timeframe"] == "today 12-m"
        assert result["summary"]["geo"] == ""
        
        # Test with custom parameters
        result = self.adapter.trends_summary(
            ["artificial intelligence"], 
            timeframe="today 3-m", 
            geo="US"
        )
        
        assert result["summary"]["timeframe"] == "today 3-m"
        assert result["summary"]["geo"] == "US" 
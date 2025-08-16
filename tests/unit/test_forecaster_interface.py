"""
Unit tests for Forecaster Service Interface
Tests interface contract compliance and response shapes.
"""

import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch

from app.services.forecaster_interface import ForecasterServiceInterface, AbstractForecasterService
from app.models.forecaster_models import (
    PredictionRequest, PredictionResponse, TrendsSummaryRequest, TrendsSummaryResponse,
    CompareRequest, CompareResponse, TrainingRequest, TrainingResponse,
    EvaluationRequest, EvaluationResponse, ListModelsRequest, ListModelsResponse,
    HealthResponse, CacheStatsResponse, CacheClearResponse, ErrorResponse,
    create_prediction_response, create_trends_summary_response, create_compare_response,
    create_training_response, create_evaluation_response, create_list_models_response,
    create_health_response, create_cache_stats_response, create_cache_clear_response,
    create_error_response
)


class MockForecasterService(AbstractForecasterService):
    """Mock implementation of the Forecaster Service Interface for testing."""
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Mock prediction implementation."""
        self._validate_model_id(model_id)
        return {
            "status": "success",
            "forecast": {
                "values": [85, 87, 89, 91, 93],
                "dates": ["2024-01-15", "2024-01-22", "2024-01-29", "2024-02-05", "2024-02-12"],
                "confidence_intervals": {
                    "95%": {
                        "lower": [80, 82, 84, 86, 88],
                        "upper": [90, 92, 94, 96, 98]
                    }
                }
            },
            "model_performance": {
                "test_metrics": {"mae": 2.5, "rmse": 3.1},
                "train_metrics": {"mae": 2.1, "rmse": 2.8},
                "data_info": {"train_samples": 156, "test_samples": 39}
            },
            "model_info": {
                "model_id": model_id,
                "keyword": "artificial intelligence",
                "model_type": "tft",
                "forecast_horizon": forecast_horizon or 25,
                "generated_at": "2024-01-15T10:40:00Z"
            },
            "timestamp": "2024-01-15T10:40:00Z"
        }
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Mock trends summary implementation."""
        self._validate_keywords(keywords)
        return {
            "status": "success",
            "summary": {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo,
                "statistics": {"mean": 75.5, "std": 12.3}
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Mock compare implementation."""
        self._validate_keywords(keywords)
        if len(keywords) < 2:
            raise ValueError("At least 2 keywords are required for comparison")
        return {
            "status": "success",
            "comparison": {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo,
                "comparison_data": {"correlation": 0.85}
            },
            "timestamp": "2024-01-15T10:35:00Z"
        }
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Mock training implementation."""
        self._validate_time_series_data(time_series_data, dates)
        return {
            "status": "success",
            "model_id": "model_abc123def456",
            "keyword": keyword,
            "model_type": model_type,
            "evaluation_metrics": {
                "test_metrics": {
                    "mae": 2.5,
                    "rmse": 3.1,
                    "mape": 3.2,
                    "directional_accuracy": 0.85
                },
                "train_metrics": {
                    "mae": 2.1,
                    "rmse": 2.8,
                    "mape": 2.5
                },
                "data_info": {
                    "train_samples": 156,
                    "test_samples": 39,
                    "total_samples": 195
                },
                "training_info": {
                    "training_time_seconds": 45.2,
                    "mlflow_run_id": "run_xyz789"
                }
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Mock evaluation implementation."""
        self._validate_model_id(model_id)
        return {
            "status": "success",
            "model_id": model_id,
            "evaluation_metrics": {
                "test_metrics": {
                    "mae": 2.5,
                    "rmse": 3.1,
                    "mape": 3.2,
                    "directional_accuracy": 0.85,
                    "coverage_95": 0.92
                },
                "train_metrics": {
                    "mae": 2.1,
                    "rmse": 2.8,
                    "mape": 2.5
                },
                "data_info": {
                    "train_samples": 156,
                    "test_samples": 39,
                    "total_samples": 195
                },
                "training_info": {
                    "training_time_seconds": 45.2,
                    "created_at": "2024-01-15T10:30:00Z"
                }
            },
            "timestamp": "2024-01-15T10:35:00Z"
        }
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Mock list models implementation."""
        return {
            "status": "success",
            "models": [
                {
                    "model_id": "model_abc123",
                    "keyword": keyword or "artificial intelligence",
                    "model_type": model_type or "n_beats",
                    "created_at": "2024-01-15T10:30:00Z",
                    "test_mae": 2.5,
                    "test_rmse": 3.1,
                    "directional_accuracy": 0.85
                }
            ],
            "total_count": 1,
            "timestamp": "2024-01-15T10:50:00Z"
        }
    
    def health(self) -> Dict[str, Any]:
        """Mock health check implementation."""
        return {
            "status": "healthy",
            "service": "Google Trends Quantile Forecaster API",
            "version": "v1",
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def cache_stats(self) -> Dict[str, Any]:
        """Mock cache stats implementation."""
        return {
            "status": "success",
            "cache_stats": {
                "cache_size": 150,
                "hit_rate": 0.85,
                "miss_rate": 0.15,
                "total_requests": 1000
            },
            "timestamp": "2024-01-15T10:30:00Z"
        }
    
    def cache_clear(self) -> Dict[str, Any]:
        """Mock cache clear implementation."""
        return {
            "status": "success",
            "message": "Trends cache cleared successfully",
            "timestamp": "2024-01-15T10:30:00Z"
        }


class TestForecasterServiceInterface:
    """Test the Forecaster Service Interface contract compliance."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.forecaster: ForecasterServiceInterface = MockForecasterService()
    
    def test_predict_method_signature(self):
        """Test that predict method has correct signature and returns expected shape."""
        result = self.forecaster.predict("model_abc123", forecast_horizon=25)
        
        # Check response structure
        assert "status" in result
        assert "forecast" in result
        assert "model_performance" in result
        assert "model_info" in result
        assert "timestamp" in result
        
        # Check forecast structure
        forecast = result["forecast"]
        assert "values" in forecast
        assert "dates" in forecast
        assert "confidence_intervals" in forecast
        
        # Check model_info structure
        model_info = result["model_info"]
        assert "model_id" in model_info
        assert "keyword" in model_info
        assert "model_type" in model_info
        assert "forecast_horizon" in model_info
    
    def test_trends_summary_method_signature(self):
        """Test that trends_summary method has correct signature and returns expected shape."""
        result = self.forecaster.trends_summary(["artificial intelligence", "machine learning"])
        
        # Check response structure
        assert "status" in result
        assert "summary" in result
        assert "timestamp" in result
        
        # Check summary structure
        summary = result["summary"]
        assert "keywords" in summary
        assert "timeframe" in summary
        assert "geo" in summary
        assert "statistics" in summary
    
    def test_compare_method_signature(self):
        """Test that compare method has correct signature and returns expected shape."""
        result = self.forecaster.compare(["chatgpt", "claude"])
        
        # Check response structure
        assert "status" in result
        assert "comparison" in result
        assert "timestamp" in result
        
        # Check comparison structure
        comparison = result["comparison"]
        assert "keywords" in comparison
        assert "timeframe" in comparison
        assert "geo" in comparison
        assert "comparison_data" in comparison
    
    def test_train_method_signature(self):
        """Test that train method has correct signature and returns expected shape."""
        time_series_data = [70.0] * 52  # Minimum required data points
        dates = [f"2023-{i:02d}-01" for i in range(1, 53)]
        
        result = self.forecaster.train(
            keyword="artificial intelligence",
            time_series_data=time_series_data,
            dates=dates,
            model_type="n_beats"
        )
        
        # Check response structure
        assert "status" in result
        assert "model_id" in result
        assert "keyword" in result
        assert "model_type" in result
        assert "evaluation_metrics" in result
        assert "timestamp" in result
        
        # Check evaluation_metrics structure
        metrics = result["evaluation_metrics"]
        assert "test_metrics" in metrics
        assert "train_metrics" in metrics
        assert "data_info" in metrics
        assert "training_info" in metrics
    
    def test_evaluate_method_signature(self):
        """Test that evaluate method has correct signature and returns expected shape."""
        result = self.forecaster.evaluate("model_abc123")
        
        # Check response structure
        assert "status" in result
        assert "model_id" in result
        assert "evaluation_metrics" in result
        assert "timestamp" in result
        
        # Check evaluation_metrics structure
        metrics = result["evaluation_metrics"]
        assert "test_metrics" in metrics
        assert "train_metrics" in metrics
        assert "data_info" in metrics
        assert "training_info" in metrics
    
    def test_list_models_method_signature(self):
        """Test that list_models method has correct signature and returns expected shape."""
        result = self.forecaster.list_models(limit=10, offset=0)
        
        # Check response structure
        assert "status" in result
        assert "models" in result
        assert "total_count" in result
        assert "timestamp" in result
        
        # Check models structure
        models = result["models"]
        assert isinstance(models, list)
        if models:
            model = models[0]
            assert "model_id" in model
            assert "keyword" in model
            assert "model_type" in model
            assert "created_at" in model
    
    def test_health_method_signature(self):
        """Test that health method has correct signature and returns expected shape."""
        result = self.forecaster.health()
        
        # Check response structure
        assert "status" in result
        assert "service" in result
        assert "version" in result
        assert "timestamp" in result
    
    def test_cache_stats_method_signature(self):
        """Test that cache_stats method has correct signature and returns expected shape."""
        result = self.forecaster.cache_stats()
        
        # Check response structure
        assert "status" in result
        assert "cache_stats" in result
        assert "timestamp" in result
        
        # Check cache_stats structure
        cache_stats = result["cache_stats"]
        assert "cache_size" in cache_stats
        assert "hit_rate" in cache_stats
        assert "miss_rate" in cache_stats
        assert "total_requests" in cache_stats
    
    def test_cache_clear_method_signature(self):
        """Test that cache_clear method has correct signature and returns expected shape."""
        result = self.forecaster.cache_clear()
        
        # Check response structure
        assert "status" in result
        assert "message" in result
        assert "timestamp" in result


class TestAbstractForecasterService:
    """Test the AbstractForecasterService validation methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = MockForecasterService()
    
    def test_validate_model_id_valid(self):
        """Test model ID validation with valid input."""
        valid_id = "model_abc123"
        result = self.service._validate_model_id(valid_id)
        assert result == valid_id
    
    def test_validate_model_id_invalid(self):
        """Test model ID validation with invalid input."""
        with pytest.raises(ValueError, match="Model ID must be a non-empty string"):
            self.service._validate_model_id("")
        
        with pytest.raises(ValueError, match="Model ID must be a non-empty string"):
            self.service._validate_model_id(None)
        
        with pytest.raises(ValueError, match="Model ID must be a non-empty string"):
            self.service._validate_model_id(123)
    
    def test_validate_keywords_valid(self):
        """Test keywords validation with valid input."""
        valid_keywords = ["artificial intelligence", "machine learning"]
        result = self.service._validate_keywords(valid_keywords)
        assert result == valid_keywords
    
    def test_validate_keywords_invalid(self):
        """Test keywords validation with invalid input."""
        with pytest.raises(ValueError, match="Keywords must be a non-empty list"):
            self.service._validate_keywords([])
        
        with pytest.raises(ValueError, match="Keywords must be a non-empty list"):
            self.service._validate_keywords(None)
        
        with pytest.raises(ValueError, match="Each keyword must be a non-empty string"):
            self.service._validate_keywords(["", "valid"])
        
        with pytest.raises(ValueError, match="Each keyword must be a non-empty string"):
            self.service._validate_keywords([123, "valid"])
    
    def test_validate_time_series_data_valid(self):
        """Test time series data validation with valid input."""
        time_series_data = [70.0] * 52
        dates = [f"2023-{i:02d}-01" for i in range(1, 53)]
        
        result_data, result_dates = self.service._validate_time_series_data(time_series_data, dates)
        assert result_data == time_series_data
        assert result_dates == dates
    
    def test_validate_time_series_data_invalid(self):
        """Test time series data validation with invalid input."""
        # Test insufficient data points
        with pytest.raises(ValueError, match="Time series data must have at least 52 data points"):
            self.service._validate_time_series_data([70.0] * 51, ["2023-01-01"] * 51)
        
        # Test mismatched lengths
        with pytest.raises(ValueError, match="Time series data and dates must have the same length"):
            self.service._validate_time_series_data([70.0] * 52, ["2023-01-01"] * 51)
        
        # Test non-numeric values
        with pytest.raises(ValueError, match="Time series data value at index 0 must be numeric"):
            self.service._validate_time_series_data(["invalid"] + [70.0] * 51, ["2023-01-01"] * 52)
        
        # Test invalid date format
        with pytest.raises(ValueError, match="Date at index 0 must be in YYYY-MM-DD format"):
            self.service._validate_time_series_data([70.0] * 52, ["invalid"] + ["2023-01-01"] * 51)


class TestForecasterModels:
    """Test the forecaster data models and factory functions."""
    
    def test_prediction_request_model(self):
        """Test PredictionRequest model."""
        request = PredictionRequest(model_id="model_abc123", forecast_horizon=25)
        assert request.model_id == "model_abc123"
        assert request.forecast_horizon == 25
        
        # Test default value
        request = PredictionRequest(model_id="model_abc123")
        assert request.forecast_horizon == 25
        
        # Test serialization
        data = request.to_dict()
        assert data["model_id"] == "model_abc123"
        assert data["forecast_horizon"] == 25
        
        # Test deserialization
        new_request = PredictionRequest.from_dict(data)
        assert new_request.model_id == request.model_id
        assert new_request.forecast_horizon == request.forecast_horizon
    
    def test_trends_summary_request_model(self):
        """Test TrendsSummaryRequest model."""
        request = TrendsSummaryRequest(
            keywords=["ai", "ml"],
            timeframe="today 12-m",
            geo="US"
        )
        assert request.keywords == ["ai", "ml"]
        assert request.timeframe == "today 12-m"
        assert request.geo == "US"
        
        # Test serialization
        data = request.to_dict()
        assert data["keywords"] == ["ai", "ml"]
        
        # Test deserialization
        new_request = TrendsSummaryRequest.from_dict(data)
        assert new_request.keywords == request.keywords
    
    def test_training_request_model(self):
        """Test TrainingRequest model."""
        request = TrainingRequest(
            keyword="artificial intelligence",
            time_series_data=[70.0] * 52,
            dates=[f"2023-{i:02d}-01" for i in range(1, 53)],
            model_type="n_beats",
            model_parameters={"n_epochs": 100}
        )
        assert request.keyword == "artificial intelligence"
        assert request.model_type == "n_beats"
        assert request.model_parameters["n_epochs"] == 100
        
        # Test serialization with None model_parameters
        request.model_parameters = None
        data = request.to_dict()
        assert data["model_parameters"] == {}
    
    def test_factory_functions(self):
        """Test factory functions for creating responses."""
        # Test prediction response factory
        forecast = {"values": [85, 87], "dates": ["2024-01-15", "2024-01-22"]}
        model_performance = {"test_metrics": {"mae": 2.5}}
        model_info = {"model_id": "model_abc123"}
        
        response = create_prediction_response(forecast, model_performance, model_info)
        assert response.status == "success"
        assert response.forecast == forecast
        assert response.model_performance == model_performance
        assert response.model_info == model_info
        assert isinstance(response.timestamp, str)
        
        # Test trends summary response factory
        summary = {"keywords": ["ai"], "statistics": {"mean": 75.5}}
        response = create_trends_summary_response(summary)
        assert response.status == "success"
        assert response.summary == summary
        
        # Test error response factory
        error_response = create_error_response("VALIDATION_ERROR", "Invalid input")
        assert error_response.status == "error"
        assert error_response.error_code == "VALIDATION_ERROR"
        assert error_response.message == "Invalid input"


class TestInterfaceCompliance:
    """Test that the interface complies with the contract."""
    
    def test_interface_methods_exist(self):
        """Test that all required methods exist in the interface."""
        forecaster: ForecasterServiceInterface = MockForecasterService()
        
        # Check that all methods exist and are callable
        assert hasattr(forecaster, 'predict')
        assert hasattr(forecaster, 'trends_summary')
        assert hasattr(forecaster, 'compare')
        assert hasattr(forecaster, 'train')
        assert hasattr(forecaster, 'evaluate')
        assert hasattr(forecaster, 'list_models')
        assert hasattr(forecaster, 'health')
        assert hasattr(forecaster, 'cache_stats')
        assert hasattr(forecaster, 'cache_clear')
        
        # Check that methods are callable
        assert callable(forecaster.predict)
        assert callable(forecaster.trends_summary)
        assert callable(forecaster.compare)
        assert callable(forecaster.train)
        assert callable(forecaster.evaluate)
        assert callable(forecaster.list_models)
        assert callable(forecaster.health)
        assert callable(forecaster.cache_stats)
        assert callable(forecaster.cache_clear)
    
    def test_response_shapes_match_current_api(self):
        """Test that response shapes match current API endpoints."""
        forecaster: ForecasterServiceInterface = MockForecasterService()
        
        # Test prediction response shape
        result = forecaster.predict("model_abc123")
        assert "status" in result
        assert "forecast" in result
        assert "model_performance" in result
        assert "model_info" in result
        assert "timestamp" in result
        
        # Test trends summary response shape
        result = forecaster.trends_summary(["ai"])
        assert "status" in result
        assert "summary" in result
        assert "timestamp" in result
        
        # Test compare response shape
        result = forecaster.compare(["ai", "ml"])
        assert "status" in result
        assert "comparison" in result
        assert "timestamp" in result
        
        # Test training response shape
        time_series_data = [70.0] * 52
        dates = [f"2023-{i:02d}-01" for i in range(1, 53)]
        result = forecaster.train("ai", time_series_data, dates, "n_beats")
        assert "status" in result
        assert "model_id" in result
        assert "keyword" in result
        assert "model_type" in result
        assert "evaluation_metrics" in result
        assert "timestamp" in result 
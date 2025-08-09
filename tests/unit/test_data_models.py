"""
Unit tests for data models
"""

import pytest
from datetime import datetime
from app.models.trend_model import TrendData, TrendsRequest, TrendsResponse
from app.models.prediction_model import ModelMetadata, PredictionResult, TrainingRequest, PredictionRequest
from app.utils.validators import InputValidator
from app.utils.error_handlers import ValidationError


class TestTrendData:
    """Test TrendData model"""
    
    def test_valid_trend_data(self):
        """Test creating valid TrendData"""
        data = TrendData(
            keyword="python",
            dates=["2023-01-01", "2023-01-02"],
            interest_values=[50.0, 60.0],
            timeframe="today 12-m"
        )
        
        assert data.keyword == "python"
        assert data.data_points == 2
        assert data.average_interest == 55.0
        assert data.max_interest == 60.0
        assert data.min_interest == 50.0
    
    def test_invalid_keyword(self):
        """Test TrendData with invalid keyword"""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            TrendData(
                keyword="",
                dates=["2023-01-01"],
                interest_values=[50.0]
            )
    
    def test_mismatched_data_lengths(self):
        """Test TrendData with mismatched dates and values"""
        with pytest.raises(ValueError, match="Dates and interest values must have same length"):
            TrendData(
                keyword="python",
                dates=["2023-01-01", "2023-01-02"],
                interest_values=[50.0]
            )
    
    def test_empty_data(self):
        """Test TrendData with empty data"""
        with pytest.raises(ValueError, match="At least one data point is required"):
            TrendData(
                keyword="python",
                dates=[],
                interest_values=[]
            )
    
    def test_invalid_interest_values(self):
        """Test TrendData with invalid interest values"""
        with pytest.raises(ValueError, match="Interest value at index 0 must be between 0 and 100"):
            TrendData(
                keyword="python",
                dates=["2023-01-01"],
                interest_values=[150.0]
            )
    
    def test_to_dict(self):
        """Test TrendData to_dict method"""
        data = TrendData(
            keyword="python",
            dates=["2023-01-01"],
            interest_values=[50.0]
        )
        
        result = data.to_dict()
        assert result['keyword'] == "python"
        assert result['data_points'] == 1
        assert result['average_interest'] == 50.0


class TestTrendsRequest:
    """Test TrendsRequest model"""
    
    def test_valid_trends_request(self):
        """Test creating valid TrendsRequest"""
        request = TrendsRequest(
            keywords=["python", "javascript"],
            timeframe="today 12-m",
            geo="US"
        )
        
        assert request.keywords == ["python", "javascript"]
        assert request.timeframe == "today 12-m"
        assert request.geo == "US"
    
    def test_empty_keywords(self):
        """Test TrendsRequest with empty keywords"""
        with pytest.raises(ValueError, match="At least one keyword is required"):
            TrendsRequest(keywords=[])
    
    def test_too_many_keywords(self):
        """Test TrendsRequest with too many keywords"""
        with pytest.raises(ValueError, match="Maximum 5 keywords allowed per request"):
            TrendsRequest(keywords=["a", "b", "c", "d", "e", "f"])
    
    def test_empty_keyword(self):
        """Test TrendsRequest with empty keyword"""
        with pytest.raises(ValueError, match="Keyword at index 0 cannot be empty"):
            TrendsRequest(keywords=[""])
    
    def test_long_keyword(self):
        """Test TrendsRequest with too long keyword"""
        long_keyword = "a" * 101
        with pytest.raises(ValueError, match="Keyword at index 0 is too long"):
            TrendsRequest(keywords=[long_keyword])


class TestModelMetadata:
    """Test ModelMetadata model"""
    
    def test_valid_model_metadata(self):
        """Test creating valid ModelMetadata"""
        metadata = ModelMetadata(
            model_id="test-id",
            keyword="python",
            training_date=datetime.utcnow(),
            parameters={"epochs": 100},
            metrics={"loss": 0.1}
        )
        
        assert metadata.model_id == "test-id"
        assert metadata.keyword == "python"
        assert metadata.status == "completed"
    
    def test_auto_generated_id(self):
        """Test ModelMetadata with auto-generated ID"""
        metadata = ModelMetadata(
            keyword="python",
            training_date=datetime.utcnow(),
            parameters={},
            metrics={}
        )
        
        assert metadata.model_id is not None
        assert len(metadata.model_id) > 0
    
    def test_invalid_status(self):
        """Test ModelMetadata with invalid status"""
        with pytest.raises(ValueError, match="Status must be one of"):
            ModelMetadata(
                keyword="python",
                training_date=datetime.utcnow(),
                parameters={},
                metrics={},
                status="invalid"
            )


class TestTrainingRequest:
    """Test TrainingRequest model"""
    
    def test_valid_training_request(self):
        """Test creating valid TrainingRequest"""
        request = TrainingRequest(
            keyword="python",
            time_series_data=[50.0] * 20,  # 20 data points
            model_params={"epochs": 100}
        )
        
        assert request.keyword == "python"
        assert len(request.time_series_data) == 20
        assert request.model_params["epochs"] == 100
    
    def test_empty_keyword(self):
        """Test TrainingRequest with empty keyword"""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            TrainingRequest(
                keyword="",
                time_series_data=[50.0] * 20,
                model_params={}
            )
    
    def test_insufficient_data_points(self):
        """Test TrainingRequest with insufficient data points"""
        with pytest.raises(ValueError, match="At least 10 data points required for training"):
            TrainingRequest(
                keyword="python",
                time_series_data=[50.0] * 5,
                model_params={}
            )
    
    def test_too_many_data_points(self):
        """Test TrainingRequest with too many data points"""
        with pytest.raises(ValueError, match="Maximum 10,000 data points allowed"):
            TrainingRequest(
                keyword="python",
                time_series_data=[50.0] * 15000,
                model_params={}
            )


class TestPredictionRequest:
    """Test PredictionRequest model"""
    
    def test_valid_prediction_request(self):
        """Test creating valid PredictionRequest"""
        request = PredictionRequest(
            model_id="test-id",
            prediction_weeks=25
        )
        
        assert request.model_id == "test-id"
        assert request.prediction_weeks == 25
    
    def test_empty_model_id(self):
        """Test PredictionRequest with empty model ID"""
        with pytest.raises(ValueError, match="Model ID cannot be empty"):
            PredictionRequest(model_id="")
    
    def test_invalid_prediction_weeks(self):
        """Test PredictionRequest with invalid prediction weeks"""
        with pytest.raises(ValueError, match="Prediction weeks must be between 1 and 100"):
            PredictionRequest(
                model_id="test-id",
                prediction_weeks=0
            )


class TestInputValidator:
    """Test InputValidator class"""
    
    def test_validate_keywords_valid(self):
        """Test validating valid keywords"""
        keywords = ["python", "javascript"]
        result = InputValidator.validate_keywords(keywords)
        assert result == ["python", "javascript"]
    
    def test_validate_keywords_empty(self):
        """Test validating empty keywords"""
        with pytest.raises(ValidationError, match="Keywords list cannot be empty"):
            InputValidator.validate_keywords([])
    
    def test_validate_keywords_too_many(self):
        """Test validating too many keywords"""
        with pytest.raises(ValidationError, match="Maximum 5 keywords allowed per request"):
            InputValidator.validate_keywords(["a", "b", "c", "d", "e", "f"])
    
    def test_validate_timeframe_valid(self):
        """Test validating valid timeframe"""
        result = InputValidator.validate_timeframe("today 12-m")
        assert result == "today 12-m"
    
    def test_validate_timeframe_invalid(self):
        """Test validating invalid timeframe"""
        with pytest.raises(ValidationError, match="Invalid timeframe"):
            InputValidator.validate_timeframe("invalid")
    
    def test_validate_training_parameters_valid(self):
        """Test validating valid training parameters"""
        params = {"epochs": 100, "batch_size": 10}
        result = InputValidator.validate_training_parameters(params)
        assert result["epochs"] == 100
        assert result["batch_size"] == 10
    
    def test_validate_training_parameters_invalid_epochs(self):
        """Test validating invalid epochs"""
        params = {"epochs": 0}
        with pytest.raises(ValidationError, match="Epochs must be between 1 and 1000"):
            InputValidator.validate_training_parameters(params)
    
    def test_validate_model_id_valid(self):
        """Test validating valid model ID"""
        model_id = "12345678-1234-1234-1234-123456789abc"
        result = InputValidator.validate_model_id(model_id)
        assert result == model_id
    
    def test_validate_model_id_invalid(self):
        """Test validating invalid model ID"""
        with pytest.raises(ValidationError, match="Model ID must be a valid UUID format"):
            InputValidator.validate_model_id("invalid-id")
    
    def test_validate_trends_request_valid(self):
        """Test validating valid trends request"""
        data = {
            "keywords": ["python"],
            "timeframe": "today 12-m",
            "geo": "US"
        }
        result = InputValidator.validate_trends_request(data)
        assert isinstance(result, TrendsRequest)
        assert result.keywords == ["python"]
    
    def test_validate_training_request_valid(self):
        """Test validating valid training request"""
        data = {
            "keyword": "python",
            "time_series_data": [50.0] * 20,
            "model_params": {"epochs": 100}
        }
        result = InputValidator.validate_training_request(data)
        assert isinstance(result, TrainingRequest)
        assert result.keyword == "python"
    
    def test_validate_prediction_request_valid(self):
        """Test validating valid prediction request"""
        data = {"prediction_weeks": 25}
        model_id = "12345678-1234-1234-1234-123456789abc"
        result = InputValidator.validate_prediction_request(data, model_id)
        assert isinstance(result, PredictionRequest)
        assert result.prediction_weeks == 25 
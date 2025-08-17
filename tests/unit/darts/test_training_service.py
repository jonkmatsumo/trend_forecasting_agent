"""
Unit tests for Darts Model Service.
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import numpy as np

from app.services.darts.training_service import TrainingService
from app.models.darts.darts_models import (
    ModelTrainingRequest, ModelEvaluationMetrics, ModelType
)
from app.utils.error_handlers import ValidationError, ModelError


class TestTrainingService:
    """Test cases for TrainingService."""
    
    @pytest.fixture
    def temp_models_dir(self):
        """Create a temporary directory for models."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def model_service(self, temp_models_dir):
        """Create a TrainingService instance with temporary directory."""
        return TrainingService(models_dir=temp_models_dir)
    
    @pytest.fixture
    def sample_training_request(self):
        """Create a sample training request."""
        # Generate 52 weeks of sample data
        dates = []
        values = []
        start_date = datetime(2023, 1, 1)

        for i in range(52):
            dates.append((start_date + timedelta(weeks=i)).strftime("%Y-%m-%d"))
            values.append(50 + i + (i % 7) * 2)  # Some trend + weekly pattern

        return ModelTrainingRequest(
            keyword="artificial intelligence",
            time_series_data=values,
            dates=dates,
            model_type=ModelType.LSTM,
            train_test_split=0.8,
            forecast_horizon=25,
            validation_strategy="holdout",
            model_parameters={
                "input_chunk_length": 12,
                "n_epochs": 5,  # Small number for testing
                "batch_size": 4
            }
        )
    
    def test_initialization(self, temp_models_dir):
        """Test service initialization."""
        service = TrainingService(models_dir=temp_models_dir)
        
        assert service.models_dir == Path(temp_models_dir)
        assert service.models_dir.exists()
        assert service.experiment_name == "google_trends_forecaster"
        assert len(service.model_mapping) == 13  # All model types (excluding prophet)
    
    def test_model_mapping_contains_all_types(self, model_service):
        """Test that model mapping contains all supported model types."""
        expected_types = [
            "lstm", "gru", "tcn", "transformer",
            "n_beats", "tft", "arima", "exponential_smoothing",
            "random_forest", "auto_arima",
            "auto_ets", "auto_theta", "auto_ces"
        ]
        
        for model_type in expected_types:
            assert model_type in model_service.model_mapping
    
    def test_validate_training_request_valid(self, model_service, sample_training_request):
        """Test validation of valid training request."""
        # Should not raise any exception
        model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_empty_keyword(self, model_service, sample_training_request):
        """Test validation with empty keyword."""
        sample_training_request.keyword = ""
        
        with pytest.raises(ValidationError, match="Keyword cannot be empty"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_insufficient_data(self, model_service, sample_training_request):
        """Test validation with insufficient data points."""
        sample_training_request.time_series_data = [1, 2, 3, 4, 5]  # Only 5 points
        sample_training_request.dates = ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22", "2023-01-29"]
        
        with pytest.raises(ValidationError, match="At least 52 data points required"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_missing_dates(self, model_service, sample_training_request):
        """Test validation with missing dates."""
        sample_training_request.dates = None
        
        with pytest.raises(ValidationError, match="Dates are required"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_mismatched_lengths(self, model_service, sample_training_request):
        """Test validation with mismatched data and dates lengths."""
        sample_training_request.dates = sample_training_request.dates[:-1]  # Remove one date
        
        with pytest.raises(ValidationError, match="Dates and time_series_data must have same length"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_invalid_split(self, model_service, sample_training_request):
        """Test validation with invalid train/test split."""
        sample_training_request.train_test_split = 0.05  # Too small
        
        with pytest.raises(ValidationError, match="train_test_split must be between 0.1 and 0.9"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_invalid_horizon(self, model_service, sample_training_request):
        """Test validation with invalid forecast horizon."""
        sample_training_request.forecast_horizon = 0
        
        with pytest.raises(ValidationError, match="forecast_horizon must be positive"):
            model_service._validate_training_request(sample_training_request)
    
    def test_validate_training_request_invalid_strategy(self, model_service, sample_training_request):
        """Test validation with invalid validation strategy."""
        sample_training_request.validation_strategy = "invalid_strategy"
        
        with pytest.raises(ValidationError, match="validation_strategy must be one of"):
            model_service._validate_training_request(sample_training_request)
    
    def test_create_time_series(self, model_service, sample_training_request):
        """Test creation of TimeSeries object."""
        time_series = model_service._create_time_series(sample_training_request)
        
        assert time_series is not None
        assert len(time_series) == len(sample_training_request.time_series_data)
        assert time_series.time_index[0] == datetime(2023, 1, 1)
    
    def test_split_time_series(self, model_service, sample_training_request):
        """Test splitting time series into train/test sets."""
        time_series = model_service._create_time_series(sample_training_request)
        train_series, test_series = model_service._split_time_series(time_series, 0.8)
        
        expected_train_length = int(len(time_series) * 0.8)
        expected_test_length = len(time_series) - expected_train_length
        
        assert len(train_series) == expected_train_length
        assert len(test_series) == expected_test_length

    @patch('app.services.darts.training_service.RNNModel')
    def test_create_model_lstm(self, mock_rnn, model_service, sample_training_request):
        """Test creating LSTM model."""
        mock_model = Mock()
        mock_rnn.return_value = mock_model

        # Patch the model_mapping to use our mock
        model_service.model_mapping["lstm"] = mock_rnn

        model = model_service._create_model("lstm", {"input_chunk_length": 12})

        # Verify the model was created with correct parameters
        mock_rnn.assert_called_once()
        call_args = mock_rnn.call_args[1]  # Get keyword arguments
        assert call_args["model"] == "LSTM"
        assert call_args["input_chunk_length"] == 12
    
    def test_create_model_unsupported_type(self, model_service):
        """Test creating model with unsupported type."""
        with pytest.raises(ModelError, match="Failed to create model unsupported_model"):
            model_service._create_model("unsupported_model", {})
    
    @patch('app.services.darts.training_service.mae')
    @patch('app.services.darts.training_service.rmse')
    @patch('app.services.darts.training_service.mape')
    def test_evaluate_model(self, mock_mape, mock_rmse, mock_mae, model_service, sample_training_request):
        """Test model evaluation."""
        # Setup mocks
        mock_mae.return_value = 2.5
        mock_rmse.return_value = 3.2
        mock_mape.return_value = 8.5
        
        # Create mock model and series
        mock_model = Mock()
        mock_model.predict.return_value = Mock(values=lambda: [[1, 2, 3, 4, 5]])
        
        train_series = Mock()
        test_series = Mock()
        train_series.__len__ = Mock(return_value=40)
        test_series.__len__ = Mock(return_value=10)
        test_series.values.return_value = [[1, 2, 3, 4, 5]]
        
        training_time = 30.5
        
        metrics = model_service._evaluate_model(
            mock_model, train_series, test_series, training_time, 
            "test_model_123", "test_keyword", "lstm"
        )
        
        assert isinstance(metrics, ModelEvaluationMetrics)
        assert metrics.test_mae == 2.5
        assert metrics.test_rmse == 3.2
        assert metrics.test_mape == 8.5
        assert metrics.training_time_seconds == 30.5
        assert metrics.train_samples == 40
        assert metrics.test_samples == 10
        assert metrics.total_samples == 50
    
    def test_calculate_directional_accuracy(self, model_service):
        """Test directional accuracy calculation."""
        # Create mock series with known direction changes
        mock_actual = Mock()
        mock_actual.values.return_value = np.array([[1, 2, 3, 4, 5]])  # Increasing trend
        
        mock_predicted = Mock()
        mock_predicted.values.return_value = np.array([[1, 2, 3, 4, 5]])  # Same increasing trend
        
        accuracy = model_service._calculate_directional_accuracy(mock_actual, mock_predicted)
        
        assert accuracy == 100.0  # Perfect directional accuracy
    
    def test_calculate_directional_accuracy_mixed(self, model_service):
        """Test directional accuracy with mixed directions."""
        # Create mock series with mixed direction changes
        mock_actual = Mock()
        mock_actual.values.return_value = np.array([[1, 2, 1, 2, 1]])  # Up, down, up, down
        
        mock_predicted = Mock()
        mock_predicted.values.return_value = np.array([[1, 2, 1, 2, 1]])  # Same pattern
        
        accuracy = model_service._calculate_directional_accuracy(mock_actual, mock_predicted)
        
        assert accuracy == 100.0  # Perfect directional accuracy
    
    def test_calculate_confidence_coverage(self, model_service):
        """Test confidence interval coverage calculation."""
        mock_actual = Mock()
        mock_predicted = Mock()
        
        coverage = model_service._calculate_confidence_coverage(mock_actual, mock_predicted)
        
        assert coverage == 0.0  # Default value for now

    @patch('pickle.dump')
    def test_save_model(self, mock_pickle_dump, model_service, sample_training_request, temp_models_dir):
        """Test saving model and metadata."""
        mock_model = Mock()
        model_id = "test_model_123"

        metrics = ModelEvaluationMetrics(
            model_id="test_model_123",
            keyword="test",
            model_type=ModelType.LSTM,
            train_mae=2.5, train_rmse=3.2, train_mape=8.5,
            test_mae=2.5, test_rmse=3.2, test_mape=8.5,
            directional_accuracy=0.75, coverage_95=0.0,
            train_samples=40, test_samples=10, total_samples=50,
            training_time_seconds=30.5
        )

        model_service._save_model(mock_model, model_id, sample_training_request, metrics)
        
        # Check that model directory was created
        model_dir = Path(temp_models_dir) / model_id
        assert model_dir.exists()
        
        # Check that files were created
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "metadata.json").exists()
        assert (model_dir / "evaluation.json").exists()
        assert (model_dir / "training_request.json").exists()
    
    def test_load_model(self, model_service, temp_models_dir):
        """Test loading a saved model."""
        model_id = "test_model_123"
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Create a simple object and save it
        test_model = {"type": "test_model", "parameters": {"test": "value"}}
        model_path = model_dir / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(test_model, f)
        
        # Test loading
        loaded_model = model_service.load_model(model_id)
        assert loaded_model == test_model
    
    def test_load_model_not_found(self, model_service):
        """Test loading non-existent model."""
        with pytest.raises(ModelError, match="Model test_nonexistent not found"):
            model_service.load_model("test_nonexistent")
    
    def test_get_model_metadata(self, model_service, temp_models_dir):
        """Test loading model metadata."""
        model_id = "test_model_123"
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Create sample metadata
        metadata = {
            "model_id": model_id,
            "keyword": "test",
            "training_date": datetime.now().isoformat(),
            "parameters": {"input_chunk_length": 12},
            "metrics": {"test_mae": 2.5, "test_rmse": 3.2, "test_mape": 8.5},
            "model_path": "test/path",
            "model_type": "lstm",
            "status": "completed",
            "data_points": 100,
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=str)
        
        # Test loading
        loaded_metadata = model_service.get_model_metadata(model_id)
        assert loaded_metadata["keyword"] == metadata["keyword"]
        assert loaded_metadata["model_id"] == metadata["model_id"]
    
    def test_get_model_metadata_not_found(self, model_service):
        """Test loading non-existent metadata."""
        with pytest.raises(ModelError, match="Model metadata test_nonexistent not found"):
            model_service.get_model_metadata("test_nonexistent")
    
    def test_get_evaluation_metrics(self, model_service, temp_models_dir):
        """Test loading evaluation metrics."""
        model_id = "test_model_123"
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Create sample metrics
        metrics = ModelEvaluationMetrics(
            model_id="test_model_123",
            keyword="test",
            model_type=ModelType.LSTM,
            train_mae=2.5, train_rmse=3.2, train_mape=8.5,
            test_mae=2.5, test_rmse=3.2, test_mape=8.5,
            directional_accuracy=0.75, coverage_95=0.0,
            train_samples=40, test_samples=10, total_samples=50,
            training_time_seconds=30.5
        )
        
        metrics_path = model_dir / "evaluation.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics.to_dict(), f, default=str)
        
        # Test loading
        loaded_metrics = model_service.get_evaluation_metrics(model_id)
        assert loaded_metrics.test_mae == metrics.test_mae
        assert loaded_metrics.test_rmse == metrics.test_rmse
    
    def test_get_evaluation_metrics_not_found(self, model_service):
        """Test loading non-existent evaluation metrics."""
        with pytest.raises(ModelError, match="Model evaluation metrics test_nonexistent not found"):
            model_service.get_evaluation_metrics("test_nonexistent")
    
    def test_list_models_empty(self, model_service):
        """Test listing models when directory is empty."""
        models = model_service.list_models()
        assert models == []
    
    def test_list_models_with_models(self, model_service, temp_models_dir):
        """Test listing models when models exist."""
        # Create a model directory with metadata
        model_id = "test_model_123"
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        
        metadata = {
            "model_id": model_id,
            "keyword": "test",
            "training_date": datetime.now().isoformat(),
            "parameters": {"input_chunk_length": 12},
            "metrics": {"test_mae": 2.5, "test_rmse": 3.2, "test_mape": 8.5},
            "model_path": "test/path",
            "model_type": "lstm",
            "status": "completed",
            "data_points": 100,
            "created_at": datetime.now().isoformat()
        }
        
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, default=str)
        
        # Test listing
        models = model_service.list_models()
        assert len(models) == 1
        assert models[0]["model_id"] == model_id
    
    def test_delete_model_success(self, model_service, temp_models_dir):
        """Test successful model deletion."""
        model_id = "test_model_123"
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        
        # Create some files
        (model_dir / "model.pkl").touch()
        (model_dir / "metadata.json").touch()
        
        # Test deletion
        result = model_service.delete_model(model_id)
        assert result is True
        assert not model_dir.exists()
    
    def test_delete_model_not_found(self, model_service):
        """Test deleting non-existent model."""
        result = model_service.delete_model("test_nonexistent")
        assert result is False
    
    @patch('app.services.darts.training_service.mlflow')
    def test_log_to_mlflow(self, mock_mlflow, model_service, sample_training_request, temp_models_dir):
        """Test MLflow logging."""
        model_id = "test_model_123"
        metrics = ModelEvaluationMetrics(
            model_id="test_model_123",
            keyword="test",
            model_type=ModelType.LSTM,
            train_mae=2.5, train_rmse=3.2, train_mape=8.5,
            test_mae=2.5, test_rmse=3.2, test_mape=8.5,
            directional_accuracy=0.75, coverage_95=0.0,
            train_samples=40, test_samples=10, total_samples=50,
            training_time_seconds=30.5
        )
        training_time = 30.5
        
        # Create model directory and files for logging
        model_dir = Path(temp_models_dir) / model_id
        model_dir.mkdir(exist_ok=True)
        (model_dir / "model.pkl").touch()
        (model_dir / "metadata.json").touch()
        
        # Test logging (should not raise exception)
        model_service._log_to_mlflow(model_id, sample_training_request, metrics, training_time)
        
        # Verify MLflow was called
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
    
    @patch('app.services.darts.training_service.mlflow')
    def test_log_to_mlflow_error_handling(self, mock_mlflow, model_service, sample_training_request):
        """Test MLflow logging error handling."""
        mock_mlflow.set_experiment.side_effect = Exception("MLflow error")
        
        model_id = "test_model_123"
        metrics = ModelEvaluationMetrics(
            model_id="test_model_123",
            keyword="test",
            model_type=ModelType.LSTM,
            train_mae=2.5, train_rmse=3.2, train_mape=8.5,
            test_mae=2.5, test_rmse=3.2, test_mape=8.5,
            directional_accuracy=0.75, coverage_95=0.0,
            train_samples=40, test_samples=10, total_samples=50,
            training_time_seconds=30.5
        )
        training_time = 30.5
        
        # Should not raise exception, just log warning
        model_service._log_to_mlflow(model_id, sample_training_request, metrics, training_time) 
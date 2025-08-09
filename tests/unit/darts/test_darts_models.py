"""
Unit tests for Darts data models
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from app.models.darts.darts_models import (
    ModelType, DartsTimeSeriesData, ModelTrainingRequest, 
    ModelEvaluationMetrics, ForecastResult, DartsModelValidator,
    generate_model_id, DEFAULT_MODEL_PARAMETERS
)
from darts import TimeSeries
import pandas as pd
import numpy as np


class TestModelType:
    """Test ModelType enum"""
    
    def test_model_type_values(self):
        """Test that all model types have correct values"""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.TCN.value == "tcn"
        assert ModelType.TRANSFORMER.value == "transformer"
        assert ModelType.PROPHET.value == "prophet"
        assert ModelType.ARIMA.value == "arima"
        assert ModelType.EXPONENTIAL_SMOOTHING.value == "exponential_smoothing"
        assert ModelType.RANDOM_FOREST.value == "random_forest"
        assert ModelType.N_BEATS.value == "n_beats"
        assert ModelType.TFT.value == "tft"
        assert ModelType.GRU.value == "gru"
        assert ModelType.AUTO_ARIMA.value == "auto_arima"
        assert ModelType.AUTO_ETS.value == "auto_ets"
        assert ModelType.AUTO_THETA.value == "auto_theta"
        assert ModelType.AUTO_CES.value == "auto_ces"
    
    def test_model_type_from_string(self):
        """Test creating ModelType from string"""
        assert ModelType("lstm") == ModelType.LSTM
        assert ModelType("n_beats") == ModelType.N_BEATS
        assert ModelType("tft") == ModelType.TFT
    
    def test_invalid_model_type(self):
        """Test that invalid model type raises ValueError"""
        with pytest.raises(ValueError):
            ModelType("invalid_model")


class TestDartsTimeSeriesData:
    """Test DartsTimeSeriesData class"""
    
    def setup_method(self):
        """Set up test data"""
        self.dates = [datetime(2023, 1, 1) + timedelta(weeks=i) for i in range(52)]
        self.values = [50 + i * 0.5 + np.random.normal(0, 2) for i in range(52)]
        self.time_series = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(self.dates), self.values
        )
    
    def test_valid_darts_time_series_data(self):
        """Test creating valid DartsTimeSeriesData"""
        data = DartsTimeSeriesData(
            keyword="test_keyword",
            time_series=self.time_series,
            dates=self.dates,
            values=self.values
        )
        
        assert data.keyword == "test_keyword"
        assert len(data.dates) == 52
        assert len(data.values) == 52
        assert data.frequency == "W"
        assert data.metadata == {}
    
    def test_empty_keyword(self):
        """Test that empty keyword raises ValueError"""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            DartsTimeSeriesData(
                keyword="",
                time_series=self.time_series,
                dates=self.dates,
                values=self.values
            )
    
    def test_mismatched_lengths(self):
        """Test that mismatched dates and values raises ValueError"""
        with pytest.raises(ValueError, match="Dates and values must have same length"):
            DartsTimeSeriesData(
                keyword="test",
                time_series=self.time_series,
                dates=self.dates,
                values=self.values[:-1]  # One less value
            )
    
    def test_insufficient_data_points(self):
        """Test that insufficient data points raises ValueError"""
        short_dates = self.dates[:50]  # Less than 52 points
        short_values = self.values[:50]
        short_ts = TimeSeries.from_times_and_values(
            pd.DatetimeIndex(short_dates), short_values
        )
        
        with pytest.raises(ValueError, match="Time series must have at least 52 data points"):
            DartsTimeSeriesData(
                keyword="test",
                time_series=short_ts,
                dates=short_dates,
                values=short_values
            )
    
    def test_to_dict(self):
        """Test to_dict method"""
        data = DartsTimeSeriesData(
            keyword="test_keyword",
            time_series=self.time_series,
            dates=self.dates,
            values=self.values,
            metadata={"source": "test"}
        )
        
        result = data.to_dict()
        assert result["keyword"] == "test_keyword"
        assert result["frequency"] == "W"
        assert result["length"] == 52
        assert result["metadata"] == {"source": "test"}
        assert len(result["dates"]) == 52
        assert len(result["values"]) == 52


class TestModelTrainingRequest:
    """Test ModelTrainingRequest class"""
    
    def setup_method(self):
        """Set up test data"""
        self.dates = [f"2023-{i+1:02d}-01" for i in range(52)]
        self.values = [50 + i * 0.5 + np.random.normal(0, 2) for i in range(52)]
    
    def test_valid_model_training_request(self):
        """Test creating valid ModelTrainingRequest"""
        request = ModelTrainingRequest(
            keyword="test_keyword",
            time_series_data=self.values,
            dates=self.dates,
            model_type=ModelType.LSTM
        )
        
        assert request.keyword == "test_keyword"
        assert len(request.time_series_data) == 52
        assert request.model_type == ModelType.LSTM
        assert request.train_test_split == 0.8
        assert request.forecast_horizon == 25
        assert request.validation_strategy == "holdout"
    
    def test_empty_keyword(self):
        """Test that empty keyword raises ValueError"""
        with pytest.raises(ValueError, match="Keyword cannot be empty"):
            ModelTrainingRequest(
                keyword="",
                time_series_data=self.values,
                dates=self.dates,
                model_type=ModelType.LSTM
            )
    
    def test_insufficient_data_points(self):
        """Test that insufficient data points raises ValueError"""
        with pytest.raises(ValueError, match="Time series data must have at least 52 data points"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values[:50],  # Less than 52 points
                dates=self.dates[:50],
                model_type=ModelType.LSTM
            )
    
    def test_mismatched_lengths(self):
        """Test that mismatched dates and data raises ValueError"""
        with pytest.raises(ValueError, match="Dates and time_series_data must have same length"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values,
                dates=self.dates[:-1],  # One less date
                model_type=ModelType.LSTM
            )
    
    def test_invalid_train_test_split(self):
        """Test that invalid train_test_split raises ValueError"""
        with pytest.raises(ValueError, match="train_test_split must be between 0.1 and 0.9"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values,
                dates=self.dates,
                model_type=ModelType.LSTM,
                train_test_split=0.05  # Too low
            )
        
        with pytest.raises(ValueError, match="train_test_split must be between 0.1 and 0.9"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values,
                dates=self.dates,
                model_type=ModelType.LSTM,
                train_test_split=0.95  # Too high
            )
    
    def test_invalid_forecast_horizon(self):
        """Test that invalid forecast_horizon raises ValueError"""
        with pytest.raises(ValueError, match="forecast_horizon must be positive"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values,
                dates=self.dates,
                model_type=ModelType.LSTM,
                forecast_horizon=0
            )
    
    def test_invalid_validation_strategy(self):
        """Test that invalid validation_strategy raises ValueError"""
        with pytest.raises(ValueError, match="validation_strategy must be one of"):
            ModelTrainingRequest(
                keyword="test",
                time_series_data=self.values,
                dates=self.dates,
                model_type=ModelType.LSTM,
                validation_strategy="invalid"
            )
    
    def test_to_dict(self):
        """Test to_dict method"""
        request = ModelTrainingRequest(
            keyword="test_keyword",
            time_series_data=self.values,
            dates=self.dates,
            model_type=ModelType.N_BEATS,
            train_test_split=0.7,
            forecast_horizon=30,
            model_parameters={"n_epochs": 100},
            validation_strategy="rolling_window"
        )
        
        result = request.to_dict()
        assert result["keyword"] == "test_keyword"
        assert result["model_type"] == "n_beats"
        assert result["train_test_split"] == 0.7
        assert result["forecast_horizon"] == 30
        assert result["model_parameters"] == {"n_epochs": 100}
        assert result["validation_strategy"] == "rolling_window"


class TestModelEvaluationMetrics:
    """Test ModelEvaluationMetrics class"""
    
    def test_valid_model_evaluation_metrics(self):
        """Test creating valid ModelEvaluationMetrics"""
        metrics = ModelEvaluationMetrics(
            model_id="model_123",
            keyword="test_keyword",
            model_type=ModelType.LSTM,
            train_mae=2.1,
            train_rmse=2.8,
            train_mape=2.5,
            test_mae=2.5,
            test_rmse=3.1,
            test_mape=3.2,
            directional_accuracy=0.85,
            coverage_95=0.92,
            train_samples=156,
            test_samples=39,
            total_samples=195,
            training_time_seconds=45.2
        )
        
        assert metrics.model_id == "model_123"
        assert metrics.keyword == "test_keyword"
        assert metrics.model_type == ModelType.LSTM
        assert metrics.test_mae == 2.5
        assert metrics.directional_accuracy == 0.85
        assert metrics.coverage_95 == 0.92
    
    def test_invalid_directional_accuracy(self):
        """Test that invalid directional_accuracy raises ValueError"""
        with pytest.raises(ValueError, match="directional_accuracy must be between 0 and 1"):
            ModelEvaluationMetrics(
                model_id="model_123",
                keyword="test",
                model_type=ModelType.LSTM,
                train_mae=2.1, train_rmse=2.8, train_mape=2.5,
                test_mae=2.5, test_rmse=3.1, test_mape=3.2,
                directional_accuracy=1.5,  # Invalid
                coverage_95=0.92,
                train_samples=156, test_samples=39, total_samples=195,
                training_time_seconds=45.2
            )
    
    def test_invalid_coverage_95(self):
        """Test that invalid coverage_95 raises ValueError"""
        with pytest.raises(ValueError, match="coverage_95 must be between 0 and 1"):
            ModelEvaluationMetrics(
                model_id="model_123",
                keyword="test",
                model_type=ModelType.LSTM,
                train_mae=2.1, train_rmse=2.8, train_mape=2.5,
                test_mae=2.5, test_rmse=3.1, test_mape=3.2,
                directional_accuracy=0.85,
                coverage_95=-0.1,  # Invalid
                train_samples=156, test_samples=39, total_samples=195,
                training_time_seconds=45.2
            )
    
    def test_invalid_sample_counts(self):
        """Test that invalid sample counts raises ValueError"""
        with pytest.raises(ValueError, match="Sample counts must be positive"):
            ModelEvaluationMetrics(
                model_id="model_123",
                keyword="test",
                model_type=ModelType.LSTM,
                train_mae=2.1, train_rmse=2.8, train_mape=2.5,
                test_mae=2.5, test_rmse=3.1, test_mape=3.2,
                directional_accuracy=0.85,
                coverage_95=0.92,
                train_samples=0,  # Invalid
                test_samples=39, total_samples=195,
                training_time_seconds=45.2
            )
    
    def test_negative_training_time(self):
        """Test that negative training time raises ValueError"""
        with pytest.raises(ValueError, match="Training time cannot be negative"):
            ModelEvaluationMetrics(
                model_id="model_123",
                keyword="test",
                model_type=ModelType.LSTM,
                train_mae=2.1, train_rmse=2.8, train_mape=2.5,
                test_mae=2.5, test_rmse=3.1, test_mape=3.2,
                directional_accuracy=0.85,
                coverage_95=0.92,
                train_samples=156, test_samples=39, total_samples=195,
                training_time_seconds=-1.0  # Invalid
            )
    
    def test_to_dict(self):
        """Test to_dict method"""
        metrics = ModelEvaluationMetrics(
            model_id="model_123",
            keyword="test_keyword",
            model_type=ModelType.TFT,
            train_mae=2.1, train_rmse=2.8, train_mape=2.5,
            test_mae=2.5, test_rmse=3.1, test_mape=3.2,
            directional_accuracy=0.85,
            coverage_95=0.92,
            train_samples=156, test_samples=39, total_samples=195,
            training_time_seconds=45.2,
            mlflow_run_id="run_xyz789"
        )
        
        result = metrics.to_dict()
        assert result["model_id"] == "model_123"
        assert result["model_type"] == "tft"
        assert result["test_metrics"]["mae"] == 2.5
        assert result["test_metrics"]["directional_accuracy"] == 0.85
        assert result["training_info"]["mlflow_run_id"] == "run_xyz789"


class TestForecastResult:
    """Test ForecastResult class"""
    
    def setup_method(self):
        """Set up test data"""
        self.metrics = ModelEvaluationMetrics(
            model_id="model_123",
            keyword="test_keyword",
            model_type=ModelType.LSTM,
            train_mae=2.1, train_rmse=2.8, train_mape=2.5,
            test_mae=2.5, test_rmse=3.1, test_mape=3.2,
            directional_accuracy=0.85,
            coverage_95=0.92,
            train_samples=156, test_samples=39, total_samples=195,
            training_time_seconds=45.2
        )
        
        self.forecast_dates = [datetime(2024, 1, 1) + timedelta(weeks=i) for i in range(25)]
        self.forecast_values = [85 + i * 0.5 for i in range(25)]
        self.confidence_intervals = {
            "95%": {
                "lower": [80 + i * 0.5 for i in range(25)],
                "upper": [90 + i * 0.5 for i in range(25)]
            }
        }
    
    def test_valid_forecast_result(self):
        """Test creating valid ForecastResult"""
        result = ForecastResult(
            model_id="model_123",
            keyword="test_keyword",
            forecast_values=self.forecast_values,
            forecast_dates=self.forecast_dates,
            confidence_intervals=self.confidence_intervals,
            model_metrics=self.metrics,
            forecast_horizon=25
        )
        
        assert result.model_id == "model_123"
        assert result.keyword == "test_keyword"
        assert len(result.forecast_values) == 25
        assert len(result.forecast_dates) == 25
        assert result.forecast_horizon == 25
    
    def test_mismatched_forecast_lengths(self):
        """Test that mismatched forecast values and dates raises ValueError"""
        with pytest.raises(ValueError, match="forecast_values and forecast_dates must have same length"):
            ForecastResult(
                model_id="model_123",
                keyword="test_keyword",
                forecast_values=self.forecast_values,
                forecast_dates=self.forecast_dates[:-1],  # One less date
                confidence_intervals=self.confidence_intervals,
                model_metrics=self.metrics,
                forecast_horizon=25
            )
    
    def test_mismatched_horizon(self):
        """Test that mismatched forecast horizon raises ValueError"""
        with pytest.raises(ValueError, match="forecast_values length must match forecast_horizon"):
            ForecastResult(
                model_id="model_123",
                keyword="test_keyword",
                forecast_values=self.forecast_values,
                forecast_dates=self.forecast_dates,
                confidence_intervals=self.confidence_intervals,
                model_metrics=self.metrics,
                forecast_horizon=30  # Different from values length
            )
    
    def test_empty_confidence_intervals(self):
        """Test that empty confidence intervals raises ValueError"""
        with pytest.raises(ValueError, match="confidence_intervals cannot be empty"):
            ForecastResult(
                model_id="model_123",
                keyword="test_keyword",
                forecast_values=self.forecast_values,
                forecast_dates=self.forecast_dates,
                confidence_intervals={},  # Empty
                model_metrics=self.metrics,
                forecast_horizon=25
            )
    
    def test_to_dict(self):
        """Test to_dict method"""
        result = ForecastResult(
            model_id="model_123",
            keyword="test_keyword",
            forecast_values=self.forecast_values,
            forecast_dates=self.forecast_dates,
            confidence_intervals=self.confidence_intervals,
            model_metrics=self.metrics,
            forecast_horizon=25
        )
        
        dict_result = result.to_dict()
        assert dict_result["model_id"] == "model_123"
        assert dict_result["keyword"] == "test_keyword"
        assert len(dict_result["forecast"]["values"]) == 25
        assert len(dict_result["forecast"]["dates"]) == 25
        assert dict_result["model_info"]["forecast_horizon"] == 25


class TestDartsModelValidator:
    """Test DartsModelValidator class"""
    
    def test_validate_model_type_valid(self):
        """Test validating valid model types"""
        assert DartsModelValidator.validate_model_type("lstm") == ModelType.LSTM
        assert DartsModelValidator.validate_model_type("n_beats") == ModelType.N_BEATS
        assert DartsModelValidator.validate_model_type("tft") == ModelType.TFT
    
    def test_validate_model_type_invalid(self):
        """Test validating invalid model types"""
        with pytest.raises(ValueError, match="Invalid model_type"):
            DartsModelValidator.validate_model_type("invalid_model")
    
    def test_validate_time_series_data_valid(self):
        """Test validating valid time series data"""
        data = [50.0, 51.0, 52.0] * 20  # 60 points
        DartsModelValidator.validate_time_series_data(data)
    
    def test_validate_time_series_data_empty(self):
        """Test validating empty time series data"""
        with pytest.raises(ValueError, match="Time series data cannot be empty"):
            DartsModelValidator.validate_time_series_data([])
    
    def test_validate_time_series_data_insufficient(self):
        """Test validating insufficient time series data"""
        data = [50.0, 51.0, 52.0] * 10  # 30 points
        with pytest.raises(ValueError, match="Time series data must have at least 52 data points"):
            DartsModelValidator.validate_time_series_data(data)
    
    def test_validate_time_series_data_non_numeric(self):
        """Test validating non-numeric time series data"""
        data = [50.0, "invalid", 52.0] * 20
        with pytest.raises(ValueError, match="All time series values must be numeric"):
            DartsModelValidator.validate_time_series_data(data)
    
    def test_validate_time_series_data_negative(self):
        """Test validating negative time series data"""
        data = [50.0, -1.0, 52.0] * 20
        with pytest.raises(ValueError, match="Time series values cannot be negative"):
            DartsModelValidator.validate_time_series_data(data)
    
    def test_validate_dates_valid(self):
        """Test validating valid dates"""
        dates = ["2023-01-01", "2023-01-08", "2023-01-15"]
        DartsModelValidator.validate_dates(dates)
    
    def test_validate_dates_empty(self):
        """Test validating empty dates"""
        with pytest.raises(ValueError, match="Dates cannot be empty"):
            DartsModelValidator.validate_dates([])
    
    def test_validate_dates_invalid_format(self):
        """Test validating invalid date format"""
        dates = ["2023-01-01", "invalid-date", "2023-01-15"]
        with pytest.raises(ValueError, match="All dates must be in ISO format"):
            DartsModelValidator.validate_dates(dates)
    
    def test_validate_model_parameters_valid(self):
        """Test validating valid model parameters"""
        params = {"n_epochs": 100, "batch_size": 32}
        DartsModelValidator.validate_model_parameters(ModelType.LSTM, params)
    
    def test_validate_model_parameters_not_dict(self):
        """Test validating non-dict model parameters"""
        with pytest.raises(ValueError, match="model_parameters must be a dictionary"):
            DartsModelValidator.validate_model_parameters(ModelType.LSTM, "invalid")
    
    def test_validate_model_parameters_invalid_epochs(self):
        """Test validating invalid epochs"""
        params = {"n_epochs": 0, "batch_size": 32}
        with pytest.raises(ValueError, match="n_epochs must be positive"):
            DartsModelValidator.validate_model_parameters(ModelType.LSTM, params)
    
    def test_validate_model_parameters_invalid_batch_size(self):
        """Test validating invalid batch size"""
        params = {"n_epochs": 100, "batch_size": -1}
        with pytest.raises(ValueError, match="batch_size must be positive"):
            DartsModelValidator.validate_model_parameters(ModelType.LSTM, params)


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_generate_model_id(self):
        """Test generate_model_id function"""
        model_id = generate_model_id()
        assert model_id.startswith("model_")
        assert len(model_id) == 18  # "model_" + 12 hex chars
    
    def test_default_model_parameters(self):
        """Test DEFAULT_MODEL_PARAMETERS"""
        # Test that all model types have default parameters
        for model_type in ModelType:
            assert model_type in DEFAULT_MODEL_PARAMETERS
            assert isinstance(DEFAULT_MODEL_PARAMETERS[model_type], dict)
        
        # Test specific model parameters
        lstm_params = DEFAULT_MODEL_PARAMETERS[ModelType.LSTM]
        assert "input_chunk_length" in lstm_params
        assert "n_epochs" in lstm_params
        assert "batch_size" in lstm_params
        
        n_beats_params = DEFAULT_MODEL_PARAMETERS[ModelType.N_BEATS]
        assert "num_stacks" in n_beats_params
        assert "layer_widths" in n_beats_params
        
        tft_params = DEFAULT_MODEL_PARAMETERS[ModelType.TFT]
        assert "hidden_size" in tft_params
        assert "num_attention_heads" in tft_params 
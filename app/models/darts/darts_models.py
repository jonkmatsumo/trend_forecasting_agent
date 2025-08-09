"""
Darts Data Models for Google Trends Quantile Forecaster

This module defines the data models and structures used for Darts-based time series forecasting,
including model types, training requests, evaluation metrics, and forecast results.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from darts import TimeSeries


class ModelType(Enum):
    """Supported Darts model types"""
    LSTM = "lstm"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    PROPHET = "prophet"
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    RANDOM_FOREST = "random_forest"
    N_BEATS = "n_beats"
    TFT = "tft"
    GRU = "gru"
    AUTO_ARIMA = "auto_arima"
    AUTO_ETS = "auto_ets"
    AUTO_THETA = "auto_theta"
    AUTO_CES = "auto_ces"


@dataclass
class DartsTimeSeriesData:
    """Enhanced time series data for Darts"""
    keyword: str
    time_series: TimeSeries
    dates: List[datetime]
    values: List[float]
    frequency: str = "W"  # Weekly frequency
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.keyword:
            raise ValueError("Keyword cannot be empty")
        if len(self.dates) != len(self.values):
            raise ValueError("Dates and values must have same length")
        if len(self.values) < 52:
            raise ValueError("Time series must have at least 52 data points (1 year)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "keyword": self.keyword,
            "dates": [d.isoformat() for d in self.dates],
            "values": self.values,
            "frequency": self.frequency,
            "metadata": self.metadata,
            "length": len(self.values)
        }


@dataclass
class ModelTrainingRequest:
    """Request for model training with Darts"""
    keyword: str
    time_series_data: List[float]
    dates: List[str]
    model_type: ModelType
    train_test_split: float = 0.8  # 80% train, 20% test
    forecast_horizon: int = 25  # weeks
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    validation_strategy: str = "holdout"  # holdout, expanding_window, rolling_window
    
    def __post_init__(self):
        if not self.keyword:
            raise ValueError("Keyword cannot be empty")
        if len(self.time_series_data) < 52:
            raise ValueError("Time series data must have at least 52 data points")
        if len(self.dates) != len(self.time_series_data):
            raise ValueError("Dates and time_series_data must have same length")
        if not 0.1 <= self.train_test_split <= 0.9:
            raise ValueError("train_test_split must be between 0.1 and 0.9")
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        if self.validation_strategy not in ["holdout", "expanding_window", "rolling_window"]:
            raise ValueError("validation_strategy must be one of: holdout, expanding_window, rolling_window")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "keyword": self.keyword,
            "time_series_data": self.time_series_data,
            "dates": self.dates,
            "model_type": self.model_type.value,
            "train_test_split": self.train_test_split,
            "forecast_horizon": self.forecast_horizon,
            "model_parameters": self.model_parameters,
            "validation_strategy": self.validation_strategy
        }


@dataclass
class ModelEvaluationMetrics:
    """Comprehensive model evaluation metrics"""
    model_id: str
    keyword: str
    model_type: ModelType
    
    # Training metrics
    train_mae: float
    train_rmse: float
    train_mape: float
    
    # Test metrics (holdout set)
    test_mae: float
    test_rmse: float
    test_mape: float
    
    # Additional metrics
    directional_accuracy: float  # % of correct trend directions
    coverage_95: float  # 95% confidence interval coverage
    
    # Data info
    train_samples: int
    test_samples: int
    total_samples: int
    
    # Training info
    training_time_seconds: float
    mlflow_run_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if not 0 <= self.directional_accuracy <= 1:
            raise ValueError("directional_accuracy must be between 0 and 1")
        if not 0 <= self.coverage_95 <= 1:
            raise ValueError("coverage_95 must be between 0 and 1")
        if self.train_samples <= 0 or self.test_samples <= 0:
            raise ValueError("Sample counts must be positive")
        if self.training_time_seconds < 0:
            raise ValueError("Training time cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "model_id": self.model_id,
            "keyword": self.keyword,
            "model_type": self.model_type.value,
            "test_metrics": {
                "mae": self.test_mae,
                "rmse": self.test_rmse,
                "mape": self.test_mape,
                "directional_accuracy": self.directional_accuracy,
                "coverage_95": self.coverage_95
            },
            "train_metrics": {
                "mae": self.train_mae,
                "rmse": self.train_rmse,
                "mape": self.train_mape
            },
            "data_info": {
                "train_samples": self.train_samples,
                "test_samples": self.test_samples,
                "total_samples": self.total_samples
            },
            "training_info": {
                "training_time_seconds": self.training_time_seconds,
                "mlflow_run_id": self.mlflow_run_id,
                "created_at": self.created_at.isoformat()
            }
        }


@dataclass
class ForecastResult:
    """Forecast results with confidence intervals"""
    model_id: str
    keyword: str
    forecast_values: List[float]
    forecast_dates: List[datetime]
    confidence_intervals: Dict[str, Dict[str, List[float]]]  # e.g., {"95%": {"lower": [...], "upper": [...]}}
    model_metrics: ModelEvaluationMetrics
    forecast_horizon: int
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def __post_init__(self):
        if len(self.forecast_values) != len(self.forecast_dates):
            raise ValueError("forecast_values and forecast_dates must have same length")
        if len(self.forecast_values) != self.forecast_horizon:
            raise ValueError("forecast_values length must match forecast_horizon")
        if not self.confidence_intervals:
            raise ValueError("confidence_intervals cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "model_id": self.model_id,
            "keyword": self.keyword,
            "forecast": {
                "values": self.forecast_values,
                "dates": [d.isoformat() for d in self.forecast_dates],
                "confidence_intervals": self.confidence_intervals
            },
            "model_performance": self.model_metrics.to_dict(),
            "model_info": {
                "model_id": self.model_id,
                "keyword": self.keyword,
                "model_type": self.model_metrics.model_type.value,
                "forecast_horizon": self.forecast_horizon,
                "generated_at": self.generated_at.isoformat()
            }
        }


class DartsModelValidator:
    """Validation utilities for Darts models"""
    
    @staticmethod
    def validate_model_type(model_type: str) -> ModelType:
        """Validate and convert string model type to enum"""
        try:
            return ModelType(model_type)
        except ValueError:
            valid_types = [mt.value for mt in ModelType]
            raise ValueError(f"Invalid model_type: {model_type}. Valid types: {valid_types}")
    
    @staticmethod
    def validate_time_series_data(data: List[float], min_points: int = 52) -> None:
        """Validate time series data"""
        if not data:
            raise ValueError("Time series data cannot be empty")
        if len(data) < min_points:
            raise ValueError(f"Time series data must have at least {min_points} data points")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("All time series values must be numeric")
        if any(x < 0 for x in data):
            raise ValueError("Time series values cannot be negative")
    
    @staticmethod
    def validate_dates(dates: List[str]) -> None:
        """Validate date strings"""
        if not dates:
            raise ValueError("Dates cannot be empty")
        try:
            [datetime.fromisoformat(d) for d in dates]
        except ValueError:
            raise ValueError("All dates must be in ISO format (YYYY-MM-DD)")
    
    @staticmethod
    def validate_model_parameters(model_type: ModelType, parameters: Dict[str, Any]) -> None:
        """Validate model-specific parameters"""
        # Basic validation - specific validation will be done in the service layer
        if not isinstance(parameters, dict):
            raise ValueError("model_parameters must be a dictionary")
        
        # Add model-specific validation here as needed
        if model_type in [ModelType.LSTM, ModelType.GRU, ModelType.TCN, ModelType.TRANSFORMER, ModelType.N_BEATS, ModelType.TFT]:
            if "n_epochs" in parameters and parameters["n_epochs"] <= 0:
                raise ValueError("n_epochs must be positive")
            if "batch_size" in parameters and parameters["batch_size"] <= 0:
                raise ValueError("batch_size must be positive")


def generate_model_id() -> str:
    """Generate a unique model ID"""
    return f"model_{uuid.uuid4().hex[:12]}"


# Default model parameters for each model type
DEFAULT_MODEL_PARAMETERS = {
    ModelType.LSTM: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'n_epochs': 100,
        'batch_size': 32,
        'hidden_size': 50,
        'num_layers': 2,
        'dropout': 0.1
    },
    ModelType.TCN: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'num_filters': 64,
        'kernel_size': 3,
        'num_layers': 3,
        'dropout': 0.1
    },
    ModelType.TRANSFORMER: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1
    },
    ModelType.PROPHET: {
        'changepoint_prior_scale': 0.05,
        'seasonality_prior_scale': 10.0,
        'holidays_prior_scale': 10.0
    },
    ModelType.ARIMA: {
        'p': 1,
        'd': 1,
        'q': 1
    },
    ModelType.EXPONENTIAL_SMOOTHING: {
        'trend': 'add',
        'seasonal': 'add',
        'seasonal_periods': 52
    },
    ModelType.RANDOM_FOREST: {
        'lags': 12,
        'n_estimators': 100,
        'max_depth': 10
    },
    ModelType.N_BEATS: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'generic_architecture': True,
        'num_stacks': 30,
        'num_blocks': 1,
        'num_layers': 4,
        'layer_widths': 256,
        'n_epochs': 100,
        'batch_size': 32
    },
    ModelType.TFT: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'hidden_size': 64,
        'lstm_layers': 2,
        'num_attention_heads': 4,
        'full_attention': False,
        'dropout': 0.1,
        'hidden_continuous_size': 32,
        'categorical_embedding_sizes': {},
        'n_epochs': 100,
        'batch_size': 32
    },
    ModelType.GRU: {
        'input_chunk_length': 12,
        'output_chunk_length': 1,
        'hidden_size': 50,
        'num_layers': 2,
        'dropout': 0.1,
        'n_epochs': 100,
        'batch_size': 32
    },
    ModelType.AUTO_ARIMA: {
        'start_p': 1,
        'start_q': 1,
        'max_p': 5,
        'max_q': 5,
        'max_d': 2,
        'seasonal': True,
        'stepwise': True,
        'approximation': False,
        'method': 'lbfgs'
    },
    ModelType.AUTO_ETS: {
        'seasonal_periods': 52,
        'model': 'ZZZ',
        'damped': None,
        'allow_multiplicative_trend': True,
        'restrict': True,
        'remove_bias': False
    },
    ModelType.AUTO_THETA: {
        'seasonal_periods': 52,
        'seasonal_decomp_model': 'additive'
    },
    ModelType.AUTO_CES: {
        'seasonal_periods': 52,
        'model': 'ZZZ'
    }
} 
"""
Data models for prediction and model metadata
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

# Import Darts model types for compatibility
try:
    from .darts_models import ModelType
except ImportError:
    # Fallback for backward compatibility
    class ModelType:
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
class ModelMetadata:
    """Data model for model metadata (supports both legacy LSTM and Darts models)"""
    
    keyword: str
    training_date: datetime
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    model_id: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    model_path: Optional[str] = None
    scaler_path: Optional[str] = None
    status: str = "completed"
    data_points: int = 0
    created_at: Optional[datetime] = None
    model_type: str = "lstm"  # Default to LSTM for backward compatibility
    darts_model_path: Optional[str] = None  # Path to saved Darts model
    
    def __post_init__(self):
        """Validate and set defaults"""
        if not self.model_id:
            self.model_id = str(uuid.uuid4())
        
        if not self.created_at:
            self.created_at = datetime.utcnow()
        
        if not self.training_date:
            self.training_date = self.created_at
        
        # Validate required fields
        if not self.keyword or not self.keyword.strip():
            raise ValueError("Keyword cannot be empty")
        
        if not isinstance(self.parameters, dict):
            raise ValueError("Parameters must be a dictionary")
        
        if not isinstance(self.metrics, dict):
            raise ValueError("Metrics must be a dictionary")
        
        # Validate status
        valid_statuses = ["pending", "training", "completed", "failed"]
        if self.status not in valid_statuses:
            raise ValueError(f"Status must be one of: {valid_statuses}")
        
        # Validate model type
        valid_model_types = [
            "lstm", "tcn", "transformer", "prophet", "arima", 
            "exponential_smoothing", "random_forest", "n_beats", 
            "tft", "gru", "auto_arima", "auto_ets", "auto_theta", "auto_ces"
        ]
        if self.model_type not in valid_model_types:
            raise ValueError(f"Model type must be one of: {valid_model_types}")
    
    @property
    def is_darts_model(self) -> bool:
        """Check if this is a Darts model (not legacy LSTM)"""
        return self.model_type != "lstm"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'model_id': self.model_id,
            'keyword': self.keyword,
            'training_date': self.training_date.isoformat(),
            'parameters': self.parameters,
            'metrics': self.metrics,
            'mlflow_run_id': self.mlflow_run_id,
            'model_path': self.model_path,
            'scaler_path': self.scaler_path,
            'status': self.status,
            'data_points': self.data_points,
            'created_at': self.created_at.isoformat(),
            'model_type': self.model_type,
            'darts_model_path': self.darts_model_path,
            'is_darts_model': self.is_darts_model
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ModelMetadata':
        """Create ModelMetadata from dictionary"""
        return cls(
            model_id=data['model_id'],
            keyword=data['keyword'],
            training_date=datetime.fromisoformat(data['training_date']),
            parameters=data['parameters'],
            metrics=data['metrics'],
            mlflow_run_id=data.get('mlflow_run_id'),
            model_path=data.get('model_path'),
            scaler_path=data.get('scaler_path'),
            status=data.get('status', 'completed'),
            data_points=data.get('data_points', 0),
            created_at=datetime.fromisoformat(data['created_at']) if data.get('created_at') else None,
            model_type=data.get('model_type', 'lstm'),  # Default to lstm for backward compatibility
            darts_model_path=data.get('darts_model_path')
        )


@dataclass
class PredictionResult:
    """Data model for prediction results (supports both legacy and Darts models)"""
    
    model_id: str
    predictions: List[Dict[str, Any]]
    confidence_intervals: List[Dict[str, float]]
    generated_date: datetime
    prediction_weeks: int
    keyword: str
    status: str = "completed"
    model_type: str = "lstm"  # Default to LSTM for backward compatibility
    forecast_values: Optional[List[float]] = None  # Darts forecast values
    forecast_dates: Optional[List[str]] = None  # Darts forecast dates
    
    def __post_init__(self):
        """Validate and set defaults"""
        if not self.generated_date:
            self.generated_date = datetime.utcnow()
        
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")
        
        if not self.keyword:
            raise ValueError("Keyword cannot be empty")
        
        if self.prediction_weeks <= 0:
            raise ValueError("Prediction weeks must be positive")
        
        # For legacy models, validate predictions match weeks
        if not self.is_darts_model and len(self.predictions) != self.prediction_weeks:
            raise ValueError("Number of predictions must match prediction_weeks")
        
        if not self.is_darts_model and len(self.confidence_intervals) != self.prediction_weeks:
            raise ValueError("Number of confidence intervals must match prediction_weeks")
        
        # For Darts models, validate forecast values match weeks
        if self.is_darts_model and self.forecast_values and len(self.forecast_values) != self.prediction_weeks:
            raise ValueError("Number of forecast values must match prediction_weeks")
    
    @property
    def is_darts_model(self) -> bool:
        """Check if this is a Darts model result"""
        return self.model_type != "lstm"
    
    @property
    def prediction_count(self) -> int:
        """Get the number of predictions"""
        if self.is_darts_model and self.forecast_values:
            return len(self.forecast_values)
        return len(self.predictions)
    
    @property
    def average_prediction(self) -> float:
        """Calculate average prediction value"""
        if self.is_darts_model and self.forecast_values:
            if not self.forecast_values:
                return 0.0
            return sum(self.forecast_values) / len(self.forecast_values)
        
        if not self.predictions:
            return 0.0
        values = [pred.get('value', 0) for pred in self.predictions]
        return sum(values) / len(values)
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        result = {
            'model_id': self.model_id,
            'keyword': self.keyword,
            'generated_date': self.generated_date.isoformat(),
            'prediction_weeks': self.prediction_weeks,
            'prediction_count': self.prediction_count,
            'average_prediction': self.average_prediction,
            'status': self.status,
            'model_type': self.model_type,
            'is_darts_model': self.is_darts_model
        }
        
        # Include appropriate prediction data based on model type
        if self.is_darts_model:
            result.update({
                'forecast_values': self.forecast_values,
                'forecast_dates': self.forecast_dates,
                'confidence_intervals': self.confidence_intervals
            })
        else:
            result.update({
                'predictions': self.predictions,
                'confidence_intervals': self.confidence_intervals
            })
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'PredictionResult':
        """Create PredictionResult from dictionary"""
        return cls(
            model_id=data['model_id'],
            keyword=data['keyword'],
            predictions=data.get('predictions', []),
            confidence_intervals=data.get('confidence_intervals', []),
            generated_date=datetime.fromisoformat(data['generated_date']),
            prediction_weeks=data['prediction_weeks'],
            status=data.get('status', 'completed'),
            model_type=data.get('model_type', 'lstm'),
            forecast_values=data.get('forecast_values'),
            forecast_dates=data.get('forecast_dates')
        )


@dataclass
class TrainingRequest:
    """Data model for model training request (supports both legacy and Darts models)"""
    
    keyword: str
    time_series_data: List[float]
    model_params: Dict[str, Any]
    model_type: str = "lstm"  # Default to LSTM for backward compatibility
    dates: Optional[List[str]] = None  # Required for Darts models
    train_test_split: float = 0.8  # For Darts models
    forecast_horizon: int = 25  # For Darts models
    validation_strategy: str = "holdout"  # For Darts models
    
    def __post_init__(self):
        """Validate training request data"""
        if not self.keyword or not self.keyword.strip():
            raise ValueError("Keyword cannot be empty")
        
        # Validate data points based on model type
        if self.is_darts_model:
            if len(self.time_series_data) < 52:
                raise ValueError("At least 52 data points required for Darts models")
            if not self.dates:
                raise ValueError("Dates are required for Darts models")
            if len(self.dates) != len(self.time_series_data):
                raise ValueError("Dates and time_series_data must have same length")
        else:
            # Legacy LSTM validation
            if len(self.time_series_data) < 10:
                raise ValueError("At least 10 data points required for training")
        
        if len(self.time_series_data) > 10000:
            raise ValueError("Maximum 10,000 data points allowed")
        
        # Validate each data point
        for i, value in enumerate(self.time_series_data):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Data point at index {i} must be a number")
            if value < 0 or value > 100:
                raise ValueError(f"Data point at index {i} must be between 0 and 100")
        
        # Set default parameters if not provided
        if not self.model_params:
            self.model_params = {}
        
        # Validate model parameters based on type
        if self.is_darts_model:
            self._validate_darts_params()
        else:
            self._validate_legacy_params()
    
    @property
    def is_darts_model(self) -> bool:
        """Check if this is a Darts model request"""
        return self.model_type != "lstm"
    
    def _validate_legacy_params(self):
        """Validate legacy LSTM model training parameters"""
        # Validate epochs
        epochs = self.model_params.get('epochs', 150)
        if not isinstance(epochs, int) or epochs < 1 or epochs > 1000:
            raise ValueError("Epochs must be between 1 and 1000")
        
        # Validate batch_size
        batch_size = self.model_params.get('batch_size', 5)
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 100:
            raise ValueError("Batch size must be between 1 and 100")
        
        # Validate lstm_units
        lstm_units = self.model_params.get('lstm_units', 4)
        if not isinstance(lstm_units, int) or lstm_units < 1 or lstm_units > 100:
            raise ValueError("LSTM units must be between 1 and 100")
    
    def _validate_darts_params(self):
        """Validate Darts model parameters"""
        # Validate train_test_split
        if not 0.1 <= self.train_test_split <= 0.9:
            raise ValueError("train_test_split must be between 0.1 and 0.9")
        
        # Validate forecast_horizon
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        
        # Validate validation_strategy
        valid_strategies = ["holdout", "expanding_window", "rolling_window"]
        if self.validation_strategy not in valid_strategies:
            raise ValueError(f"validation_strategy must be one of: {valid_strategies}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        result = {
            'keyword': self.keyword,
            'time_series_data': self.time_series_data,
            'model_params': self.model_params,
            'data_points': len(self.time_series_data),
            'model_type': self.model_type,
            'is_darts_model': self.is_darts_model
        }
        
        # Add Darts-specific fields
        if self.is_darts_model:
            result.update({
                'dates': self.dates,
                'train_test_split': self.train_test_split,
                'forecast_horizon': self.forecast_horizon,
                'validation_strategy': self.validation_strategy
            })
        
        return result


@dataclass
class PredictionRequest:
    """Data model for prediction request (supports both legacy and Darts models)"""
    
    model_id: str
    prediction_weeks: int = 25
    model_type: str = "lstm"  # Default to LSTM for backward compatibility
    
    def __post_init__(self):
        """Validate prediction request data"""
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")
        
        if self.prediction_weeks < 1 or self.prediction_weeks > 100:
            raise ValueError("Prediction weeks must be between 1 and 100")
    
    @property
    def is_darts_model(self) -> bool:
        """Check if this is a Darts model request"""
        return self.model_type != "lstm"
    
    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            'model_id': self.model_id,
            'prediction_weeks': self.prediction_weeks,
            'model_type': self.model_type,
            'is_darts_model': self.is_darts_model
        } 
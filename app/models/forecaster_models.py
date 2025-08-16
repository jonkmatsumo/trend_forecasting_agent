"""
Forecaster Data Models
Defines request and response models that mirror current API endpoint shapes.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class ModelType(str, Enum):
    """Supported model types for training."""
    LSTM = "lstm"
    GRU = "gru"
    TCN = "tcn"
    TRANSFORMER = "transformer"
    N_BEATS = "n_beats"
    TFT = "tft"
    ARIMA = "arima"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    PROPHET = "prophet"
    RANDOM_FOREST = "random_forest"
    AUTO_ARIMA = "auto_arima"
    AUTO_ETS = "auto_ets"
    AUTO_THETA = "auto_theta"
    AUTO_CES = "auto_ces"


class ValidationStrategy(str, Enum):
    """Supported validation strategies."""
    HOLDOUT = "holdout"
    EXPANDING_WINDOW = "expanding_window"
    ROLLING_WINDOW = "rolling_window"


@dataclass
class PredictionRequest:
    """Request model for prediction operations."""
    model_id: str
    forecast_horizon: Optional[int] = 25
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PredictionResponse:
    """Response model for prediction operations - mirrors /models/{model_id}/predict endpoint."""
    status: str
    forecast: Dict[str, Any]
    model_performance: Dict[str, Any]
    model_info: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrendsSummaryRequest:
    """Request model for trends summary operations."""
    keywords: List[str]
    timeframe: str = "today 12-m"
    geo: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrendsSummaryRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrendsSummaryResponse:
    """Response model for trends summary operations - mirrors /trends/summary endpoint."""
    status: str
    summary: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CompareRequest:
    """Request model for compare operations."""
    keywords: List[str]
    timeframe: str = "today 12-m"
    geo: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompareRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CompareResponse:
    """Response model for compare operations - mirrors /trends/compare endpoint."""
    status: str
    comparison: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingRequest:
    """Request model for training operations."""
    keyword: str
    time_series_data: List[float]
    dates: List[str]
    model_type: str
    train_test_split: float = 0.8
    forecast_horizon: int = 25
    model_parameters: Optional[Dict[str, Any]] = None
    validation_strategy: str = "holdout"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        if self.model_parameters is None:
            data['model_parameters'] = {}
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class TrainingResponse:
    """Response model for training operations - mirrors /models/train endpoint."""
    status: str
    model_id: str
    keyword: str
    model_type: str
    evaluation_metrics: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class EvaluationRequest:
    """Request model for evaluation operations."""
    model_id: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationResponse:
    """Response model for evaluation operations - mirrors /models/{model_id}/evaluate endpoint."""
    status: str
    model_id: str
    evaluation_metrics: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ListModelsRequest:
    """Request model for list models operations."""
    keyword: Optional[str] = None
    model_type: Optional[str] = None
    limit: int = 50
    offset: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ListModelsRequest':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ListModelsResponse:
    """Response model for list models operations - mirrors /models endpoint."""
    status: str
    models: List[Dict[str, Any]]
    total_count: int
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class HealthResponse:
    """Response model for health check operations - mirrors /health endpoint."""
    status: str
    service: str
    version: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CacheStatsResponse:
    """Response model for cache stats operations - mirrors /trends/cache/stats endpoint."""
    status: str
    cache_stats: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CacheClearResponse:
    """Response model for cache clear operations - mirrors /trends/cache/clear endpoint."""
    status: str
    message: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Error response models
@dataclass
class ErrorResponse:
    """Standard error response model."""
    status: str
    error_code: str
    message: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# Factory functions for creating responses that match current API shapes
def create_prediction_response(
    forecast: Dict[str, Any],
    model_performance: Dict[str, Any],
    model_info: Dict[str, Any]
) -> PredictionResponse:
    """Create a prediction response matching the current API shape."""
    return PredictionResponse(
        status="success",
        forecast=forecast,
        model_performance=model_performance,
        model_info=model_info,
        timestamp=datetime.utcnow().isoformat()
    )


def create_trends_summary_response(summary: Dict[str, Any]) -> TrendsSummaryResponse:
    """Create a trends summary response matching the current API shape."""
    return TrendsSummaryResponse(
        status="success",
        summary=summary,
        timestamp=datetime.utcnow().isoformat()
    )


def create_compare_response(comparison: Dict[str, Any]) -> CompareResponse:
    """Create a compare response matching the current API shape."""
    return CompareResponse(
        status="success",
        comparison=comparison,
        timestamp=datetime.utcnow().isoformat()
    )


def create_training_response(
    model_id: str,
    keyword: str,
    model_type: str,
    evaluation_metrics: Dict[str, Any]
) -> TrainingResponse:
    """Create a training response matching the current API shape."""
    return TrainingResponse(
        status="success",
        model_id=model_id,
        keyword=keyword,
        model_type=model_type,
        evaluation_metrics=evaluation_metrics,
        timestamp=datetime.utcnow().isoformat()
    )


def create_evaluation_response(
    model_id: str,
    evaluation_metrics: Dict[str, Any]
) -> EvaluationResponse:
    """Create an evaluation response matching the current API shape."""
    return EvaluationResponse(
        status="success",
        model_id=model_id,
        evaluation_metrics=evaluation_metrics,
        timestamp=datetime.utcnow().isoformat()
    )


def create_list_models_response(
    models: List[Dict[str, Any]],
    total_count: int
) -> ListModelsResponse:
    """Create a list models response matching the current API shape."""
    return ListModelsResponse(
        status="success",
        models=models,
        total_count=total_count,
        timestamp=datetime.utcnow().isoformat()
    )


def create_health_response(service: str, version: str) -> HealthResponse:
    """Create a health response matching the current API shape."""
    return HealthResponse(
        status="healthy",
        service=service,
        version=version,
        timestamp=datetime.utcnow().isoformat()
    )


def create_cache_stats_response(cache_stats: Dict[str, Any]) -> CacheStatsResponse:
    """Create a cache stats response matching the current API shape."""
    return CacheStatsResponse(
        status="success",
        cache_stats=cache_stats,
        timestamp=datetime.utcnow().isoformat()
    )


def create_cache_clear_response() -> CacheClearResponse:
    """Create a cache clear response matching the current API shape."""
    return CacheClearResponse(
        status="success",
        message="Trends cache cleared successfully",
        timestamp=datetime.utcnow().isoformat()
    )


def create_error_response(error_code: str, message: str) -> ErrorResponse:
    """Create an error response matching the current API shape."""
    return ErrorResponse(
        status="error",
        error_code=error_code,
        message=message,
        timestamp=datetime.utcnow().isoformat()
    ) 
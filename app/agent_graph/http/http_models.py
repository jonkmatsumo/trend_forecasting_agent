"""
HTTP Request/Response Models
Data models for HTTP client requests and responses.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class HTTPMethod(str, Enum):
    """HTTP methods supported by the client."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


@dataclass
class HTTPRequest:
    """HTTP request model."""
    method: HTTPMethod
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    json: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = None
    
    def __post_init__(self):
        """Validate request parameters."""
        if self.data is not None and self.json is not None:
            raise ValueError("Cannot specify both data and json")


@dataclass
class HTTPResponse:
    """HTTP response model."""
    status_code: int
    headers: Dict[str, str]
    data: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    url: str = ""
    elapsed_time: float = 0.0
    
    @property
    def is_success(self) -> bool:
        """Check if response indicates success."""
        return 200 <= self.status_code < 300
    
    @property
    def is_client_error(self) -> bool:
        """Check if response indicates client error."""
        return 400 <= self.status_code < 500
    
    @property
    def is_server_error(self) -> bool:
        """Check if response indicates server error."""
        return 500 <= self.status_code < 600


@dataclass
class HTTPError(Exception):
    """HTTP error exception."""
    status_code: int
    message: str
    response: Optional[HTTPResponse] = None
    request: Optional[HTTPRequest] = None
    
    def __str__(self) -> str:
        return f"HTTP {self.status_code}: {self.message}"


# Specific request/response models for forecaster endpoints
@dataclass
class HealthRequest(HTTPRequest):
    """Health check request."""
    def __init__(self, base_url: str):
        super().__init__(
            method=HTTPMethod.GET,
            url=f"{base_url}/health"
        )


@dataclass
class TrendsSummaryRequest(HTTPRequest):
    """Trends summary request."""
    def __init__(self, base_url: str, keywords: List[str], timeframe: str = "today 12-m", geo: str = ""):
        super().__init__(
            method=HTTPMethod.POST,
            url=f"{base_url}/api/trends/summary",
            json={
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo
            }
        )


@dataclass
class CompareRequest(HTTPRequest):
    """Compare trends request."""
    def __init__(self, base_url: str, keywords: List[str], timeframe: str = "today 12-m", geo: str = ""):
        super().__init__(
            method=HTTPMethod.POST,
            url=f"{base_url}/api/trends/compare",
            json={
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo
            }
        )


@dataclass
class ListModelsRequest(HTTPRequest):
    """List models request."""
    def __init__(self, base_url: str, keyword: Optional[str] = None, model_type: Optional[str] = None,
                 limit: int = 50, offset: int = 0):
        params = {"limit": limit, "offset": offset}
        if keyword:
            params["keyword"] = keyword
        if model_type:
            params["model_type"] = model_type
            
        super().__init__(
            method=HTTPMethod.GET,
            url=f"{base_url}/api/models",
            params=params
        )


@dataclass
class PredictRequest(HTTPRequest):
    """Predict request."""
    def __init__(self, base_url: str, model_id: str, forecast_horizon: Optional[int] = None):
        json_data = {"model_id": model_id}
        if forecast_horizon:
            json_data["forecast_horizon"] = forecast_horizon
            
        super().__init__(
            method=HTTPMethod.POST,
            url=f"{base_url}/api/models/{model_id}/predict",
            json=json_data
        )


@dataclass
class TrainRequest(HTTPRequest):
    """Train model request."""
    def __init__(self, base_url: str, keyword: str, time_series_data: List[float], dates: List[str],
                 model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
                 model_parameters: Optional[Dict[str, Any]] = None, validation_strategy: str = "holdout"):
        json_data = {
            "keyword": keyword,
            "time_series_data": time_series_data,
            "dates": dates,
            "model_type": model_type,
            "train_test_split": train_test_split,
            "forecast_horizon": forecast_horizon,
            "validation_strategy": validation_strategy
        }
        if model_parameters:
            json_data["model_parameters"] = model_parameters
            
        super().__init__(
            method=HTTPMethod.POST,
            url=f"{base_url}/api/models/train",
            json=json_data
        )


@dataclass
class EvaluateRequest(HTTPRequest):
    """Evaluate model request."""
    def __init__(self, base_url: str, model_id: str):
        super().__init__(
            method=HTTPMethod.GET,
            url=f"{base_url}/api/models/{model_id}/evaluate"
        ) 
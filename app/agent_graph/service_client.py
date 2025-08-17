"""
Service Client for LangGraph Agent
Provides a clean interface for the agent to interact with forecaster services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
from app.services.forecaster_interface import ForecasterServiceInterface


class ForecasterClient(Protocol):
    """Protocol defining the interface for forecaster service clients."""
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        ...
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        ...
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        ...
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        ...
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        ...
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        ...
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        ...


class InProcessForecasterClient:
    """In-process implementation of the forecaster client."""
    
    def __init__(self, forecaster_service: ForecasterServiceInterface):
        """Initialize the in-process client.
        
        Args:
            forecaster_service: The forecaster service interface to use
        """
        self.service = forecaster_service
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        return self.service.health()
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        return self.service.trends_summary(keywords, timeframe, geo)
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        return self.service.compare(keywords, timeframe, geo)
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        return self.service.list_models(keyword, model_type, limit, offset)
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        return self.service.predict(model_id, forecast_horizon)
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        return self.service.train(
            keyword, time_series_data, dates, model_type, train_test_split,
            forecast_horizon, model_parameters, validation_strategy
        )
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        return self.service.evaluate(model_id)


class HTTPForecasterClient:
    """HTTP implementation of the forecaster client (for future use)."""
    
    def __init__(self, base_url: str, timeout: int = 30):
        """Initialize the HTTP client.
        
        Args:
            base_url: Base URL for the forecaster API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        # TODO: Implement HTTP client with proper session management
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        # TODO: Implement HTTP call to /health endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        # TODO: Implement HTTP call to /trends/summary endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        # TODO: Implement HTTP call to /trends/compare endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        # TODO: Implement HTTP call to /models endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        # TODO: Implement HTTP call to /models/{model_id}/predict endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        # TODO: Implement HTTP call to /models/train endpoint
        raise NotImplementedError("HTTP client not yet implemented")
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        # TODO: Implement HTTP call to /models/{model_id}/evaluate endpoint
        raise NotImplementedError("HTTP client not yet implemented") 
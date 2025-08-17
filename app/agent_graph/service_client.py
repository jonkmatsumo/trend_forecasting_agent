"""
Service Client for LangGraph Agent
Provides a clean interface for the agent to interact with forecaster services.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Protocol
import logging

from app.services.forecaster_interface import ForecasterServiceInterface
from app.client.http.http_client import HTTPClient
from app.client.http.http_config import HTTPClientConfig, load_http_config


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
        self.forecaster_service = forecaster_service
        self.logger = logging.getLogger(__name__)
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        try:
            return self.forecaster_service.health()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        try:
            return self.forecaster_service.get_trends_summary(keywords, timeframe, geo)
        except Exception as e:
            self.logger.error(f"Trends summary failed: {e}")
            raise
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        try:
            return self.forecaster_service.compare_keywords(keywords, timeframe, geo)
        except Exception as e:
            self.logger.error(f"Compare failed: {e}")
            raise
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        try:
            return self.forecaster_service.list_models(keyword, model_type, limit, offset)
        except Exception as e:
            self.logger.error(f"List models failed: {e}")
            raise
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        try:
            return self.forecaster_service.predict(model_id, forecast_horizon)
        except Exception as e:
            self.logger.error(f"Predict failed: {e}")
            raise
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        try:
            return self.forecaster_service.train_model(
                keyword, time_series_data, dates, model_type,
                train_test_split, forecast_horizon, model_parameters, validation_strategy
            )
        except Exception as e:
            self.logger.error(f"Train failed: {e}")
            raise
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        try:
            return self.forecaster_service.evaluate_model(model_id)
        except Exception as e:
            self.logger.error(f"Evaluate failed: {e}")
            raise


class HTTPForecasterClient:
    """HTTP implementation of the forecaster client."""
    
    def __init__(self, base_url: str, config: Optional[HTTPClientConfig] = None):
        """Initialize the HTTP client.
        
        Args:
            base_url: Base URL for the forecaster API
            config: HTTP client configuration (optional)
        """
        if config is None:
            config = load_http_config()
            config.base_url = base_url
        
        self.http_client = HTTPClient(config)
        self.logger = logging.getLogger(__name__)
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        try:
            return self.http_client.health()
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        return self.http_client.trends_summary(keywords, timeframe, geo)
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        return self.http_client.compare(keywords, timeframe, geo)
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        return self.http_client.list_models(keyword, model_type, limit, offset)
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        return self.http_client.predict(model_id, forecast_horizon)
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        return self.http_client.train(
            keyword, time_series_data, dates, model_type,
            train_test_split, forecast_horizon, model_parameters, validation_strategy
        )
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        return self.http_client.evaluate(model_id)
    
    def close(self) -> None:
        """Close the HTTP client."""
        self.http_client.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 
"""
Forecaster Service Interface
Defines the contract for all forecaster operations that the agent can call.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Dict, Any, List, Optional, Union
from datetime import datetime


class ForecasterServiceInterface(Protocol):
    """
    Protocol defining the interface for all forecaster operations.
    
    This interface abstracts all current API endpoints, allowing the agent
    to call services without knowing internal module details.
    """
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate prediction using a trained model.
        
        Args:
            model_id: Unique model identifier
            forecast_horizon: Number of periods to forecast (default: 25)
            
        Returns:
            Dictionary containing forecast results with same shape as current /models/{model_id}/predict endpoint
        """
        ...
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """
        Get summary statistics for trends data.
        
        Args:
            keywords: List of keywords to analyze
            timeframe: Time frame for analysis (default: "today 12-m")
            geo: Geographic location (default: "")
            
        Returns:
            Dictionary containing trends summary with same shape as current /trends/summary endpoint
        """
        ...
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """
        Compare trends between multiple keywords.
        
        Args:
            keywords: List of keywords to compare (minimum 2)
            timeframe: Time frame for comparison (default: "today 12-m")
            geo: Geographic location (default: "")
            
        Returns:
            Dictionary containing comparison results with same shape as current /trends/compare endpoint
        """
        ...
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """
        Train a new model with provided time series data.
        
        Args:
            keyword: Target keyword for forecasting
            time_series_data: Interest values (minimum 52 points)
            dates: Date strings in YYYY-MM-DD format
            model_type: Type of model to train
            train_test_split: Train/test split ratio (0.1-0.9, default: 0.8)
            forecast_horizon: Number of periods to forecast (default: 25)
            model_parameters: Model-specific hyperparameters
            validation_strategy: Validation approach (default: "holdout")
            
        Returns:
            Dictionary containing training results with same shape as current /models/train endpoint
        """
        ...
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """
        Retrieve comprehensive evaluation metrics for a trained model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Dictionary containing evaluation metrics with same shape as current /models/{model_id}/evaluate endpoint
        """
        ...
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """
        List all available models with basic information.
        
        Args:
            keyword: Filter by keyword
            model_type: Filter by model type
            limit: Maximum number of models to return (default: 50)
            offset: Number of models to skip (default: 0)
            
        Returns:
            Dictionary containing models list with same shape as current /models endpoint
        """
        ...
    
    def health(self) -> Dict[str, Any]:
        """
        Health check for the forecaster service.
        
        Returns:
            Dictionary containing health status with same shape as current /health endpoint
        """
        ...
    
    def cache_stats(self) -> Dict[str, Any]:
        """
        Get trends service cache statistics.
        
        Returns:
            Dictionary containing cache statistics with same shape as current /trends/cache/stats endpoint
        """
        ...
    
    def cache_clear(self) -> Dict[str, Any]:
        """
        Clear the trends service cache.
        
        Returns:
            Dictionary containing cache clear result with same shape as current /trends/cache/clear endpoint
        """
        ...


class AbstractForecasterService(ABC):
    """
    Abstract base class implementing the Forecaster Service Interface.
    
    Provides default implementations and common functionality for all forecaster services.
    """
    
    @abstractmethod
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        pass
    
    @abstractmethod
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        pass
    
    @abstractmethod
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        pass
    
    @abstractmethod
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        pass
    
    @abstractmethod
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        pass
    
    @abstractmethod
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        pass
    
    @abstractmethod
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        pass
    
    @abstractmethod
    def cache_stats(self) -> Dict[str, Any]:
        """Get trends service cache statistics."""
        pass
    
    @abstractmethod
    def cache_clear(self) -> Dict[str, Any]:
        """Clear the trends service cache."""
        pass
    
    def _validate_model_id(self, model_id: str) -> str:
        """
        Validate model ID format.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            Validated model ID
            
        Raises:
            ValueError: If model ID is invalid
        """
        if not model_id or not isinstance(model_id, str):
            raise ValueError("Model ID must be a non-empty string")
        return model_id.strip()
    
    def _validate_keywords(self, keywords: List[str]) -> List[str]:
        """
        Validate keywords list.
        
        Args:
            keywords: Keywords to validate
            
        Returns:
            Validated keywords list
            
        Raises:
            ValueError: If keywords are invalid
        """
        if not keywords or not isinstance(keywords, list):
            raise ValueError("Keywords must be a non-empty list")
        
        validated_keywords = []
        for keyword in keywords:
            if not keyword or not isinstance(keyword, str):
                raise ValueError("Each keyword must be a non-empty string")
            validated_keywords.append(keyword.strip())
        
        return validated_keywords
    
    def _validate_time_series_data(self, time_series_data: List[float], dates: List[str]) -> tuple:
        """
        Validate time series data and dates.
        
        Args:
            time_series_data: Time series values
            dates: Corresponding dates
            
        Returns:
            Tuple of validated time series data and dates
            
        Raises:
            ValueError: If data is invalid
        """
        if not time_series_data or not isinstance(time_series_data, list):
            raise ValueError("Time series data must be a non-empty list")
        
        if not dates or not isinstance(dates, list):
            raise ValueError("Dates must be a non-empty list")
        
        if len(time_series_data) != len(dates):
            raise ValueError("Time series data and dates must have the same length")
        
        if len(time_series_data) < 52:
            raise ValueError("Time series data must have at least 52 data points")
        
        # Validate that all values are numeric
        for i, value in enumerate(time_series_data):
            if not isinstance(value, (int, float)):
                raise ValueError(f"Time series data value at index {i} must be numeric")
        
        # Validate date format (basic check)
        for i, date_str in enumerate(dates):
            if not isinstance(date_str, str):
                raise ValueError(f"Date at index {i} must be a string")
            # Basic YYYY-MM-DD format check
            if len(date_str) != 10 or date_str[4] != '-' or date_str[7] != '-':
                raise ValueError(f"Date at index {i} must be in YYYY-MM-DD format")
        
        return time_series_data, dates 
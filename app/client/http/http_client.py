"""
HTTP Client Implementation
Robust HTTP client with session management, retry logic, and error handling.
"""

import time
import random
import logging
from typing import Dict, Any, Optional, Union
from urllib.parse import urljoin
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .http_config import HTTPClientConfig
from .http_models import (
    HTTPRequest, HTTPResponse, HTTPError, HTTPMethod,
    HealthRequest, TrendsSummaryRequest, CompareRequest,
    ListModelsRequest, PredictRequest, TrainRequest, EvaluateRequest
)


class HTTPClient:
    """Robust HTTP client with session management and retry logic."""
    
    def __init__(self, config: HTTPClientConfig):
        """Initialize HTTP client with configuration.
        
        Args:
            config: HTTP client configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create and configure requests session.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=self.config.pool_connections,
            pool_maxsize=self.config.pool_maxsize,
            max_retries=self._create_retry_strategy()
        )
        
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update(self.config.default_headers)
        
        return session
    
    def _create_retry_strategy(self) -> Retry:
        """Create retry strategy with exponential backoff.
        
        Returns:
            Configured retry strategy
        """
        return Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
            respect_retry_after_header=True
        )
    
    def _log_request(self, request: HTTPRequest) -> None:
        """Log HTTP request details.
        
        Args:
            request: HTTP request to log
        """
        if not self.config.enable_request_logging:
            return
        
        log_data = {
            "method": request.method.value,
            "url": request.url,
            "headers": {k: v for k, v in request.headers.items() if k.lower() != "authorization"},
            "params": request.params
        }
        
        if self.config.log_request_body:
            if request.json:
                log_data["json"] = request.json
            elif request.data:
                log_data["data"] = request.data
        
        self.logger.debug(f"HTTP Request: {log_data}")
    
    def _log_response(self, response: HTTPResponse, elapsed_time: float) -> None:
        """Log HTTP response details.
        
        Args:
            response: HTTP response to log
            elapsed_time: Request elapsed time
        """
        if not self.config.enable_request_logging:
            return
        
        log_data = {
            "status_code": response.status_code,
            "url": response.url,
            "elapsed_time": elapsed_time,
            "headers": response.headers
        }
        
        if self.config.log_response_body:
            if response.data:
                log_data["data"] = response.data
            elif response.text:
                log_data["text"] = response.text[:1000]  # Limit text length
        
        self.logger.debug(f"HTTP Response: {log_data}")
    
    def _make_request(self, request: HTTPRequest) -> HTTPResponse:
        """Make HTTP request with retry logic and error handling.
        
        Args:
            request: HTTP request to make
            
        Returns:
            HTTP response
            
        Raises:
            HTTPError: If request fails after retries
        """
        start_time = time.time()
        
        try:
            # Log request
            self._log_request(request)
            
            # Prepare request parameters
            kwargs = {
                "timeout": request.timeout or self.config.timeout,
                "headers": request.headers,
                "params": request.params
            }
            
            if request.json is not None:
                kwargs["json"] = request.json
            elif request.data is not None:
                kwargs["data"] = request.data
            
            # Make request
            response = self.session.request(
                method=request.method.value,
                url=request.url,
                **kwargs
            )
            
            elapsed_time = time.time() - start_time
            
            # Create response model
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                url=response.url,
                elapsed_time=elapsed_time
            )
            
            # Parse response content
            try:
                if response.headers.get("content-type", "").startswith("application/json"):
                    http_response.data = response.json()
                else:
                    http_response.text = response.text
            except Exception as e:
                self.logger.warning(f"Failed to parse response content: {e}")
                http_response.text = response.text
            
            # Log response
            self._log_response(http_response, elapsed_time)
            
            # Check for errors
            if not http_response.is_success:
                raise HTTPError(
                    status_code=http_response.status_code,
                    message=f"HTTP {http_response.status_code}: {response.text}",
                    response=http_response,
                    request=request
                )
            
            return http_response
            
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            self.logger.error(f"Request failed: {e}")
            raise HTTPError(
                status_code=0,
                message=f"Request failed: {str(e)}",
                request=request
            )
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service.
        
        Returns:
            Health check response data
            
        Raises:
            HTTPError: If health check fails
        """
        request = HealthRequest(self.config.base_url)
        response = self._make_request(request)
        return response.data or {}
    
    def trends_summary(self, keywords: list, timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data.
        
        Args:
            keywords: List of keywords to analyze
            timeframe: Time frame for analysis
            geo: Geographic location filter
            
        Returns:
            Trends summary data
            
        Raises:
            HTTPError: If request fails
        """
        request = TrendsSummaryRequest(self.config.base_url, keywords, timeframe, geo)
        response = self._make_request(request)
        return response.data or {}
    
    def compare(self, keywords: list, timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords.
        
        Args:
            keywords: List of keywords to compare
            timeframe: Time frame for comparison
            geo: Geographic location filter
            
        Returns:
            Comparison data
            
        Raises:
            HTTPError: If request fails
        """
        request = CompareRequest(self.config.base_url, keywords, timeframe, geo)
        response = self._make_request(request)
        return response.data or {}
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information.
        
        Args:
            keyword: Filter by keyword
            model_type: Filter by model type
            limit: Maximum number of models to return
            offset: Number of models to skip
            
        Returns:
            List of models data
            
        Raises:
            HTTPError: If request fails
        """
        request = ListModelsRequest(self.config.base_url, keyword, model_type, limit, offset)
        response = self._make_request(request)
        return response.data or {}
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate predictions using a trained model.
        
        Args:
            model_id: Unique model identifier
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Prediction data
            
        Raises:
            HTTPError: If request fails
        """
        request = PredictRequest(self.config.base_url, model_id, forecast_horizon)
        response = self._make_request(request)
        return response.data or {}
    
    def train(self, keyword: str, time_series_data: list, dates: list, model_type: str,
              train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data.
        
        Args:
            keyword: Keyword for the model
            time_series_data: Time series values
            dates: Corresponding dates
            model_type: Type of model to train
            train_test_split: Training/test split ratio
            forecast_horizon: Forecast horizon
            model_parameters: Model-specific parameters
            validation_strategy: Validation strategy
            
        Returns:
            Training results data
            
        Raises:
            HTTPError: If request fails
        """
        request = TrainRequest(
            self.config.base_url, keyword, time_series_data, dates, model_type,
            train_test_split, forecast_horizon, model_parameters, validation_strategy
        )
        response = self._make_request(request)
        return response.data or {}
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Evaluation metrics data
            
        Raises:
            HTTPError: If request fails
        """
        request = EvaluateRequest(self.config.base_url, model_id)
        response = self._make_request(request)
        return response.data or {}
    
    def close(self) -> None:
        """Close the HTTP session."""
        if self.session:
            self.session.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close() 
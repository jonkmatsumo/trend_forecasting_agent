"""
HTTP Adapter for Forecaster Service Interface
Makes HTTP calls to forecaster service with retry logic and timeout handling.
"""

import logging
import time
import uuid
import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urljoin

from app.services.forecaster_interface import AbstractForecasterService
from app.models.forecaster_models import (
    create_prediction_response, create_trends_summary_response, create_compare_response,
    create_training_response, create_evaluation_response, create_list_models_response,
    create_health_response, create_cache_stats_response, create_cache_clear_response,
    create_error_response
)
from app.utils.error_handlers import (
    ValidationError, ModelError, TrendsAPIError, RateLimitError, NotFoundError
)


class HTTPAdapter(AbstractForecasterService):
    """
    HTTP Adapter that implements the Forecaster Service Interface.
    
    This adapter makes HTTP calls to a forecaster service, providing
    the option to run the forecaster as a separate microservice.
    """
    
    def __init__(self, base_url: str, timeout: int = 30, max_retries: int = 3):
        """Initialize the HTTP adapter.
        
        Args:
            base_url: Base URL of the forecaster service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.logger = logging.getLogger(__name__)
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        # Request tracking
        self.request_id = None
    
    def _generate_request_id(self) -> str:
        """Generate a unique request ID for tracking."""
        return str(uuid.uuid4())
    
    def _log_request_start(self, operation: str, **kwargs):
        """Log the start of a request operation."""
        self.request_id = self._generate_request_id()
        self.logger.info(f"Request {self.request_id}: Starting {operation}", extra={
            'request_id': self.request_id,
            'operation': operation,
            'parameters': kwargs,
            'adapter_type': 'http'
        })
    
    def _log_request_end(self, operation: str, success: bool, duration: float, **kwargs):
        """Log the end of a request operation."""
        self.logger.info(f"Request {self.request_id}: Completed {operation}", extra={
            'request_id': self.request_id,
            'operation': operation,
            'success': success,
            'duration_ms': round(duration * 1000, 2),
            'result': kwargs,
            'adapter_type': 'http'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                     params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with retry logic and exponential backoff.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            
        Returns:
            Response data as dictionary
            
        Raises:
            Various exceptions based on HTTP status codes
        """
        url = urljoin(self.base_url, endpoint)
        
        # Prepare request
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-Request-ID': self.request_id
        }
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=self.timeout
                )
                
                duration = time.time() - start_time
                
                # Log the request attempt
                self.logger.debug(f"HTTP request attempt {attempt + 1}/{self.max_retries + 1}", extra={
                    'request_id': self.request_id,
                    'method': method,
                    'url': url,
                    'status_code': response.status_code,
                    'duration_ms': round(duration * 1000, 2),
                    'attempt': attempt + 1
                })
                
                # Handle successful response
                if response.status_code < 400:
                    try:
                        return response.json()
                    except json.JSONDecodeError as e:
                        raise ModelError(f"Invalid JSON response: {str(e)}")
                
                # Handle specific HTTP status codes
                if response.status_code == 400:
                    raise ValidationError(f"Bad request: {response.text}")
                elif response.status_code == 404:
                    raise NotFoundError(f"Resource not found: {response.text}")
                elif response.status_code == 429:
                    raise RateLimitError(f"Rate limit exceeded: {response.text}")
                elif response.status_code >= 500:
                    # Server error - retry if we have attempts left
                    if attempt < self.max_retries:
                        wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff + jitter
                        self.logger.warning(f"Server error, retrying in {wait_time:.2f}s", extra={
                            'request_id': self.request_id,
                            'status_code': response.status_code,
                            'attempt': attempt + 1,
                            'wait_time': wait_time
                        })
                        time.sleep(wait_time)
                        last_exception = ModelError(f"Server error: {response.text}")
                        continue
                    else:
                        raise ModelError(f"Server error after {self.max_retries} retries: {response.text}")
                else:
                    # Other client errors
                    raise ModelError(f"HTTP {response.status_code}: {response.text}")
                    
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                # Network errors - retry if we have attempts left
                if attempt < self.max_retries:
                    wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff + jitter
                    self.logger.warning(f"Network error, retrying in {wait_time:.2f}s", extra={
                        'request_id': self.request_id,
                        'error': str(e),
                        'attempt': attempt + 1,
                        'wait_time': wait_time
                    })
                    time.sleep(wait_time)
                    last_exception = e
                    continue
                else:
                    raise ModelError(f"Network error after {self.max_retries} retries: {str(e)}")
            
            except Exception as e:
                # Other exceptions - don't retry
                raise e
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise ModelError("Request failed after all retry attempts")
    
    def _map_exception_to_error_response(self, e: Exception) -> Dict[str, Any]:
        """Map internal exceptions to error responses with appropriate HTTP status semantics."""
        if isinstance(e, ValidationError):
            return create_error_response("VALIDATION_ERROR", str(e)).to_dict()
        elif isinstance(e, NotFoundError):
            return create_error_response("NOT_FOUND", str(e)).to_dict()
        elif isinstance(e, RateLimitError):
            return create_error_response("RATE_LIMIT_ERROR", str(e)).to_dict()
        elif isinstance(e, ModelError):
            return create_error_response("MODEL_ERROR", str(e)).to_dict()
        else:
            return create_error_response("INTERNAL_ERROR", f"Unexpected error: {str(e)}").to_dict()
    
    def predict(self, model_id: str, forecast_horizon: Optional[int] = None) -> Dict[str, Any]:
        """Generate prediction using a trained model."""
        start_time = time.time()
        operation = "predict"
        
        try:
            self._log_request_start(operation, model_id=model_id, forecast_horizon=forecast_horizon)
            
            # Validate inputs
            model_id = self._validate_model_id(model_id)
            forecast_horizon = forecast_horizon or 25
            
            # Prepare request data
            data = {
                "model_id": model_id,
                "forecast_horizon": forecast_horizon
            }
            
            # Make HTTP request
            response_data = self._make_request("POST", f"/api/models/{model_id}/predict", data=data)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def trends_summary(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Get summary statistics for trends data."""
        start_time = time.time()
        operation = "trends_summary"
        
        try:
            self._log_request_start(operation, keywords=keywords, timeframe=timeframe, geo=geo)
            
            # Validate inputs
            keywords = self._validate_keywords(keywords)
            
            # Prepare request data
            data = {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo
            }
            
            # Make HTTP request
            response_data = self._make_request("POST", "/trends/summary", data=data)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def compare(self, keywords: List[str], timeframe: str = "today 12-m", geo: str = "") -> Dict[str, Any]:
        """Compare trends between multiple keywords."""
        start_time = time.time()
        operation = "compare"
        
        try:
            self._log_request_start(operation, keywords=keywords, timeframe=timeframe, geo=geo)
            
            # Validate inputs
            keywords = self._validate_keywords(keywords)
            if len(keywords) < 2:
                raise ValidationError("At least 2 keywords are required for comparison")
            
            # Prepare request data
            data = {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo
            }
            
            # Make HTTP request
            response_data = self._make_request("POST", "/trends/compare", data=data)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def train(self, keyword: str, time_series_data: List[float], dates: List[str], 
              model_type: str, train_test_split: float = 0.8, forecast_horizon: int = 25,
              model_parameters: Optional[Dict[str, Any]] = None,
              validation_strategy: str = "holdout") -> Dict[str, Any]:
        """Train a new model with provided time series data."""
        start_time = time.time()
        operation = "train"
        
        try:
            self._log_request_start(operation, keyword=keyword, model_type=model_type, 
                                  train_test_split=train_test_split, forecast_horizon=forecast_horizon)
            
            # Validate inputs
            time_series_data, dates = self._validate_time_series_data(time_series_data, dates)
            
            # Prepare request data
            data = {
                "keyword": keyword,
                "time_series_data": time_series_data,
                "dates": dates,
                "model_type": model_type,
                "train_test_split": train_test_split,
                "forecast_horizon": forecast_horizon,
                "model_parameters": model_parameters or {},
                "validation_strategy": validation_strategy
            }
            
            # Make HTTP request
            response_data = self._make_request("POST", "/api/models/train", data=data)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def evaluate(self, model_id: str) -> Dict[str, Any]:
        """Retrieve comprehensive evaluation metrics for a trained model."""
        start_time = time.time()
        operation = "evaluate"
        
        try:
            self._log_request_start(operation, model_id=model_id)
            
            # Validate inputs
            model_id = self._validate_model_id(model_id)
            
            # Make HTTP request
            response_data = self._make_request("GET", f"/api/models/{model_id}/evaluate")
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def list_models(self, keyword: Optional[str] = None, model_type: Optional[str] = None,
                   limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """List all available models with basic information."""
        start_time = time.time()
        operation = "list_models"
        
        try:
            self._log_request_start(operation, keyword=keyword, model_type=model_type, 
                                  limit=limit, offset=offset)
            
            # Prepare query parameters
            params = {
                "limit": limit,
                "offset": offset
            }
            if keyword:
                params["keyword"] = keyword
            if model_type:
                params["model_type"] = model_type
            
            # Make HTTP request
            response_data = self._make_request("GET", "/api/models", params=params)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def health(self) -> Dict[str, Any]:
        """Health check for the forecaster service."""
        start_time = time.time()
        operation = "health"
        
        try:
            self._log_request_start(operation)
            
            # Make HTTP request
            response_data = self._make_request("GET", "/health")
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get trends service cache statistics."""
        start_time = time.time()
        operation = "cache_stats"
        
        try:
            self._log_request_start(operation)
            
            # Make HTTP request
            response_data = self._make_request("GET", "/trends/cache/stats")
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e)
    
    def cache_clear(self) -> Dict[str, Any]:
        """Clear the trends service cache."""
        start_time = time.time()
        operation = "cache_clear"
        
        try:
            self._log_request_start(operation)
            
            # Make HTTP request
            response_data = self._make_request("POST", "/trends/cache/clear")
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return response_data
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e) 
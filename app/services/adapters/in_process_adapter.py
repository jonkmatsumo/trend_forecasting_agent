"""
In-Process Adapter for Forecaster Service Interface
Directly calls existing Python services without HTTP overhead.
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.forecaster_interface import AbstractForecasterService
from app.services.pytrends.trends_service import TrendsService
from app.services.darts.training_service import TrainingService
from app.services.darts.prediction_service import PredictionService
from app.services.darts.evaluation_service import EvaluationService
from app.models.forecaster_models import (
    create_prediction_response, create_trends_summary_response, create_compare_response,
    create_training_response, create_evaluation_response, create_list_models_response,
    create_health_response, create_cache_stats_response, create_cache_clear_response,
    create_error_response
)
from app.utils.error_handlers import (
    ValidationError, ModelError, TrendsAPIError, RateLimitError, NotFoundError
)


class InProcessAdapter(AbstractForecasterService):
    """
    In-Process Adapter that implements the Forecaster Service Interface.
    
    This adapter directly calls existing Python services without HTTP overhead,
    making it ideal for development and co-located deployments.
    """
    
    def __init__(self):
        """Initialize the in-process adapter with all required services."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.trends_service = TrendsService()
        self.training_service = TrainingService()
        self.prediction_service = PredictionService(self.training_service)
        self.evaluation_service = EvaluationService(self.training_service)
        
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
            'parameters': kwargs
        })
    
    def _log_request_end(self, operation: str, success: bool, duration: float, **kwargs):
        """Log the end of a request operation."""
        self.logger.info(f"Request {self.request_id}: Completed {operation}", extra={
            'request_id': self.request_id,
            'operation': operation,
            'success': success,
            'duration_ms': round(duration * 1000, 2),
            'result': kwargs
        })
    
    def _map_exception_to_error_response(self, e: Exception) -> Dict[str, Any]:
        """Map internal exceptions to error responses with appropriate HTTP status semantics."""
        if isinstance(e, ValidationError):
            return create_error_response("VALIDATION_ERROR", str(e)).to_dict()
        elif isinstance(e, NotFoundError):
            return create_error_response("NOT_FOUND", str(e)).to_dict()
        elif isinstance(e, RateLimitError):
            return create_error_response("RATE_LIMIT_ERROR", str(e)).to_dict()
        elif isinstance(e, (ModelError, TrendsAPIError)):
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
            
            # Generate forecast
            forecast_result = self.prediction_service.generate_forecast(
                model_id=model_id,
                forecast_horizon=forecast_horizon,
                include_confidence_intervals=True
            )
            
            # Get model metadata and evaluation metrics
            metadata = self.training_service.get_model_metadata(model_id)
            evaluation_metrics = self.training_service.get_evaluation_metrics(model_id)
            
            # Build response data
            forecast_data = {
                "values": forecast_result.forecast_values,
                "dates": [d.isoformat() for d in forecast_result.forecast_dates],
                "confidence_intervals": forecast_result.confidence_intervals
            }
            
            model_performance = {
                "test_metrics": evaluation_metrics.to_dict(),
                "train_metrics": evaluation_metrics.to_dict(),  # Simplified for now
                "data_info": {
                    "train_samples": metadata.get("train_samples", 0),
                    "test_samples": metadata.get("test_samples", 0),
                    "total_samples": metadata.get("total_samples", 0)
                }
            }
            
            model_info = {
                "model_id": model_id,
                "keyword": metadata.get("keyword", ""),
                "model_type": metadata.get("model_type", ""),
                "forecast_horizon": forecast_horizon,
                "generated_at": datetime.utcnow().isoformat()
            }
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_prediction_response(forecast_data, model_performance, model_info).to_dict()
            
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
            
            # Create trends request
            from app.models.pytrends.pytrend_model import TrendsRequest
            trends_request = TrendsRequest(
                keywords=keywords,
                timeframe=timeframe,
                geo=geo
            )
            
            # Fetch trends data
            trends_response = self.trends_service.fetch_trends_data(trends_request)
            
            # Build summary data
            summary_data = {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo,
                "statistics": {
                    "total_keywords": len(keywords),
                    "data_points": len(trends_response.data) if trends_response.data else 0,
                    "timeframe": timeframe,
                    "geo": geo
                }
            }
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_trends_summary_response(summary_data).to_dict()
            
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
            
            # Create trends request
            from app.models.pytrends.pytrend_model import TrendsRequest
            trends_request = TrendsRequest(
                keywords=keywords,
                timeframe=timeframe,
                geo=geo
            )
            
            # Fetch trends data
            trends_response = self.trends_service.fetch_trends_data(trends_request)
            
            # Build comparison data
            comparison_data = {
                "keywords": keywords,
                "timeframe": timeframe,
                "geo": geo,
                "comparison_data": {
                    "total_keywords": len(keywords),
                    "data_available": len(trends_response.data) if trends_response.data else 0,
                    "timeframe": timeframe,
                    "geo": geo
                }
            }
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_compare_response(comparison_data).to_dict()
            
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
            
            # Create training request
            from app.models.darts.darts_models import ModelTrainingRequest
            training_request = ModelTrainingRequest(
                keyword=keyword,
                time_series_data=time_series_data,
                dates=dates,
                model_type=model_type,
                train_test_split=train_test_split,
                forecast_horizon=forecast_horizon,
                model_parameters=model_parameters or {},
                validation_strategy=validation_strategy
            )
            
            # Train model
            model_id, evaluation_metrics = self.training_service.train_model(training_request)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration, model_id=model_id)
            
            return create_training_response(
                model_id=model_id,
                keyword=keyword,
                model_type=model_type,
                evaluation_metrics=evaluation_metrics.to_dict()
            ).to_dict()
            
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
            
            # Get evaluation metrics
            evaluation_metrics = self.training_service.get_evaluation_metrics(model_id)
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_evaluation_response(
                model_id=model_id,
                evaluation_metrics=evaluation_metrics.to_dict()
            ).to_dict()
            
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
            
            # Get all model IDs
            model_ids = self.training_service.list_models()
            
            # Filter models
            filtered_models = []
            for model_id in model_ids:
                try:
                    metadata = self.training_service.get_model_metadata(model_id)
                    
                    # Apply filters
                    if keyword and metadata.get("keyword") != keyword:
                        continue
                    if model_type and metadata.get("model_type") != model_type:
                        continue
                    
                    # Get evaluation metrics
                    evaluation_metrics = self.training_service.get_evaluation_metrics(model_id)
                    
                    model_info = {
                        "model_id": model_id,
                        "keyword": metadata.get("keyword", ""),
                        "model_type": metadata.get("model_type", ""),
                        "created_at": metadata.get("training_date", ""),
                        "test_mae": evaluation_metrics.test_mae,
                        "test_rmse": evaluation_metrics.test_rmse,
                        "directional_accuracy": evaluation_metrics.directional_accuracy
                    }
                    
                    filtered_models.append(model_info)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get metadata for model {model_id}: {str(e)}")
                    continue
            
            # Apply pagination
            total_count = len(filtered_models)
            paginated_models = filtered_models[offset:offset + limit]
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration, total_count=total_count)
            
            return create_list_models_response(
                models=paginated_models,
                total_count=total_count
            ).to_dict()
            
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
            
            # Simple health check - verify services are initialized
            health_status = "healthy"
            
            # Check if services are properly initialized
            if not all([self.trends_service, self.training_service, 
                       self.prediction_service, self.evaluation_service]):
                health_status = "degraded"
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_health_response(
                service="Google Trends Quantile Forecaster API",
                version="v1"
            ).to_dict()
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return create_health_response(
                service="Google Trends Quantile Forecaster API",
                version="v1"
            ).to_dict()
    
    def cache_stats(self) -> Dict[str, Any]:
        """Get trends service cache statistics."""
        start_time = time.time()
        operation = "cache_stats"
        
        try:
            self._log_request_start(operation)
            
            # Get cache statistics from trends service
            cache_stats = {
                "cache_size": len(self.trends_service._cache),
                "cache_ttl": self.trends_service._cache_ttl,
                "rate_limit_counter": self.trends_service.rate_limit_counter,
                "max_requests_per_minute": self.trends_service.max_requests_per_minute
            }
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_cache_stats_response(cache_stats).to_dict()
            
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
            
            # Clear the cache
            self.trends_service._cache.clear()
            
            duration = time.time() - start_time
            self._log_request_end(operation, True, duration)
            
            return create_cache_clear_response().to_dict()
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_request_end(operation, False, duration, error=str(e))
            return self._map_exception_to_error_response(e) 
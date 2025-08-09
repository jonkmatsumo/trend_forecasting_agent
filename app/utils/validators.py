"""
Enhanced input validation system for API requests
"""

import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from app.utils.error_handlers import ValidationError, APIError
from app.models.pytrends.pytrend_model import TrendsRequest, TrendData
from app.models.prediction_model import TrainingRequest, PredictionRequest, ModelMetadata


class InputValidator:
    """Comprehensive input validation class"""
    
    @staticmethod
    def validate_keywords(keywords: List[str]) -> List[str]:
        """Validate and clean keywords list"""
        if not keywords:
            raise ValidationError("Keywords list cannot be empty", field="keywords")
        
        if len(keywords) > 5:
            raise ValidationError("Maximum 5 keywords allowed per request", field="keywords")
        
        validated_keywords = []
        for i, keyword in enumerate(keywords):
            if not keyword or not keyword.strip():
                raise ValidationError(f"Keyword at index {i} cannot be empty", field=f"keywords[{i}]")
            
            if len(keyword.strip()) > 100:
                raise ValidationError(f"Keyword at index {i} is too long (max 100 characters)", field=f"keywords[{i}]")
            
            # Basic sanitization - remove extra whitespace
            clean_keyword = keyword.strip()
            validated_keywords.append(clean_keyword)
        
        return validated_keywords
    
    @staticmethod
    def validate_timeframe(timeframe: str) -> str:
        """Validate timeframe parameter"""
        valid_timeframes = [
            "now 1-H", "now 4-H", "now 1-d", "now 7-d",
            "today 1-m", "today 3-m", "today 12-m", "today 5-y",
            "all"
        ]
        
        if timeframe not in valid_timeframes:
            raise ValidationError(
                f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}", 
                field="timeframe"
            )
        
        return timeframe
    
    @staticmethod
    def validate_geo(geo: str) -> str:
        """Validate geographic location parameter"""
        if not isinstance(geo, str):
            raise ValidationError("Geo must be a string", field="geo")
        
        if len(geo) > 10:
            raise ValidationError("Geo code must be 10 characters or less", field="geo")
        
        return geo.strip()
    
    @staticmethod
    def validate_training_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model training parameters"""
        if not isinstance(params, dict):
            raise ValidationError("Model parameters must be a dictionary", field="model_params")
        
        validated_params = {}
        
        # Validate epochs
        epochs = params.get('epochs', 150)
        if not isinstance(epochs, int) or epochs < 1 or epochs > 1000:
            raise ValidationError("Epochs must be between 1 and 1000", field="model_params.epochs")
        validated_params['epochs'] = epochs
        
        # Validate batch_size
        batch_size = params.get('batch_size', 5)
        if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 100:
            raise ValidationError("Batch size must be between 1 and 100", field="model_params.batch_size")
        validated_params['batch_size'] = batch_size
        
        # Validate lstm_units
        lstm_units = params.get('lstm_units', 4)
        if not isinstance(lstm_units, int) or lstm_units < 1 or lstm_units > 100:
            raise ValidationError("LSTM units must be between 1 and 100", field="model_params.lstm_units")
        validated_params['lstm_units'] = lstm_units
        
        # Validate optimizer
        optimizer = params.get('optimizer', 'adam')
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
        if optimizer.lower() not in valid_optimizers:
            raise ValidationError(f"Optimizer must be one of: {', '.join(valid_optimizers)}", field="model_params.optimizer")
        validated_params['optimizer'] = optimizer.lower()
        
        # Validate loss function
        loss = params.get('loss', 'mean_squared_error')
        valid_losses = ['mean_squared_error', 'mean_absolute_error', 'huber_loss']
        if loss not in valid_losses:
            raise ValidationError(f"Loss function must be one of: {', '.join(valid_losses)}", field="model_params.loss")
        validated_params['loss'] = loss
        
        return validated_params
    
    @staticmethod
    def validate_time_series_data(data: List[float]) -> List[float]:
        """Validate time series data"""
        if not isinstance(data, list):
            raise ValidationError("Time series data must be a list", field="time_series_data")
        
        if len(data) < 10:
            raise ValidationError("At least 10 data points required for training", field="time_series_data")
        
        if len(data) > 10000:
            raise ValidationError("Maximum 10,000 data points allowed", field="time_series_data")
        
        validated_data = []
        for i, value in enumerate(data):
            if not isinstance(value, (int, float)):
                raise ValidationError(f"Data point at index {i} must be a number", field=f"time_series_data[{i}]")
            
            if value < 0 or value > 100:
                raise ValidationError(f"Data point at index {i} must be between 0 and 100", field=f"time_series_data[{i}]")
            
            validated_data.append(float(value))
        
        return validated_data
    
    @staticmethod
    def validate_prediction_weeks(weeks: int) -> int:
        """Validate prediction weeks parameter"""
        if not isinstance(weeks, int):
            raise ValidationError("Prediction weeks must be an integer", field="prediction_weeks")
        
        if weeks < 1 or weeks > 100:
            raise ValidationError("Prediction weeks must be between 1 and 100", field="prediction_weeks")
        
        return weeks
    
    @staticmethod
    def validate_model_id(model_id: str) -> str:
        """Validate model ID format"""
        if not model_id or not isinstance(model_id, str):
            raise ValidationError("Model ID is required and must be a string", field="model_id")
        
        # Validate UUID-like format (basic validation)
        uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
        if not uuid_pattern.match(model_id):
            raise ValidationError("Model ID must be a valid UUID format", field="model_id")
        
        return model_id
    
    @staticmethod
    def validate_trends_request(data: Dict[str, Any]) -> TrendsRequest:
        """Validate and create TrendsRequest object"""
        if not isinstance(data, dict):
            raise ValidationError("Request data must be a JSON object")
        
        # Extract and validate keywords
        keywords = data.get('keywords', [])
        if isinstance(keywords, str):
            keywords = [keywords]  # Convert single keyword to list
        
        validated_keywords = InputValidator.validate_keywords(keywords)
        
        # Extract and validate other parameters
        timeframe = InputValidator.validate_timeframe(data.get('timeframe', 'today 12-m'))
        geo = InputValidator.validate_geo(data.get('geo', ''))
        
        try:
            return TrendsRequest(
                keywords=validated_keywords,
                timeframe=timeframe,
                geo=geo
            )
        except ValueError as e:
            raise ValidationError(str(e))
    
    @staticmethod
    def validate_training_request(data: Dict[str, Any]) -> TrainingRequest:
        """Validate and create TrainingRequest object"""
        if not isinstance(data, dict):
            raise ValidationError("Request data must be a JSON object")
        
        # Extract and validate keyword
        keyword = data.get('keyword', '').strip()
        if not keyword:
            raise ValidationError("Keyword is required", field="keyword")
        
        # Extract and validate time series data
        time_series_data = data.get('time_series_data', [])
        validated_data = InputValidator.validate_time_series_data(time_series_data)
        
        # Extract and validate model parameters
        model_params = data.get('model_params', {})
        validated_params = InputValidator.validate_training_parameters(model_params)
        
        try:
            return TrainingRequest(
                keyword=keyword,
                time_series_data=validated_data,
                model_params=validated_params
            )
        except ValueError as e:
            raise ValidationError(str(e))
    
    @staticmethod
    def validate_prediction_request(data: Dict[str, Any], model_id: str) -> PredictionRequest:
        """Validate and create PredictionRequest object"""
        if not isinstance(data, dict):
            data = {}  # Allow empty dict for prediction requests
        
        # Validate model ID
        validated_model_id = InputValidator.validate_model_id(model_id)
        
        # Extract and validate prediction weeks
        prediction_weeks = data.get('prediction_weeks', 25)
        validated_weeks = InputValidator.validate_prediction_weeks(prediction_weeks)
        
        try:
            return PredictionRequest(
                model_id=validated_model_id,
                prediction_weeks=validated_weeks
            )
        except ValueError as e:
            raise ValidationError(str(e))
    
    @staticmethod
    def validate_trend_data(data: Dict[str, Any]) -> TrendData:
        """Validate and create TrendData object"""
        try:
            return TrendData.from_dict(data)
        except (KeyError, ValueError) as e:
            raise ValidationError(f"Invalid trend data format: {str(e)}")
    
    @staticmethod
    def validate_model_metadata(data: Dict[str, Any]) -> ModelMetadata:
        """Validate and create ModelMetadata object"""
        try:
            return ModelMetadata.from_dict(data)
        except (KeyError, ValueError) as e:
            raise ValidationError(f"Invalid model metadata format: {str(e)}")


# Legacy validation functions for backward compatibility
def validate_trends_request(data):
    """Legacy validation function for trends request"""
    try:
        trends_request = InputValidator.validate_trends_request(data)
        return {'valid': True, 'message': 'Validation successful', 'data': trends_request}
    except ValidationError as e:
        return {'valid': False, 'message': e.message}


def validate_training_request(data):
    """Legacy validation function for training request"""
    try:
        training_request = InputValidator.validate_training_request(data)
        return {'valid': True, 'message': 'Validation successful', 'data': training_request}
    except ValidationError as e:
        return {'valid': False, 'message': e.message}


def validate_prediction_request(data):
    """Legacy validation function for prediction request"""
    try:
        # Note: This function doesn't have model_id, so we'll validate just the data structure
        if not isinstance(data, dict):
            return {'valid': False, 'message': 'Request data must be a JSON object'}
        
        prediction_weeks = data.get('prediction_weeks')
        if prediction_weeks is not None:
            try:
                InputValidator.validate_prediction_weeks(prediction_weeks)
            except ValidationError as e:
                return {'valid': False, 'message': e.message}
        
        return {'valid': True, 'message': 'Validation successful'}
    except Exception as e:
        return {'valid': False, 'message': f'Validation error: {str(e)}'}


def validate_model_parameters(params):
    """Legacy validation function for model parameters"""
    try:
        validated_params = InputValidator.validate_training_parameters(params)
        return {'valid': True, 'message': 'Validation successful', 'data': validated_params}
    except ValidationError as e:
        return {'valid': False, 'message': e.message}


def validate_model_id(model_id):
    """Legacy validation function for model ID"""
    try:
        validated_id = InputValidator.validate_model_id(model_id)
        return {'valid': True, 'message': 'Validation successful', 'data': validated_id}
    except ValidationError as e:
        return {'valid': False, 'message': e.message} 
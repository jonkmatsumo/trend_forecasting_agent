"""
Validation utilities for API requests
Input validation for trends, training, and prediction endpoints
"""

import re
from datetime import datetime, timedelta


def validate_trends_request(data):
    """
    Validate trends request data
    
    Args:
        data (dict): Request data containing keyword, timeframe, geo
        
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    if not isinstance(data, dict):
        return {'valid': False, 'message': 'Request data must be a JSON object'}
    
    # Validate keyword
    keyword = data.get('keyword')
    if not keyword:
        return {'valid': False, 'message': 'Keyword is required'}
    
    if not isinstance(keyword, str):
        return {'valid': False, 'message': 'Keyword must be a string'}
    
    if len(keyword.strip()) == 0:
        return {'valid': False, 'message': 'Keyword cannot be empty'}
    
    if len(keyword) > 100:
        return {'valid': False, 'message': 'Keyword must be 100 characters or less'}
    
    # Validate timeframe (optional)
    timeframe = data.get('timeframe')
    if timeframe is not None:
        if not isinstance(timeframe, str):
            return {'valid': False, 'message': 'Timeframe must be a string'}
        
        # Validate timeframe format (basic validation)
        valid_timeframes = [
            'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d',
            'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y',
            'all'
        ]
        if timeframe not in valid_timeframes:
            return {'valid': False, 'message': f'Invalid timeframe. Must be one of: {", ".join(valid_timeframes)}'}
    
    # Validate geo (optional)
    geo = data.get('geo')
    if geo is not None:
        if not isinstance(geo, str):
            return {'valid': False, 'message': 'Geo must be a string'}
        
        if len(geo) > 10:
            return {'valid': False, 'message': 'Geo code must be 10 characters or less'}
    
    return {'valid': True, 'message': 'Validation successful'}


def validate_training_request(data):
    """
    Validate model training request data
    
    Args:
        data (dict): Request data containing time_series_data, keyword, model_params
        
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    if not isinstance(data, dict):
        return {'valid': False, 'message': 'Request data must be a JSON object'}
    
    # Validate keyword
    keyword = data.get('keyword')
    if not keyword:
        return {'valid': False, 'message': 'Keyword is required'}
    
    if not isinstance(keyword, str):
        return {'valid': False, 'message': 'Keyword must be a string'}
    
    if len(keyword.strip()) == 0:
        return {'valid': False, 'message': 'Keyword cannot be empty'}
    
    # Validate time series data
    time_series_data = data.get('time_series_data')
    if not time_series_data:
        return {'valid': False, 'message': 'Time series data is required'}
    
    if not isinstance(time_series_data, list):
        return {'valid': False, 'message': 'Time series data must be a list'}
    
    if len(time_series_data) < 10:
        return {'valid': False, 'message': 'Time series data must contain at least 10 data points'}
    
    if len(time_series_data) > 10000:
        return {'valid': False, 'message': 'Time series data cannot exceed 10,000 data points'}
    
    # Validate each data point
    for i, point in enumerate(time_series_data):
        if not isinstance(point, (int, float)):
            return {'valid': False, 'message': f'Data point {i} must be a number'}
        
        if point < 0 or point > 100:
            return {'valid': False, 'message': f'Data point {i} must be between 0 and 100'}
    
    # Validate model parameters (optional)
    model_params = data.get('model_params', {})
    if model_params:
        if not isinstance(model_params, dict):
            return {'valid': False, 'message': 'Model parameters must be a JSON object'}
        
        # Validate specific model parameters
        validation_result = validate_model_parameters(model_params)
        if not validation_result['valid']:
            return validation_result
    
    return {'valid': True, 'message': 'Validation successful'}


def validate_prediction_request(data):
    """
    Validate prediction request data
    
    Args:
        data (dict): Request data containing prediction_weeks
        
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    if not isinstance(data, dict):
        return {'valid': False, 'message': 'Request data must be a JSON object'}
    
    # Validate prediction_weeks (optional)
    prediction_weeks = data.get('prediction_weeks')
    if prediction_weeks is not None:
        if not isinstance(prediction_weeks, int):
            return {'valid': False, 'message': 'Prediction weeks must be an integer'}
        
        if prediction_weeks < 1:
            return {'valid': False, 'message': 'Prediction weeks must be at least 1'}
        
        if prediction_weeks > 100:
            return {'valid': False, 'message': 'Prediction weeks cannot exceed 100'}
    
    return {'valid': True, 'message': 'Validation successful'}


def validate_model_parameters(params):
    """
    Validate model training parameters
    
    Args:
        params (dict): Model parameters dictionary
        
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    # Validate batch_size
    batch_size = params.get('batch_size')
    if batch_size is not None:
        if not isinstance(batch_size, int):
            return {'valid': False, 'message': 'Batch size must be an integer'}
        
        if batch_size < 1 or batch_size > 100:
            return {'valid': False, 'message': 'Batch size must be between 1 and 100'}
    
    # Validate epochs
    epochs = params.get('epochs')
    if epochs is not None:
        if not isinstance(epochs, int):
            return {'valid': False, 'message': 'Epochs must be an integer'}
        
        if epochs < 1 or epochs > 1000:
            return {'valid': False, 'message': 'Epochs must be between 1 and 1000'}
    
    # Validate lstm_units
    lstm_units = params.get('lstm_units')
    if lstm_units is not None:
        if not isinstance(lstm_units, int):
            return {'valid': False, 'message': 'LSTM units must be an integer'}
        
        if lstm_units < 1 or lstm_units > 100:
            return {'valid': False, 'message': 'LSTM units must be between 1 and 100'}
    
    # Validate optimizer
    optimizer = params.get('optimizer')
    if optimizer is not None:
        if not isinstance(optimizer, str):
            return {'valid': False, 'message': 'Optimizer must be a string'}
        
        valid_optimizers = ['adam', 'sgd', 'rmsprop', 'adagrad']
        if optimizer.lower() not in valid_optimizers:
            return {'valid': False, 'message': f'Optimizer must be one of: {", ".join(valid_optimizers)}'}
    
    # Validate loss
    loss = params.get('loss')
    if loss is not None:
        if not isinstance(loss, str):
            return {'valid': False, 'message': 'Loss function must be a string'}
        
        valid_losses = ['mean_squared_error', 'mean_absolute_error', 'huber_loss']
        if loss not in valid_losses:
            return {'valid': False, 'message': f'Loss function must be one of: {", ".join(valid_losses)}'}
    
    return {'valid': True, 'message': 'Validation successful'}


def validate_model_id(model_id):
    """
    Validate model ID format
    
    Args:
        model_id (str): Model identifier
        
    Returns:
        dict: Validation result with 'valid' boolean and 'message' string
    """
    if not model_id:
        return {'valid': False, 'message': 'Model ID is required'}
    
    if not isinstance(model_id, str):
        return {'valid': False, 'message': 'Model ID must be a string'}
    
    # Validate UUID-like format (basic validation)
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if not uuid_pattern.match(model_id):
        return {'valid': False, 'message': 'Model ID must be a valid UUID format'}
    
    return {'valid': True, 'message': 'Validation successful'} 
"""
Error handling utilities for the API
"""

import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from flask import jsonify, current_app


class APIError(Exception):
    """Custom API exception class"""
    
    def __init__(self, message: str, status_code: int = 400, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or 'API_ERROR'
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ValidationError(APIError):
    """Validation error exception"""
    
    def __init__(self, message: str, field: str = None, details: Dict[str, Any] = None):
        super().__init__(message, status_code=400, error_code='VALIDATION_ERROR', details=details)
        if field:
            self.details['field'] = field


class TrendsAPIError(APIError):
    """Google Trends API error exception"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=500, error_code='TRENDS_API_ERROR', details=details)


class ModelError(APIError):
    """Model training/prediction error exception"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message, status_code=500, error_code='MODEL_ERROR', details=details)


class NotFoundError(APIError):
    """Resource not found error exception"""
    
    def __init__(self, message: str, resource_type: str = None, resource_id: str = None):
        details = {}
        if resource_type:
            details['resource_type'] = resource_type
        if resource_id:
            details['resource_id'] = resource_id
        
        super().__init__(message, status_code=404, error_code='NOT_FOUND', details=details)


class RateLimitError(APIError):
    """Rate limit exceeded error exception"""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        details = {}
        if retry_after:
            details['retry_after'] = retry_after
        
        super().__init__(message, status_code=429, error_code='RATE_LIMIT_EXCEEDED', details=details)


def handle_api_error(error: APIError):
    """Handle custom API errors"""
    response = {
        'status': 'error',
        'error_code': error.error_code,
        'message': error.message,
        'timestamp': error.timestamp.isoformat()
    }
    
    if error.details:
        response['details'] = error.details
    
    # Log the error
    current_app.logger.error(f"API Error: {error.error_code} - {error.message}")
    if error.details:
        current_app.logger.error(f"Error details: {error.details}")
    
    return jsonify(response), error.status_code


def handle_validation_error(error: ValidationError):
    """Handle validation errors"""
    response = {
        'status': 'error',
        'error_code': 'VALIDATION_ERROR',
        'message': error.message,
        'timestamp': error.timestamp.isoformat()
    }
    
    if error.details:
        response['details'] = error.details
    
    # Log validation errors at info level (not error)
    current_app.logger.info(f"Validation Error: {error.message}")
    if error.details:
        current_app.logger.info(f"Validation details: {error.details}")
    
    return jsonify(response), 400


def handle_not_found_error(error: NotFoundError):
    """Handle not found errors"""
    response = {
        'status': 'error',
        'error_code': 'NOT_FOUND',
        'message': error.message,
        'timestamp': error.timestamp.isoformat()
    }
    
    if error.details:
        response['details'] = error.details
    
    # Log not found errors at info level
    current_app.logger.info(f"Not Found: {error.message}")
    
    return jsonify(response), 404


def handle_rate_limit_error(error: RateLimitError):
    """Handle rate limit errors"""
    response = {
        'status': 'error',
        'error_code': 'RATE_LIMIT_EXCEEDED',
        'message': error.message,
        'timestamp': error.timestamp.isoformat()
    }
    
    if error.details:
        response['details'] = error.details
    
    # Log rate limit errors at warning level
    current_app.logger.warning(f"Rate Limit Exceeded: {error.message}")
    
    return jsonify(response), 429


def handle_generic_error(error: Exception):
    """Handle generic/unexpected errors"""
    # Log unexpected errors
    current_app.logger.error(f'Unexpected error: {str(error)}')
    current_app.logger.error(traceback.format_exc())
    
    response = {
        'status': 'error',
        'error_code': 'INTERNAL_ERROR',
        'message': 'An unexpected error occurred',
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # In development mode, include more details
    if current_app.config.get('DEBUG', False):
        response['details'] = {
            'error_type': type(error).__name__,
            'error_message': str(error)
        }
    
    return jsonify(response), 500


def handle_http_error(error):
    """Handle HTTP exceptions"""
    response = {
        'status': 'error',
        'error_code': 'HTTP_ERROR',
        'message': error.description,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Log HTTP errors
    current_app.logger.warning(f"HTTP Error {error.code}: {error.description}")
    
    return jsonify(response), error.code


def register_error_handlers(app):
    """Register all error handlers with the Flask app"""
    
    # Register custom API error handlers
    app.register_error_handler(APIError, handle_api_error)
    app.register_error_handler(ValidationError, handle_validation_error)
    app.register_error_handler(NotFoundError, handle_not_found_error)
    app.register_error_handler(RateLimitError, handle_rate_limit_error)
    
    # Register HTTP error handlers
    from werkzeug.exceptions import HTTPException
    app.register_error_handler(HTTPException, handle_http_error)
    
    # Register generic error handler (must be last)
    app.register_error_handler(Exception, handle_generic_error)


def create_error_response(message: str, error_code: str = 'API_ERROR', 
                         status_code: int = 400, details: Dict[str, Any] = None) -> tuple:
    """Helper function to create standardized error responses"""
    response = {
        'status': 'error',
        'error_code': error_code,
        'message': message,
        'timestamp': datetime.utcnow().isoformat()
    }
    
    if details:
        response['details'] = details
    
    return jsonify(response), status_code 
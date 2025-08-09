"""
Unit tests for error handling system
"""

import pytest
from datetime import datetime
from flask import Flask, jsonify
from app.utils.error_handlers import (
    APIError, ValidationError, TrendsAPIError, ModelError, 
    NotFoundError, RateLimitError, handle_api_error, 
    handle_validation_error, handle_not_found_error,
    handle_rate_limit_error, handle_generic_error,
    handle_http_error, register_error_handlers,
    create_error_response
)
from werkzeug.exceptions import HTTPException, NotFound


class TestAPIError:
    """Test APIError exception class"""
    
    def test_api_error_creation(self):
        """Test creating APIError with basic parameters"""
        error = APIError("Test error message")
        
        assert error.message == "Test error message"
        assert error.status_code == 400
        assert error.error_code == "API_ERROR"
        assert isinstance(error.timestamp, datetime)
    
    def test_api_error_with_custom_params(self):
        """Test creating APIError with custom parameters"""
        error = APIError(
            message="Custom error",
            status_code=500,
            error_code="CUSTOM_ERROR",
            details={"field": "test"}
        )
        
        assert error.message == "Custom error"
        assert error.status_code == 500
        assert error.error_code == "CUSTOM_ERROR"
        assert error.details == {"field": "test"}


class TestValidationError:
    """Test ValidationError exception class"""
    
    def test_validation_error_creation(self):
        """Test creating ValidationError"""
        error = ValidationError("Validation failed", field="test_field")
        
        assert error.message == "Validation failed"
        assert error.status_code == 400
        assert error.error_code == "VALIDATION_ERROR"
        assert error.details["field"] == "test_field"
    
    def test_validation_error_without_field(self):
        """Test creating ValidationError without field"""
        error = ValidationError("Validation failed")
        
        assert error.message == "Validation failed"
        assert error.status_code == 400
        assert error.error_code == "VALIDATION_ERROR"
        assert "field" not in error.details


class TestTrendsAPIError:
    """Test TrendsAPIError exception class"""
    
    def test_trends_api_error_creation(self):
        """Test creating TrendsAPIError"""
        error = TrendsAPIError("Trends API failed", details={"retry": True})
        
        assert error.message == "Trends API failed"
        assert error.status_code == 500
        assert error.error_code == "TRENDS_API_ERROR"
        assert error.details == {"retry": True}


class TestModelError:
    """Test ModelError exception class"""
    
    def test_model_error_creation(self):
        """Test creating ModelError"""
        error = ModelError("Model training failed", details={"epoch": 10})
        
        assert error.message == "Model training failed"
        assert error.status_code == 500
        assert error.error_code == "MODEL_ERROR"
        assert error.details == {"epoch": 10}


class TestNotFoundError:
    """Test NotFoundError exception class"""
    
    def test_not_found_error_creation(self):
        """Test creating NotFoundError"""
        error = NotFoundError("Resource not found", "model", "test-id")
        
        assert error.message == "Resource not found"
        assert error.status_code == 404
        assert error.error_code == "NOT_FOUND"
        assert error.details["resource_type"] == "model"
        assert error.details["resource_id"] == "test-id"


class TestRateLimitError:
    """Test RateLimitError exception class"""
    
    def test_rate_limit_error_creation(self):
        """Test creating RateLimitError"""
        error = RateLimitError("Rate limit exceeded", retry_after=60)
        
        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.details["retry_after"] == 60
    
    def test_rate_limit_error_default_message(self):
        """Test creating RateLimitError with default message"""
        error = RateLimitError()
        
        assert error.message == "Rate limit exceeded"
        assert error.status_code == 429
        assert error.error_code == "RATE_LIMIT_EXCEEDED"


class TestErrorHandlers:
    """Test error handler functions"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    def test_handle_api_error(self, app):
        """Test handle_api_error function"""
        with app.app_context():
            error = APIError("Test error", 400, "TEST_ERROR", {"detail": "test"})
            response, status_code = handle_api_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'TEST_ERROR'
            assert data['message'] == 'Test error'
            assert data['details'] == {"detail": "test"}
            assert status_code == 400
    
    def test_handle_validation_error(self, app):
        """Test handle_validation_error function"""
        with app.app_context():
            error = ValidationError("Validation failed", "test_field", {"value": "invalid"})
            response, status_code = handle_validation_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'VALIDATION_ERROR'
            assert data['message'] == 'Validation failed'
            assert data['details'] == {"field": "test_field", "value": "invalid"}
            assert status_code == 400
    
    def test_handle_not_found_error(self, app):
        """Test handle_not_found_error function"""
        with app.app_context():
            error = NotFoundError("Resource not found", "model", "test-id")
            response, status_code = handle_not_found_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'NOT_FOUND'
            assert data['message'] == 'Resource not found'
            assert status_code == 404
    
    def test_handle_rate_limit_error(self, app):
        """Test handle_rate_limit_error function"""
        with app.app_context():
            error = RateLimitError("Rate limit exceeded", 60)
            response, status_code = handle_rate_limit_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'RATE_LIMIT_EXCEEDED'
            assert data['message'] == 'Rate limit exceeded'
            assert data['details']['retry_after'] == 60
            assert status_code == 429
    
    def test_handle_generic_error(self, app):
        """Test handle_generic_error function"""
        with app.app_context():
            error = Exception("Unexpected error")
            response, status_code = handle_generic_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'INTERNAL_ERROR'
            assert data['message'] == 'An unexpected error occurred'
            assert status_code == 500
    
    def test_handle_generic_error_debug_mode(self, app):
        """Test handle_generic_error function in debug mode"""
        app.config['DEBUG'] = True
        with app.app_context():
            error = Exception("Unexpected error")
            response, status_code = handle_generic_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'INTERNAL_ERROR'
            assert data['message'] == 'An unexpected error occurred'
            assert 'details' in data
            assert data['details']['error_type'] == 'Exception'
            assert data['details']['error_message'] == 'Unexpected error'
            assert status_code == 500
    
    def test_handle_http_error(self, app):
        """Test handle_http_error function"""
        with app.app_context():
            error = NotFound()
            response, status_code = handle_http_error(error)
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'HTTP_ERROR'
            # Update to match the actual Flask error message
            assert 'not found' in data['message'].lower()
            assert status_code == 404


class TestErrorHandlerRegistration:
    """Test error handler registration"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    def test_error_handler_integration(self, app):
        """Test error handler integration with Flask app"""
        register_error_handlers(app)
        
        @app.route('/test-api-error')
        def test_api_error():
            raise APIError("Test API error", 400, "TEST_ERROR")
        
        @app.route('/test-validation-error')
        def test_validation_error():
            raise ValidationError("Test validation error", "test_field")
        
        @app.route('/test-not-found-error')
        def test_not_found_error():
            raise NotFoundError("Test not found error", "resource", "test-id")
        
        @app.route('/test-rate-limit-error')
        def test_rate_limit_error():
            raise RateLimitError("Test rate limit error", 60)
        
        @app.route('/test-generic-error')
        def test_generic_error():
            raise Exception("Test generic error")
        
        with app.test_client() as client:
            # Test API error
            response = client.get('/test-api-error')
            assert response.status_code == 400
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'TEST_ERROR'
            
            # Test validation error
            response = client.get('/test-validation-error')
            assert response.status_code == 400
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'VALIDATION_ERROR'
            
            # Test not found error
            response = client.get('/test-not-found-error')
            assert response.status_code == 404
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'NOT_FOUND'
            
            # Test rate limit error
            response = client.get('/test-rate-limit-error')
            assert response.status_code == 429
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'RATE_LIMIT_EXCEEDED'
            
            # Test generic error
            response = client.get('/test-generic-error')
            assert response.status_code == 500
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'INTERNAL_ERROR'


class TestErrorResponseHelper:
    """Test create_error_response helper function"""
    
    @pytest.fixture
    def app(self):
        """Create test Flask app"""
        app = Flask(__name__)
        app.config['TESTING'] = True
        return app
    
    def test_create_error_response_basic(self, app):
        """Test creating basic error response"""
        with app.app_context():
            response, status_code = create_error_response("Test error")
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'API_ERROR'
            assert data['message'] == 'Test error'
            assert status_code == 400
    
    def test_create_error_response_with_details(self, app):
        """Test creating error response with details"""
        with app.app_context():
            details = {"field": "test", "value": "invalid"}
            response, status_code = create_error_response(
                "Test error", "CUSTOM_ERROR", 422, details
            )
            
            data = response.get_json()
            assert data['status'] == 'error'
            assert data['error_code'] == 'CUSTOM_ERROR'
            assert data['message'] == 'Test error'
            assert data['details'] == details
            assert status_code == 422 
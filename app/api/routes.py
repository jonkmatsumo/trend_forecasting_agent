"""
API Routes for Google Trends Quantile Forecaster
Main API endpoints and route definitions
"""

from flask import Blueprint, request, jsonify, current_app
from flask_limiter.util import get_remote_address
from datetime import datetime
import logging

from app.utils.validators import InputValidator
from app.utils.error_handlers import (
    ValidationError, TrendsAPIError, ModelError, NotFoundError, 
    RateLimitError, APIError
)
from app.models.trend_model import TrendsRequest, TrendsResponse
from app.models.prediction_model import TrainingRequest, PredictionRequest

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize validator
validator = InputValidator()

# Get logger
logger = logging.getLogger(__name__)


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for API
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Google Trends Quantile Forecaster API',
        'version': current_app.config.get('API_VERSION', 'v1'),
        'timestamp': datetime.utcnow().isoformat()
    })


@api_bp.route('/trends', methods=['POST'])
def get_trends():
    """
    Get Google Trends data for keywords
    """
    try:
        # Check content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Content-Type must be application/json',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Validate request
        try:
            data = request.get_json()
        except Exception:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Invalid JSON in request body',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        if not data:
            raise ValidationError("Request body is required")
        
        # Validate and create request object
        trends_request = validator.validate_trends_request(data)
        
        # Initialize service lazily
        from app.services.trends_service import TrendsService
        trends_service = TrendsService()
        
        # Fetch trends data
        trends_response = trends_service.fetch_trends_data(trends_request)
        
        # Return response
        return jsonify(trends_response.to_dict()), 200
        
    except ValidationError as e:
        logger.info(f"Validation error in trends endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except TrendsAPIError as e:
        logger.error(f"Trends API error: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'TRENDS_API_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except RateLimitError as e:
        logger.warning(f"Rate limit exceeded: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'RATE_LIMIT_EXCEEDED',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 429
        
    except Exception as e:
        logger.error(f"Unexpected error in trends endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/models/train', methods=['POST'])
def train_model():
    """
    Train a new LSTM model with provided time series data
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            raise ValidationError("Request body is required")
        
        # Validate and create request object
        training_request = validator.validate_training_request(data)
        
        # Initialize service lazily
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        # Train model
        model_info = model_service.train_model(
            training_request.time_series_data, 
            training_request.keyword, 
            training_request.model_params
        )
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'model_id': model_info['model_id'],
            'keyword': training_request.keyword,
            'training_metrics': model_info['metrics']
        }), 201
        
    except ValidationError as e:
        logger.info(f"Validation error in train model endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except ModelError as e:
        logger.error(f"Model training error: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'MODEL_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in train model endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/models/<model_id>/predict', methods=['POST'])
def generate_prediction(model_id):
    """
    Generate prediction using a trained model
    """
    try:
        # Validate request
        data = request.get_json() or {}
        
        # Validate and create request object
        prediction_request = validator.validate_prediction_request(data, model_id)
        
        # Initialize service lazily
        from app.services.prediction_service import PredictionService
        prediction_service = PredictionService()
        
        # Generate prediction
        prediction_result = prediction_service.generate_prediction(
            prediction_request.model_id, 
            prediction_request.prediction_weeks
        )
        
        return jsonify({
            'status': 'success',
            'model_id': prediction_request.model_id,
            'prediction_weeks': prediction_request.prediction_weeks,
            'predictions': prediction_result['predictions'],
            'confidence_intervals': prediction_result.get('confidence_intervals', [])
        }), 200
        
    except ValidationError as e:
        logger.info(f"Validation error in prediction endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except NotFoundError as e:
        logger.info(f"Model not found: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'NOT_FOUND',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 404
        
    except ModelError as e:
        logger.error(f"Prediction error: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'MODEL_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """
    Get information about a specific model
    """
    try:
        # Validate model ID
        validated_model_id = validator.validate_model_id(model_id)
        
        # Initialize service lazily
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        # Get model info
        model_info = model_service.get_model_info(validated_model_id)
        
        if not model_info:
            raise NotFoundError(f"Model with ID {validated_model_id} not found", 
                              "model", validated_model_id)
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        }), 200
        
    except ValidationError as e:
        logger.info(f"Validation error in get model info endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except NotFoundError as e:
        logger.info(f"Model not found: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'NOT_FOUND',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 404
        
    except Exception as e:
        logger.error(f"Unexpected error in get model info endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/models', methods=['GET'])
def list_models():
    """
    List all available models
    """
    try:
        # Initialize service lazily
        from app.services.model_service import ModelService
        model_service = ModelService()
        
        # Get models list
        models = model_service.list_models()
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        }), 200
        
    except Exception as e:
        logger.error(f"Unexpected error in list models endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/trends/cache/clear', methods=['POST'])
def clear_trends_cache():
    """
    Clear the trends service cache
    """
    try:
        # Initialize service lazily
        from app.services.trends_service import TrendsService
        trends_service = TrendsService()
        
        # Clear cache
        trends_service.clear_cache()
        
        return jsonify({
            'status': 'success',
            'message': 'Trends cache cleared successfully',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error clearing trends cache: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'Failed to clear cache',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/trends/cache/stats', methods=['GET'])
def get_trends_cache_stats():
    """
    Get trends service cache statistics
    """
    try:
        # Initialize service lazily
        from app.services.trends_service import TrendsService
        trends_service = TrendsService()
        
        # Get cache stats
        stats = trends_service.get_cache_stats()
        
        return jsonify({
            'status': 'success',
            'cache_stats': stats,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'Failed to get cache statistics',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/trends/summary', methods=['POST'])
def get_trends_summary():
    """
    Get summary statistics for trends data
    """
    try:
        # Check content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Content-Type must be application/json',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Validate request
        try:
            data = request.get_json()
        except Exception:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Invalid JSON in request body',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        if not data:
            raise ValidationError("Request body is required")
        
        # Extract parameters
        keywords = data.get('keywords', [])
        timeframe = data.get('timeframe', 'today 12-m')
        geo = data.get('geo', '')
        
        if not keywords:
            raise ValidationError("At least one keyword is required")
        
        # Validate keywords
        validator.validate_keywords(keywords)
        
        # Initialize service lazily
        from app.services.trends_service import TrendsService
        trends_service = TrendsService()
        
        # Get summary
        summary = trends_service.get_trends_summary(keywords, timeframe, geo)
        
        return jsonify({
            'status': 'success',
            'summary': summary,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except ValidationError as e:
        logger.info(f"Validation error in trends summary endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except TrendsAPIError as e:
        logger.error(f"Trends API error in summary endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'TRENDS_API_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in trends summary endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@api_bp.route('/trends/compare', methods=['POST'])
def compare_trends():
    """
    Compare trends between multiple keywords
    """
    try:
        # Check content type
        if not request.is_json:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Content-Type must be application/json',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        # Validate request
        try:
            data = request.get_json()
        except Exception:
            return jsonify({
                'status': 'error',
                'error_code': 'VALIDATION_ERROR',
                'message': 'Invalid JSON in request body',
                'timestamp': datetime.utcnow().isoformat()
            }), 400
        
        if not data:
            raise ValidationError("Request body is required")
        
        # Extract parameters
        keywords = data.get('keywords', [])
        timeframe = data.get('timeframe', 'today 12-m')
        geo = data.get('geo', '')
        
        if not keywords or len(keywords) < 2:
            raise ValidationError("At least 2 keywords are required for comparison")
        
        # Validate keywords
        validator.validate_keywords(keywords)
        
        # Initialize service lazily
        from app.services.trends_service import TrendsService
        trends_service = TrendsService()
        
        # Get comparison
        comparison = trends_service.compare_keywords(keywords, timeframe, geo)
        
        return jsonify({
            'status': 'success',
            'comparison': comparison,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except ValidationError as e:
        logger.info(f"Validation error in trends compare endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'VALIDATION_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 400
        
    except TrendsAPIError as e:
        logger.error(f"Trends API error in compare endpoint: {e.message}")
        return jsonify({
            'status': 'error',
            'error_code': 'TRENDS_API_ERROR',
            'message': e.message,
            'timestamp': datetime.utcnow().isoformat()
        }), 500
        
    except Exception as e:
        logger.error(f"Unexpected error in trends compare endpoint: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'An unexpected error occurred',
            'timestamp': datetime.utcnow().isoformat()
        }), 500 
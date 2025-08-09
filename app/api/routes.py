"""
API Routes for Google Trends Quantile Forecaster
Main API endpoints and route definitions
"""

from flask import Blueprint, request, jsonify, current_app
from flask_limiter.util import get_remote_address
from app.utils.validators import validate_trends_request, validate_training_request, validate_prediction_request
from app.services.trends_service import TrendsService
from app.services.model_service import ModelService
from app.services.prediction_service import PredictionService

# Create blueprint
api_bp = Blueprint('api', __name__)

# Initialize services
trends_service = TrendsService()
model_service = ModelService()
prediction_service = PredictionService()


@api_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for API
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Google Trends Quantile Forecaster API',
        'version': current_app.config.get('API_VERSION', 'v1')
    })


@api_bp.route('/trends', methods=['POST'])
def get_trends():
    """
    Get Google Trends data for a keyword
    """
    try:
        # Validate request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body is required'
            }), 400
        
        # Validate input
        validation_result = validate_trends_request(data)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Validation Error',
                'message': validation_result['message']
            }), 400
        
        # Get trends data
        keyword = data.get('keyword')
        timeframe = data.get('timeframe', 'today 12-m')
        geo = data.get('geo', '')
        
        trends_data = trends_service.get_trends_data(keyword, timeframe, geo)
        
        return jsonify({
            'status': 'success',
            'data': trends_data,
            'keyword': keyword,
            'timeframe': timeframe,
            'geo': geo
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in get_trends: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to retrieve trends data'
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
            return jsonify({
                'error': 'Bad Request',
                'message': 'Request body is required'
            }), 400
        
        # Validate input
        validation_result = validate_training_request(data)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Validation Error',
                'message': validation_result['message']
            }), 400
        
        # Extract training parameters
        time_series_data = data.get('time_series_data')
        keyword = data.get('keyword')
        model_params = data.get('model_params', {})
        
        # Train model
        model_info = model_service.train_model(time_series_data, keyword, model_params)
        
        return jsonify({
            'status': 'success',
            'message': 'Model trained successfully',
            'model_id': model_info['model_id'],
            'keyword': keyword,
            'training_metrics': model_info['metrics']
        }), 201
        
    except Exception as e:
        current_app.logger.error(f"Error in train_model: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to train model'
        }), 500


@api_bp.route('/models/<model_id>/predict', methods=['POST'])
def generate_prediction(model_id):
    """
    Generate prediction using a trained model
    """
    try:
        # Validate request
        data = request.get_json() or {}
        
        # Validate input
        validation_result = validate_prediction_request(data)
        if not validation_result['valid']:
            return jsonify({
                'error': 'Validation Error',
                'message': validation_result['message']
            }), 400
        
        # Extract prediction parameters
        prediction_weeks = data.get('prediction_weeks', 
                                  current_app.config.get('DEFAULT_PREDICTION_WEEKS', 25))
        
        # Generate prediction
        prediction_result = prediction_service.generate_prediction(model_id, prediction_weeks)
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'prediction_weeks': prediction_weeks,
            'predictions': prediction_result['predictions'],
            'confidence_intervals': prediction_result.get('confidence_intervals', [])
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in generate_prediction: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to generate prediction'
        }), 500


@api_bp.route('/models/<model_id>', methods=['GET'])
def get_model_info(model_id):
    """
    Get information about a specific model
    """
    try:
        model_info = model_service.get_model_info(model_id)
        
        if not model_info:
            return jsonify({
                'error': 'Not Found',
                'message': f'Model with ID {model_id} not found'
            }), 404
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in get_model_info: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to retrieve model information'
        }), 500


@api_bp.route('/models', methods=['GET'])
def list_models():
    """
    List all available models
    """
    try:
        models = model_service.list_models()
        
        return jsonify({
            'status': 'success',
            'models': models,
            'count': len(models)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error in list_models: {str(e)}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'Failed to retrieve models list'
        }), 500 
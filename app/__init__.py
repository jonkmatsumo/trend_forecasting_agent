"""
Google Trends Quantile Forecaster Flask Application
Main application factory and configuration
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, jsonify, current_app
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def create_app(config_name=None):
    """
    Application factory pattern for Flask app creation
    """
    app = Flask(__name__)
    
    # Load configuration
    app.config.from_object('app.config.config.Config')
    
    # Initialize extensions
    CORS(app, origins=app.config.get('CORS_ORIGINS', ['*']))
    
    # Initialize rate limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[app.config.get('RATE_LIMIT_DEFAULT', '100/hour')],
        storage_uri=app.config.get('RATE_LIMIT_STORAGE_URL', 'memory://')
    )
    
    # Setup logging
    setup_logging(app)
    
    # Register blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Register error handlers
    register_error_handlers(app)
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'service': 'Google Trends Quantile Forecaster API',
            'version': app.config.get('API_VERSION', 'v1')
        })
    
    return app


def setup_logging(app):
    """
    Configure logging for the application
    """
    if not app.debug and not app.testing:
        # Create logs directory if it doesn't exist
        if not os.path.exists(app.config.get('LOGS_DIR', 'logs')):
            os.makedirs(app.config.get('LOGS_DIR', 'logs'))
        
        # File handler for logging
        file_handler = RotatingFileHandler(
            os.path.join(app.config.get('LOGS_DIR', 'logs'), 'app.log'),
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        
        file_handler.setFormatter(logging.Formatter(
            app.config.get('LOG_FORMAT', '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
        ))
        
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Google Trends Quantile Forecaster startup')


def register_error_handlers(app):
    """
    Register error handlers for the application
    """
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({
            'error': 'Bad Request',
            'message': 'The request could not be processed',
            'status_code': 400
        }), 400
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            'error': 'Not Found',
            'message': 'The requested resource was not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        return jsonify({
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred',
            'status_code': 500
        }), 500
    
    @app.errorhandler(429)
    def too_many_requests(error):
        return jsonify({
            'error': 'Too Many Requests',
            'message': 'Rate limit exceeded',
            'status_code': 429
        }), 429 
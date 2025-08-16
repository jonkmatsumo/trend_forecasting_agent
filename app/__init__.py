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

    # Register error handlers
    from app.utils.error_handlers import register_error_handlers
    register_error_handlers(app)

    # Register blueprints
    from app.api.routes import api_bp
    from app.api.agent_routes import agent_bp
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(agent_bp, url_prefix='/agent')

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
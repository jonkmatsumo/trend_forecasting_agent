"""
Configuration settings for the Flask application
"""

import os
from datetime import timedelta


class Config:
    """Base configuration class"""
    
    # Flask Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    # API Configuration
    API_VERSION = os.environ.get('API_VERSION', 'v1')
    API_TITLE = os.environ.get('API_TITLE', 'Google Trends Quantile Forecaster API')
    API_DESCRIPTION = os.environ.get('API_DESCRIPTION', 'API for forecasting Google Trends data using LSTM models')
    
    # Rate Limiting
    RATE_LIMIT_DEFAULT = os.environ.get('RATE_LIMIT_DEFAULT', '100/hour')
    RATE_LIMIT_STORAGE_URL = os.environ.get('RATE_LIMIT_STORAGE_URL', 'memory://')
    
    # MLflow Configuration
    MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
    MLFLOW_EXPERIMENT_NAME = os.environ.get('MLFLOW_EXPERIMENT_NAME', 'google_trends_forecaster')
    
    # Data Storage
    DATA_DIR = os.environ.get('DATA_DIR', 'data')
    MODELS_DIR = os.environ.get('MODELS_DIR', 'models')
    LOGS_DIR = os.environ.get('LOGS_DIR', 'logs')
    
    # Google Trends API Configuration
    PYTRENDS_DELAY = int(os.environ.get('PYTRENDS_DELAY', '1'))
    PYTRENDS_RETRIES = int(os.environ.get('PYTRENDS_RETRIES', '3'))
    PYTRENDS_TIMEOUT = int(os.environ.get('PYTRENDS_TIMEOUT', '30'))
    
    # Model Training Configuration
    DEFAULT_BATCH_SIZE = int(os.environ.get('DEFAULT_BATCH_SIZE', '5'))
    DEFAULT_EPOCHS = int(os.environ.get('DEFAULT_EPOCHS', '150'))
    DEFAULT_PREDICTION_WEEKS = int(os.environ.get('DEFAULT_PREDICTION_WEEKS', '25'))
    DEFAULT_LSTM_UNITS = int(os.environ.get('DEFAULT_LSTM_UNITS', '4'))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FORMAT = os.environ.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
    
    # Development Server
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', '5000'))


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MLFLOW_TRACKING_URI = 'sqlite:///:memory:'
    DATA_DIR = 'test_data'
    MODELS_DIR = 'test_models'
    LOGS_DIR = 'test_logs'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production-specific settings
    RATE_LIMIT_DEFAULT = '50/hour'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 
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
    
    # Agent Configuration
    AGENT_USE_GRAPH = os.environ.get('AGENT_USE_GRAPH', 'False').lower() == 'true'
    
    # LLM Intent Classification Configuration
    INTENT_LLM_ENABLED = os.environ.get('INTENT_LLM_ENABLED', 'False').lower() == 'true'
    INTENT_LLM_PROVIDER = os.environ.get('INTENT_LLM_PROVIDER', 'openai')
    INTENT_LLM_MODEL = os.environ.get('INTENT_LLM_MODEL', 'gpt-4o-mini')
    INTENT_LLM_TIMEOUT_MS = int(os.environ.get('INTENT_LLM_TIMEOUT_MS', '2000'))
    INTENT_LLM_MAX_TOKENS = int(os.environ.get('INTENT_LLM_MAX_TOKENS', '128'))
    INTENT_LLM_TEMPERATURE = float(os.environ.get('INTENT_LLM_TEMPERATURE', '0.0'))
    INTENT_LLM_API_KEY = os.environ.get('INTENT_LLM_API_KEY')
    INTENT_LLM_BASE_URL = os.environ.get('INTENT_LLM_BASE_URL', 'http://localhost:8000')
    
    # LLM Cache Configuration
    INTENT_LLM_CACHE_SIZE = int(os.environ.get('INTENT_LLM_CACHE_SIZE', '1000'))
    INTENT_LLM_CACHE_TTL_HOURS = int(os.environ.get('INTENT_LLM_CACHE_TTL_HOURS', '24'))
    
    # LLM Ensemble Configuration
    INTENT_LLM_ENSEMBLE_WEIGHTS = {
        "semantic": float(os.environ.get('INTENT_LLM_WEIGHT_SEMANTIC', '0.60')),
        "regex": float(os.environ.get('INTENT_LLM_WEIGHT_REGEX', '0.25')),
        "llm": float(os.environ.get('INTENT_LLM_WEIGHT_LLM', '0.15'))
    }
    
    INTENT_LLM_MINIMUMS = {
        "train": float(os.environ.get('INTENT_LLM_MIN_TRAIN', '0.55')),
        "evaluate": float(os.environ.get('INTENT_LLM_MIN_EVALUATE', '0.55')),
        "forecast": float(os.environ.get('INTENT_LLM_MIN_FORECAST', '0.45')),
        "compare": float(os.environ.get('INTENT_LLM_MIN_COMPARE', '0.45')),
        "summary": float(os.environ.get('INTENT_LLM_MIN_SUMMARY', '0.45')),
        "health": float(os.environ.get('INTENT_LLM_MIN_HEALTH', '0.30')),
        "list_models": float(os.environ.get('INTENT_LLM_MIN_LIST_MODELS', '0.30'))
    }
    
    # Phase 2: Circuit Breaker Configuration
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.environ.get('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5'))
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = float(os.environ.get('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60.0'))
    CIRCUIT_BREAKER_SUCCESS_THRESHOLD = int(os.environ.get('CIRCUIT_BREAKER_SUCCESS_THRESHOLD', '2'))
    
    # Phase 2: Retry Configuration
    RETRY_MAX_ATTEMPTS = int(os.environ.get('RETRY_MAX_ATTEMPTS', '3'))
    RETRY_BASE_DELAY = float(os.environ.get('RETRY_BASE_DELAY', '1.0'))
    RETRY_MAX_DELAY = float(os.environ.get('RETRY_MAX_DELAY', '60.0'))
    RETRY_EXPONENTIAL_BASE = float(os.environ.get('RETRY_EXPONENTIAL_BASE', '2.0'))
    RETRY_JITTER = os.environ.get('RETRY_JITTER', 'True').lower() == 'true'
    
    # Phase 2: Rate Limiter Configuration
    RATE_LIMITER_TOKENS_PER_SECOND = float(os.environ.get('RATE_LIMITER_TOKENS_PER_SECOND', '10.0'))
    RATE_LIMITER_BUCKET_SIZE = int(os.environ.get('RATE_LIMITER_BUCKET_SIZE', '100'))
    RATE_LIMITER_COST_PER_REQUEST = float(os.environ.get('RATE_LIMITER_COST_PER_REQUEST', '1.0'))
    RATE_LIMITER_BURST_SIZE = int(os.environ.get('RATE_LIMITER_BURST_SIZE', '20'))
    
    # Phase 2: Monitoring Configuration
    MONITORING_ENABLED = os.environ.get('MONITORING_ENABLED', 'True').lower() == 'true'
    MONITORING_METRICS_RETENTION_HOURS = int(os.environ.get('MONITORING_METRICS_RETENTION_HOURS', '24'))
    MONITORING_HEALTH_CHECK_INTERVAL = int(os.environ.get('MONITORING_HEALTH_CHECK_INTERVAL', '300'))
    MONITORING_EXPORT_FORMAT = os.environ.get('MONITORING_EXPORT_FORMAT', 'json')
    
    # Phase 2: Security Configuration
    SECURITY_ENABLED = os.environ.get('SECURITY_ENABLED', 'True').lower() == 'true'
    SECURITY_AUDIT_LOG_FILE = os.environ.get('SECURITY_AUDIT_LOG_FILE', 'logs/audit.log')
    SECURITY_DATA_REDACTION_ENABLED = os.environ.get('SECURITY_DATA_REDACTION_ENABLED', 'True').lower() == 'true'
    SECURITY_LOG_SENSITIVE_DATA = os.environ.get('SECURITY_LOG_SENSITIVE_DATA', 'False').lower() == 'true'
    
    # Phase 2: Cost Tracking Configuration
    COST_TRACKING_ENABLED = os.environ.get('COST_TRACKING_ENABLED', 'True').lower() == 'true'
    COST_PER_TOKEN = float(os.environ.get('COST_PER_TOKEN', '0.001'))
    COST_ALERT_THRESHOLD = float(os.environ.get('COST_ALERT_THRESHOLD', '10.0'))
    COST_DAILY_LIMIT = float(os.environ.get('COST_DAILY_LIMIT', '50.0'))


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'
    
    # Development-specific Phase 2 settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
    RETRY_MAX_ATTEMPTS = 2
    RATE_LIMITER_TOKENS_PER_SECOND = 20.0
    SECURITY_LOG_SENSITIVE_DATA = True


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True
    MLFLOW_TRACKING_URI = 'sqlite:///:memory:'
    DATA_DIR = 'test_data'
    MODELS_DIR = 'test_models'
    LOGS_DIR = 'test_logs'
    
    # Testing-specific Phase 2 settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 1
    RETRY_MAX_ATTEMPTS = 1
    RATE_LIMITER_TOKENS_PER_SECOND = 100.0
    MONITORING_ENABLED = False
    SECURITY_ENABLED = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    
    # Production-specific settings
    RATE_LIMIT_DEFAULT = '50/hour'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '').split(',') if os.environ.get('CORS_ORIGINS') else []
    
    # Production-specific Phase 2 settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD = 10
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 120.0
    RETRY_MAX_ATTEMPTS = 5
    RATE_LIMITER_TOKENS_PER_SECOND = 5.0
    SECURITY_LOG_SENSITIVE_DATA = False
    COST_ALERT_THRESHOLD = 5.0


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 
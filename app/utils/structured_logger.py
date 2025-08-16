"""
Structured Logging System
Provides JSON-formatted logging with request context and comprehensive metadata.
"""

import json
import logging
import hashlib
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from logging.handlers import RotatingFileHandler

from .request_context import get_current_request_id, get_request_metadata


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def __init__(self, include_request_context: bool = True):
        """Initialize the structured formatter.
        
        Args:
            include_request_context: Whether to include request context in logs
        """
        super().__init__()
        self.include_request_context = include_request_context
    
    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log data
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add request context if available
        if self.include_request_context:
            request_id = get_current_request_id()
            if request_id:
                log_data['request_id'] = request_id
            
            request_metadata = get_request_metadata()
            if request_metadata:
                log_data['request_metadata'] = request_metadata
        
        # Add extra fields from record
        if hasattr(record, 'event_type'):
            log_data['event_type'] = record.event_type
        
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        
        if hasattr(record, 'parameters'):
            log_data['parameters'] = record.parameters
        
        if hasattr(record, 'success'):
            log_data['success'] = record.success
        
        if hasattr(record, 'duration_ms'):
            log_data['duration_ms'] = record.duration_ms
        
        if hasattr(record, 'result'):
            log_data['result'] = record.result
        
        if hasattr(record, 'error_type'):
            log_data['error_type'] = record.error_type
        
        if hasattr(record, 'error_message'):
            log_data['error_message'] = record.error_message
        
        # Add any other extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info']:
                if not key.startswith('_'):
                    log_data[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, default=str)


class KeywordHasher:
    """Utility for hashing keywords to protect sensitive data."""
    
    @staticmethod
    def hash_keywords(keywords: List[str]) -> List[str]:
        """Hash a list of keywords for privacy.
        
        Args:
            keywords: List of keywords to hash
            
        Returns:
            List of hashed keywords
        """
        return [hashlib.sha256(kw.encode()).hexdigest()[:16] for kw in keywords]
    
    @staticmethod
    def hash_single_keyword(keyword: str) -> str:
        """Hash a single keyword for privacy.
        
        Args:
            keyword: Keyword to hash
            
        Returns:
            Hashed keyword
        """
        return hashlib.sha256(keyword.encode()).hexdigest()[:16]


class StructuredLogger:
    """Structured logger with comprehensive logging capabilities."""
    
    def __init__(self, name: str, log_level: str = "INFO"):
        """Initialize the structured logger.
        
        Args:
            name: Logger name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Set up logging handlers."""
        # Console handler with structured formatting
        console_handler = logging.StreamHandler()
        console_formatter = StructuredFormatter(include_request_context=True)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logs
        log_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'app.log'),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_formatter = StructuredFormatter(include_request_context=True)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
    
    def log_intent(self, intent: str, confidence: float = 1.0, **kwargs):
        """Log intent recognition.
        
        Args:
            intent: Recognized intent
            confidence: Confidence score (0.0 to 1.0)
            **kwargs: Additional intent data
        """
        self.logger.info("Intent recognized", extra={
            'event_type': 'intent_recognition',
            'intent': intent,
            'confidence': confidence,
            **kwargs
        })
    
    def log_keywords(self, keywords: List[str], operation: str, **kwargs):
        """Log keywords with hashing for privacy.
        
        Args:
            keywords: List of keywords
            operation: Operation being performed
            **kwargs: Additional keyword data
        """
        hashed_keywords = KeywordHasher.hash_keywords(keywords)
        self.logger.info("Keywords processed", extra={
            'event_type': 'keyword_processing',
            'operation': operation,
            'keyword_count': len(keywords),
            'hashed_keywords': hashed_keywords,
            **kwargs
        })
    
    def log_outcome(self, operation: str, success: bool, duration: float, **kwargs):
        """Log operation outcome.
        
        Args:
            operation: Operation name
            success: Whether operation succeeded
            duration: Operation duration in seconds
            **kwargs: Additional outcome data
        """
        self.logger.info("Operation completed", extra={
            'event_type': 'operation_outcome',
            'operation': operation,
            'success': success,
            'duration_ms': round(duration * 1000, 2),
            **kwargs
        })
    
    def log_latency(self, operation: str, latency_ms: float, **kwargs):
        """Log operation latency.
        
        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
            **kwargs: Additional latency data
        """
        self.logger.info("Latency measured", extra={
            'event_type': 'latency_measurement',
            'operation': operation,
            'latency_ms': latency_ms,
            **kwargs
        })
    
    def log_health_check(self, service: str, status: str, **kwargs):
        """Log health check results.
        
        Args:
            service: Service name
            status: Health status
            **kwargs: Additional health data
        """
        self.logger.info("Health check performed", extra={
            'event_type': 'health_check',
            'service': service,
            'status': status,
            **kwargs
        })
    
    def log_cache_stats(self, cache_size: int, cache_hits: int, cache_misses: int, **kwargs):
        """Log cache statistics.
        
        Args:
            cache_size: Number of items in cache
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            **kwargs: Additional cache data
        """
        hit_rate = cache_hits / (cache_hits + cache_misses) if (cache_hits + cache_misses) > 0 else 0
        self.logger.info("Cache statistics", extra={
            'event_type': 'cache_stats',
            'cache_size': cache_size,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'hit_rate': round(hit_rate, 3),
            **kwargs
        })
    
    def log_performance_metrics(self, operation: str, metrics: Dict[str, Any]):
        """Log performance metrics.
        
        Args:
            operation: Operation name
            metrics: Performance metrics dictionary
        """
        self.logger.info("Performance metrics", extra={
            'event_type': 'performance_metrics',
            'operation': operation,
            'metrics': metrics
        })
    
    def log_error_rate(self, operation: str, total_requests: int, error_count: int, **kwargs):
        """Log error rate statistics.
        
        Args:
            operation: Operation name
            total_requests: Total number of requests
            error_count: Number of errors
            **kwargs: Additional error data
        """
        error_rate = error_count / total_requests if total_requests > 0 else 0
        self.logger.info("Error rate calculated", extra={
            'event_type': 'error_rate',
            'operation': operation,
            'total_requests': total_requests,
            'error_count': error_count,
            'error_rate': round(error_rate, 3),
            **kwargs
        })
    
    def log_weekly_health(self, **kwargs):
        """Log weekly health summary.
        
        Args:
            **kwargs: Health summary data
        """
        self.logger.info("Weekly health summary", extra={
            'event_type': 'weekly_health_summary',
            **kwargs
        })
    
    def log_weekly_cache_stats(self, **kwargs):
        """Log weekly cache statistics.
        
        Args:
            **kwargs: Cache statistics data
        """
        self.logger.info("Weekly cache statistics", extra={
            'event_type': 'weekly_cache_stats',
            **kwargs
        })
    
    def log_weekly_performance(self, **kwargs):
        """Log weekly performance metrics.
        
        Args:
            **kwargs: Performance metrics data
        """
        self.logger.info("Weekly performance metrics", extra={
            'event_type': 'weekly_performance_metrics',
            **kwargs
        })
    
    def log_weekly_error_rates(self, **kwargs):
        """Log weekly error rate statistics.
        
        Args:
            **kwargs: Error rate data
        """
        self.logger.info("Weekly error rate statistics", extra={
            'event_type': 'weekly_error_rates',
            **kwargs
        })


def setup_structured_logging(log_level: str = "INFO", log_format: str = "json"):
    """Set up structured logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format (json or text)
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add structured formatter
    if log_format.lower() == "json":
        formatter = StructuredFormatter(include_request_context=True)
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        os.path.join(log_dir, 'app.log'),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def create_structured_logger(name: str) -> StructuredLogger:
    """Create a structured logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(name)


# Convenience functions for common logging patterns
def log_intent(name: str, intent: str, confidence: float = 1.0, **kwargs):
    """Log intent recognition with a named logger."""
    logger = create_structured_logger(name)
    logger.log_intent(intent, confidence, **kwargs)


def log_keywords(name: str, keywords: List[str], operation: str, **kwargs):
    """Log keywords with a named logger."""
    logger = create_structured_logger(name)
    logger.log_keywords(keywords, operation, **kwargs)


def log_outcome(name: str, operation: str, success: bool, duration: float, **kwargs):
    """Log operation outcome with a named logger."""
    logger = create_structured_logger(name)
    logger.log_outcome(operation, success, duration, **kwargs)


def log_latency(name: str, operation: str, latency_ms: float, **kwargs):
    """Log latency with a named logger."""
    logger = create_structured_logger(name)
    logger.log_latency(operation, latency_ms, **kwargs) 
"""
Request Context Management
Handles request ID generation and context tracking across service calls.
"""

import uuid
import logging
import threading
from typing import Optional, Dict, Any
from contextlib import contextmanager
from datetime import datetime


class RequestContext:
    """Thread-local request context for tracking request information."""
    
    def __init__(self):
        """Initialize the request context."""
        self._local = threading.local()
    
    @property
    def request_id(self) -> Optional[str]:
        """Get the current request ID."""
        return getattr(self._local, 'request_id', None)
    
    @request_id.setter
    def request_id(self, value: str):
        """Set the current request ID."""
        self._local.request_id = value
    
    @property
    def start_time(self) -> Optional[datetime]:
        """Get the request start time."""
        return getattr(self._local, 'start_time', None)
    
    @start_time.setter
    def start_time(self, value: datetime):
        """Set the request start time."""
        self._local.start_time = value
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get request metadata."""
        return getattr(self._local, 'metadata', {})
    
    @metadata.setter
    def metadata(self, value: Dict[str, Any]):
        """Set request metadata."""
        self._local.metadata = value
    
    def add_metadata(self, key: str, value: Any):
        """Add a key-value pair to request metadata."""
        if not hasattr(self._local, 'metadata'):
            self._local.metadata = {}
        self._local.metadata[key] = value
    
    def clear(self):
        """Clear the current request context."""
        if hasattr(self._local, 'request_id'):
            delattr(self._local, 'request_id')
        if hasattr(self._local, 'start_time'):
            delattr(self._local, 'start_time')
        if hasattr(self._local, 'metadata'):
            delattr(self._local, 'metadata')


# Global request context instance
request_context = RequestContext()


def generate_request_id() -> str:
    """Generate a unique request ID."""
    return str(uuid.uuid4())


def get_current_request_id() -> Optional[str]:
    """Get the current request ID from context."""
    return request_context.request_id


def get_request_metadata() -> Dict[str, Any]:
    """Get the current request metadata."""
    return request_context.metadata.copy()


def add_request_metadata(key: str, value: Any):
    """Add metadata to the current request."""
    request_context.add_metadata(key, value)


@contextmanager
def request_context_manager(request_id: Optional[str] = None):
    """Context manager for request tracking.
    
    Args:
        request_id: Optional request ID. If not provided, one will be generated.
    
    Yields:
        The request ID for the current request.
    """
    # Generate request ID if not provided
    if request_id is None:
        request_id = generate_request_id()
    
    # Set up request context
    request_context.request_id = request_id
    request_context.start_time = datetime.utcnow()
    request_context.metadata = {}
    
    try:
        yield request_id
    finally:
        # Clear request context
        request_context.clear()


def log_request_start(operation: str, **kwargs):
    """Log the start of a request operation.
    
    Args:
        operation: Name of the operation being performed
        **kwargs: Additional parameters to log
    """
    request_id = get_current_request_id()
    if request_id:
        logger = logging.getLogger(__name__)
        logger.info(f"Request {request_id}: Starting {operation}", extra={
            'request_id': request_id,
            'operation': operation,
            'parameters': kwargs,
            'event_type': 'request_start'
        })
        
        # Add operation metadata
        add_request_metadata('operation', operation)
        add_request_metadata('parameters', kwargs)


def log_request_end(operation: str, success: bool, duration: float, **kwargs):
    """Log the end of a request operation.
    
    Args:
        operation: Name of the operation being performed
        success: Whether the operation was successful
        duration: Duration of the operation in seconds
        **kwargs: Additional result data to log
    """
    request_id = get_current_request_id()
    if request_id:
        logger = logging.getLogger(__name__)
        logger.info(f"Request {request_id}: Completed {operation}", extra={
            'request_id': request_id,
            'operation': operation,
            'success': success,
            'duration_ms': round(duration * 1000, 2),
            'result': kwargs,
            'event_type': 'request_end'
        })


def log_request_error(operation: str, error: Exception, duration: float):
    """Log a request error.
    
    Args:
        operation: Name of the operation being performed
        error: The exception that occurred
        duration: Duration of the operation in seconds
    """
    request_id = get_current_request_id()
    if request_id:
        logger = logging.getLogger(__name__)
        logger.error(f"Request {request_id}: Error in {operation}", extra={
            'request_id': request_id,
            'operation': operation,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'duration_ms': round(duration * 1000, 2),
            'event_type': 'request_error'
        }, exc_info=True)


def get_request_duration() -> Optional[float]:
    """Get the duration of the current request in seconds."""
    start_time = request_context.start_time
    if start_time:
        return (datetime.utcnow() - start_time).total_seconds()
    return None


def create_request_logger(name: str) -> logging.Logger:
    """Create a logger that automatically includes request context.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance with request context support
    """
    logger = logging.getLogger(name)
    
    # Add request context to log records
    class RequestContextFilter(logging.Filter):
        def filter(self, record):
            record.request_id = get_current_request_id()
            record.request_metadata = get_request_metadata()
            return True
    
    # Add filter if not already present
    if not any(isinstance(f, RequestContextFilter) for f in logger.filters):
        logger.addFilter(RequestContextFilter())
    
    return logger 
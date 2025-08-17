"""
Retry Handler with Exponential Backoff
Provides retry logic for LLM service calls with configurable backoff strategies.
"""

import time
import random
import logging
from typing import Callable, Any, Optional, List, Type
from dataclasses import dataclass
from functools import wraps


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: List[Type[Exception]] = None
    
    def __post_init__(self):
        if self.retryable_exceptions is None:
            self.retryable_exceptions = [Exception]


class RetryHandler:
    """Retry handler with exponential backoff."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize retry handler.
        
        Args:
            config: Retry configuration
        """
        self.config = config or RetryConfig()
        self.logger = logging.getLogger("retry_handler")
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
                
            except tuple(self.config.retryable_exceptions) as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    self.logger.error(f"All {self.config.max_attempts} retry attempts failed")
                    raise last_exception
                
                delay = self._calculate_delay(attempt)
                self.logger.warning(
                    f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                )
                time.sleep(delay)
        
        # This should never be reached, but just in case
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (exponential_base ^ (attempt - 1))
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        # Cap at max delay
        return min(delay, self.config.max_delay)


def retry(config: Optional[RetryConfig] = None):
    """Decorator for retry functionality.
    
    Args:
        config: Retry configuration
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = RetryHandler(config)
            return handler.call(func, *args, **kwargs)
        return wrapper
    return decorator


class AdaptiveRetryHandler(RetryHandler):
    """Adaptive retry handler that adjusts strategy based on failure patterns."""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        """Initialize adaptive retry handler.
        
        Args:
            config: Base retry configuration
        """
        super().__init__(config)
        self.failure_history = []
        self.success_history = []
        self.max_history_size = 100
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with adaptive retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
        """
        start_time = time.time()
        
        try:
            result = super().call(func, *args, **kwargs)
            self._record_success(time.time() - start_time)
            return result
            
        except Exception as e:
            self._record_failure(time.time() - start_time, type(e))
            raise
    
    def _record_success(self, duration: float):
        """Record successful call."""
        self.success_history.append({
            'timestamp': time.time(),
            'duration': duration
        })
        
        # Trim history
        if len(self.success_history) > self.max_history_size:
            self.success_history = self.success_history[-self.max_history_size:]
    
    def _record_failure(self, duration: float, exception_type: Type[Exception]):
        """Record failed call."""
        self.failure_history.append({
            'timestamp': time.time(),
            'duration': duration,
            'exception_type': exception_type
        })
        
        # Trim history
        if len(self.failure_history) > self.max_history_size:
            self.failure_history = self.failure_history[-self.max_history_size:]
    
    def get_failure_rate(self, window_seconds: float = 300) -> float:
        """Calculate recent failure rate.
        
        Args:
            window_seconds: Time window in seconds
            
        Returns:
            Failure rate as float between 0 and 1
        """
        now = time.time()
        cutoff = now - window_seconds
        
        recent_failures = len([
            f for f in self.failure_history 
            if f['timestamp'] >= cutoff
        ])
        
        recent_successes = len([
            s for s in self.success_history 
            if s['timestamp'] >= cutoff
        ])
        
        total_recent = recent_failures + recent_successes
        if total_recent == 0:
            return 0.0
        
        return recent_failures / total_recent
    
    def get_stats(self) -> dict:
        """Get retry statistics.
        
        Returns:
            Dictionary with retry statistics
        """
        return {
            'failure_rate_5min': self.get_failure_rate(300),
            'failure_rate_1hour': self.get_failure_rate(3600),
            'total_failures': len(self.failure_history),
            'total_successes': len(self.success_history),
            'config': {
                'max_attempts': self.config.max_attempts,
                'base_delay': self.config.base_delay,
                'max_delay': self.config.max_delay
            }
        } 
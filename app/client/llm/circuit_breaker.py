"""
Circuit Breaker Pattern Implementation
Provides fault tolerance for LLM service calls with automatic recovery.
"""

import time
import logging
from enum import Enum
from typing import Optional, Callable, Any
from dataclasses import dataclass


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5        # Failures before opening circuit
    recovery_timeout: float = 60.0    # Seconds to wait before half-open
    expected_exception: type = Exception  # Exception type to count as failure
    success_threshold: int = 2        # Successes before closing circuit


class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration settings
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.logger = logging.getLogger(f"circuit_breaker.{name}")
        
        # State tracking
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.success_count = 0
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open or function fails
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self._set_half_open()
            else:
                raise Exception(f"Circuit breaker '{self.name}' is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except self.config.expected_exception as e:
            self._on_failure()
            raise e
        except Exception as e:
            # Unexpected exception, don't count as circuit breaker failure
            self.logger.warning(f"Unexpected exception in circuit breaker: {e}")
            raise e
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._set_closed()
        else:
            # Reset failure count on success
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.config.failure_threshold:
            self._set_open()
    
    def _set_open(self):
        """Set circuit to open state."""
        if self.state != CircuitState.OPEN:
            self.logger.warning(f"Circuit breaker '{self.name}' opened after {self.failure_count} failures")
            self.state = CircuitState.OPEN
    
    def _set_half_open(self):
        """Set circuit to half-open state."""
        self.logger.info(f"Circuit breaker '{self.name}' attempting reset")
        self.state = CircuitState.HALF_OPEN
        self.success_count = 0
    
    def _set_closed(self):
        """Set circuit to closed state."""
        self.logger.info(f"Circuit breaker '{self.name}' closed after {self.success_count} successes")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if not self.last_failure_time:
            return False
        
        return time.time() - self.last_failure_time >= self.config.recovery_timeout
    
    def get_status(self) -> dict:
        """Get current circuit breaker status.
        
        Returns:
            Dictionary with status information
        """
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "recovery_timeout": self.config.recovery_timeout,
                "success_threshold": self.config.success_threshold
            }
        }
    
    def reset(self):
        """Manually reset circuit breaker to closed state."""
        self.logger.info(f"Circuit breaker '{self.name}' manually reset")
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None 
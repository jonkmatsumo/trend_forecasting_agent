"""
Unit tests for Circuit Breaker Pattern
Tests the circuit breaker implementation for fault tolerance.
"""

import time
import pytest
from unittest.mock import Mock, patch

from app.services.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state."""
        cb = CircuitBreaker("test")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
    
    def test_circuit_breaker_successful_calls(self):
        """Test successful calls don't open circuit."""
        cb = CircuitBreaker("test")
        
        def success_func():
            return "success"
        
        for _ in range(10):
            result = cb.call(success_func)
            assert result == "success"
            assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("test error")
        
        # First 2 failures should not open circuit
        for _ in range(2):
            with pytest.raises(Exception):
                cb.call(failing_func)
            assert cb.state == CircuitState.CLOSED
        
        # 3rd failure should open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
    
    def test_circuit_breaker_half_open_recovery(self):
        """Test circuit recovers through half-open state."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            recovery_timeout=0.1,
            success_threshold=2
        )
        cb = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("test error")
        
        def success_func():
            return "success"
        
        # Open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(0.2)
        
        # Make a call to trigger state transition check
        # This should transition to half-open state
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second success should close circuit
        result = cb.call(success_func)
        assert result == "success"
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_get_status(self):
        """Test circuit breaker status reporting."""
        cb = CircuitBreaker("test")
        status = cb.get_status()
        
        assert status["name"] == "test"
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert "config" in status
    
    def test_circuit_breaker_manual_reset(self):
        """Test manual circuit breaker reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test", config)
        
        def failing_func():
            raise Exception("test error")
        
        # Open circuit
        with pytest.raises(Exception):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Manual reset
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.success_count == 0
    
    def test_circuit_breaker_custom_exception(self):
        """Test circuit breaker with custom exception types."""
        class CustomException(Exception):
            pass
        
        config = CircuitBreakerConfig(
            failure_threshold=1,
            expected_exception=CustomException
        )
        cb = CircuitBreaker("test", config)
        
        def failing_func():
            raise CustomException("custom error")
        
        def other_failing_func():
            raise ValueError("other error")
        
        # Custom exception should open circuit
        with pytest.raises(CustomException):
            cb.call(failing_func)
        assert cb.state == CircuitState.OPEN
        
        # Reset for next test
        cb.reset()
        
        # Other exception should not open circuit
        with pytest.raises(ValueError):
            cb.call(other_failing_func)
        assert cb.state == CircuitState.CLOSED
    
    def test_circuit_breaker_config_defaults(self):
        """Test circuit breaker configuration defaults."""
        config = CircuitBreakerConfig()
        assert config.failure_threshold == 5
        assert config.recovery_timeout == 60.0
        assert config.success_threshold == 2
        assert config.expected_exception == Exception
    
    def test_circuit_breaker_thread_safety(self):
        """Test circuit breaker thread safety."""
        import threading
        
        cb = CircuitBreaker("test")
        results = []
        
        def worker():
            try:
                result = cb.call(lambda: "success")
                results.append(result)
            except Exception as e:
                results.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should succeed
        assert len(results) == 5
        assert all(r == "success" for r in results)
        assert cb.state == CircuitState.CLOSED


if __name__ == "__main__":
    pytest.main([__file__]) 
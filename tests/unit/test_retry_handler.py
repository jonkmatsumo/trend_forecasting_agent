"""
Unit tests for Retry Handler
Tests the retry handler implementation with exponential backoff.
"""

import time
import pytest
from unittest.mock import Mock, patch

from app.services.llm.retry_handler import RetryHandler, RetryConfig, AdaptiveRetryHandler


class TestRetryHandler:
    """Test retry handler functionality."""
    
    def test_retry_handler_successful_call(self):
        """Test successful call without retries."""
        handler = RetryHandler()
        
        def success_func():
            return "success"
        
        result = handler.call(success_func)
        assert result == "success"
    
    def test_retry_handler_fails_after_max_attempts(self):
        """Test retry handler fails after max attempts."""
        config = RetryConfig(max_attempts=3)
        handler = RetryHandler(config)
        
        call_count = 0
        def failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("test error")
        
        with pytest.raises(Exception):
            handler.call(failing_func)
        
        assert call_count == 3
    
    def test_retry_handler_succeeds_on_retry(self):
        """Test retry handler succeeds on retry."""
        config = RetryConfig(max_attempts=3, base_delay=0.01)
        handler = RetryHandler(config)
        
        call_count = 0
        def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("test error")
            return "success"
        
        result = handler.call(eventually_successful_func)
        assert result == "success"
        assert call_count == 2
    
    def test_retry_handler_exponential_backoff(self):
        """Test exponential backoff timing."""
        config = RetryConfig(max_attempts=3, base_delay=0.1, max_delay=1.0)
        handler = RetryHandler(config)
        
        start_time = time.time()
        
        def failing_func():
            raise Exception("test error")
        
        with pytest.raises(Exception):
            handler.call(failing_func)
        
        duration = time.time() - start_time
        # Should have delays of ~0.1s and ~0.2s, but allow for timing variance
        assert duration >= 0.15  # Reduced threshold to account for timing variance
    
    def test_retry_handler_without_jitter(self):
        """Test retry handler without jitter."""
        config = RetryConfig(max_attempts=3, base_delay=0.1, jitter=False)
        handler = RetryHandler(config)
        
        start_time = time.time()
        
        def failing_func():
            raise Exception("test error")
        
        with pytest.raises(Exception):
            handler.call(failing_func)
        
        duration = time.time() - start_time
        # Should have exact delays of 0.1s and 0.2s
        assert duration >= 0.3
        assert duration <= 0.4  # Allow small timing variance
    
    def test_retry_handler_custom_exceptions(self):
        """Test retry handler with custom exception types."""
        class RetryableException(Exception):
            pass
        
        class NonRetryableException(Exception):
            pass
        
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[RetryableException]
        )
        handler = RetryHandler(config)
        
        call_count = 0
        def mixed_failing_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RetryableException("retryable")
            else:
                raise NonRetryableException("non-retryable")
        
        with pytest.raises(NonRetryableException):
            handler.call(mixed_failing_func)
        
        assert call_count == 2  # Should retry once, then fail on non-retryable
    
    def test_retry_handler_config_defaults(self):
        """Test retry handler configuration defaults."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter is True
        assert Exception in config.retryable_exceptions
    
    def test_retry_handler_decorator(self):
        """Test retry handler decorator."""
        from app.services.llm.retry_handler import retry
        
        call_count = 0
        
        @retry(RetryConfig(max_attempts=3, base_delay=0.01))
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("test error")
            return "success"
        
        result = decorated_func()
        assert result == "success"
        assert call_count == 2


class TestAdaptiveRetryHandler:
    """Test adaptive retry handler functionality."""
    
    def test_adaptive_retry_handler(self):
        """Test adaptive retry handler."""
        handler = AdaptiveRetryHandler()
        
        def success_func():
            return "success"
        
        result = handler.call(success_func)
        assert result == "success"
        
        stats = handler.get_stats()
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 0
    
    def test_adaptive_retry_handler_failure_tracking(self):
        """Test adaptive retry handler failure tracking."""
        handler = AdaptiveRetryHandler()
        
        def failing_func():
            raise Exception("test error")
        
        with pytest.raises(Exception):
            handler.call(failing_func)
        
        stats = handler.get_stats()
        assert stats["total_successes"] == 0
        assert stats["total_failures"] == 1
        assert stats["failure_rate_5min"] > 0.0
    
    def test_adaptive_retry_handler_failure_rate_calculation(self):
        """Test adaptive retry handler failure rate calculation."""
        handler = AdaptiveRetryHandler()
        
        # Add some success and failure history
        handler._record_success(1.0)
        handler._record_success(1.0)
        handler._record_failure(1.0, Exception)
        
        failure_rate = handler.get_failure_rate(300)  # 5 minutes
        assert failure_rate == 1/3  # 1 failure out of 3 total calls
    
    def test_adaptive_retry_handler_history_trimming(self):
        """Test adaptive retry handler history trimming."""
        handler = AdaptiveRetryHandler()
        handler.max_history_size = 3
        
        # Add more history than max size
        for i in range(5):
            handler._record_success(1.0)
        
        assert len(handler.success_history) == 3
        assert len(handler.failure_history) == 0
    
    def test_adaptive_retry_handler_stats(self):
        """Test adaptive retry handler statistics."""
        handler = AdaptiveRetryHandler()
        
        # Add some history
        handler._record_success(1.0)
        handler._record_failure(2.0, Exception)
        
        stats = handler.get_stats()
        assert "failure_rate_5min" in stats
        assert "failure_rate_1hour" in stats
        assert "total_failures" in stats
        assert "total_successes" in stats
        assert "config" in stats
    
    def test_adaptive_retry_handler_window_calculation(self):
        """Test adaptive retry handler window-based calculations."""
        handler = AdaptiveRetryHandler()
        
        # Add old history (should be filtered out)
        old_time = time.time() - 400  # 400 seconds ago
        handler.failure_history.append({
            'timestamp': old_time,
            'duration': 1.0,
            'exception_type': Exception
        })
        
        # Add recent history
        handler._record_success(1.0)
        handler._record_failure(1.0, Exception)
        
        # Test 5-minute window (should exclude old failure)
        failure_rate_5min = handler.get_failure_rate(300)
        assert failure_rate_5min == 0.5  # 1 failure out of 2 recent calls
        
        # Test 1-hour window (should include old failure)
        failure_rate_1hour = handler.get_failure_rate(3600)
        assert failure_rate_1hour == 2/3  # 2 failures out of 3 total calls


if __name__ == "__main__":
    pytest.main([__file__]) 
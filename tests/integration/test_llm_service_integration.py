"""
Integration tests for LLM Service
Tests the integration of LLM service components with all system features.
"""

import time
import pytest
from unittest.mock import Mock, patch

from app.services.llm.llm_client import LLMClient, LLMError, IntentClassificationResult


class TestLLMServiceIntegration:
    """Test LLM service with all system components integration."""
    
    def test_llm_service_integration(self):
        """Test LLM service with all system components."""
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9,
                    rationale="test rationale"
                )
            
            def _health_check_impl(self):
                return True
        
        # Test with all system components
        client = MockLLMClient("test-model")
        
        # Test successful classification
        result = client.classify_intent("test query")
        assert result.intent == "test"
        assert result.confidence == 0.9
        
        # Test health check
        assert client.health_check() is True
        
        # Test model info includes all system stats
        info = client.get_model_info()
        assert "circuit_breaker_status" in info
        assert "rate_limiter_stats" in info
        assert "request_stats" in info
    
    def test_llm_service_metrics_recording(self):
        """Test that metrics are recorded during LLM operations."""
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9,
                    tokens_used=100,
                    cost=0.01
                )
            
            def _health_check_impl(self):
                return True
        
        client = MockLLMClient("test-model")
        
        # Perform classification
        result = client.classify_intent("test query")
        
        # Check that metrics were recorded
        assert client.request_count == 1
        assert client.total_tokens == 100
        assert client.total_cost == 0.01
        
        # Check that result has latency (may be 0 for very fast operations)
        assert result.latency_ms is not None
        assert result.latency_ms >= 0
    
    def test_llm_service_failure_handling(self):
        """Test failure handling in LLM service."""
        class FailingLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                raise Exception("Simulated failure")
            
            def _health_check_impl(self):
                return False
        
        client = FailingLLMClient("test-model")
        
        # Test that failure is properly handled
        with pytest.raises(LLMError):
            client.classify_intent("test query")
        
        # Check that failure metrics were recorded
        assert client.request_count == 1
        assert client.total_tokens == 0
        assert client.total_cost == 0.0
        
        # Test health check failure
        assert client.health_check() is False
    
    def test_llm_service_rate_limiting(self):
        """Test rate limiting in LLM service."""
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        # Create client with very restrictive rate limiting
        from app.services.llm.rate_limiter import RateLimitConfig
        rate_config = RateLimitConfig(tokens_per_second=0.1, bucket_size=1)
        
        client = MockLLMClient("test-model", rate_limit_config=rate_config)
        
        # First request should succeed
        result1 = client.classify_intent("test query 1")
        assert result1.intent == "test"
        
        # Second request should fail due to rate limiting
        with pytest.raises(LLMError) as exc_info:
            client.classify_intent("test query 2")
        assert "Rate limit exceeded" in str(exc_info.value)
    
    def test_llm_service_circuit_breaker(self):
        """Test circuit breaker in LLM service."""
        class FailingLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                raise Exception("Simulated failure")
            
            def _health_check_impl(self):
                return False
        
        # Create client with low failure threshold
        from app.services.llm.circuit_breaker import CircuitBreakerConfig
        circuit_config = CircuitBreakerConfig(failure_threshold=1)
        
        client = FailingLLMClient("test-model", circuit_breaker_config=circuit_config)
        
        # First failure should open circuit
        with pytest.raises(LLMError):
            client.classify_intent("test query")
        
        # Second request should fail immediately due to open circuit
        with pytest.raises(LLMError) as exc_info:
            client.classify_intent("test query")
        assert "Circuit breaker" in str(exc_info.value)
    
    def test_llm_service_retry_logic(self):
        """Test retry logic in LLM service."""
        call_count = 0
        
        class EventuallySuccessfulLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise Exception("Simulated temporary failure")
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        # Create client with retry configuration
        from app.services.llm.retry_handler import RetryConfig
        retry_config = RetryConfig(max_attempts=3, base_delay=0.01)
        
        client = EventuallySuccessfulLLMClient("test-model", retry_config=retry_config)
        
        # Should succeed after retry
        result = client.classify_intent("test query")
        assert result.intent == "test"
        assert call_count == 2  # Should have been called twice
    
    def test_llm_service_health_status(self):
        """Test health status reporting."""
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        client = MockLLMClient("test-model")
        
        # Get comprehensive health status
        health_status = client.get_health_status()
        
        assert "service_healthy" in health_status
        assert "circuit_breaker" in health_status
        assert "rate_limiter" in health_status
        assert "model_info" in health_status
        
        # Check circuit breaker status
        cb_status = health_status["circuit_breaker"]
        assert "state" in cb_status
        assert "failure_count" in cb_status
        
        # Check rate limiter status
        rl_status = health_status["rate_limiter"]
        assert "current_tokens" in rl_status
        assert "request_count" in rl_status
    
    def test_system_components_initialization(self):
        """Test that all system components are properly initialized."""
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        client = MockLLMClient("test-model")
        
        # Check that all system components are initialized
        assert hasattr(client, 'circuit_breaker')
        assert hasattr(client, 'retry_handler')
        assert hasattr(client, 'rate_limiter')
        
        # Check component types
        from app.services.llm.circuit_breaker import CircuitBreaker
        from app.services.llm.retry_handler import RetryHandler
        from app.services.llm.rate_limiter import TokenBucketRateLimiter
        
        assert isinstance(client.circuit_breaker, CircuitBreaker)
        assert isinstance(client.retry_handler, RetryHandler)
        assert isinstance(client.rate_limiter, TokenBucketRateLimiter)
    
    def test_system_component_configuration(self):
        """Test that all system components can be configured."""
        from app.services.llm.circuit_breaker import CircuitBreakerConfig
        from app.services.llm.retry_handler import RetryConfig
        from app.services.llm.rate_limiter import RateLimitConfig
        
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        # Create custom configurations
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30.0,
            success_threshold=1
        )
        
        retry_config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0
        )
        
        rate_config = RateLimitConfig(
            tokens_per_second=5.0,
            bucket_size=50,
            cost_per_request=2.0
        )
        
        # Create client with custom configurations
        client = MockLLMClient(
            "test-model",
            circuit_breaker_config=circuit_config,
            retry_config=retry_config,
            rate_limit_config=rate_config
        )
        
        # Verify configurations were applied
        assert client.circuit_breaker.config.failure_threshold == 3
        assert client.retry_handler.config.max_attempts == 5
        assert client.rate_limiter.config.tokens_per_second == 5.0


if __name__ == "__main__":
    pytest.main([__file__]) 
"""
LLM Client Abstract Base Class
Defines the interface for LLM-based intent classification with Phase 2 enhancements.
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

from app.client.llm.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from app.client.llm.retry_handler import RetryHandler, RetryConfig
from app.client.llm.rate_limiter import TokenBucketRateLimiter, RateLimitConfig
from app.services.monitoring_service import monitoring_service
from app.services.security_service import security_service


@dataclass
class IntentClassificationResult:
    """Result from LLM intent classification."""
    intent: str
    confidence: float
    rationale: Optional[str] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None
    cost: Optional[float] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients with Phase 2 enhancements."""
    
    def __init__(self, model: str, timeout_ms: int = 2000, max_tokens: int = 128,
                 circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                 retry_config: Optional[RetryConfig] = None,
                 rate_limit_config: Optional[RateLimitConfig] = None):
        """Initialize LLM client with Phase 2 components.
        
        Args:
            model: Model name/identifier
            timeout_ms: Request timeout in milliseconds
            max_tokens: Maximum tokens for response
            circuit_breaker_config: Circuit breaker configuration
            retry_config: Retry configuration
            rate_limit_config: Rate limiting configuration
        """
        self.model = model
        self.timeout_ms = timeout_ms
        self.max_tokens = max_tokens
        
        # Initialize Phase 2 components
        self.circuit_breaker = CircuitBreaker(
            name=f"llm_{self.__class__.__name__.lower()}",
            config=circuit_breaker_config
        )
        
        self.retry_handler = RetryHandler(retry_config)
        self.rate_limiter = TokenBucketRateLimiter(rate_limit_config)
        
        # Track request statistics
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0.0
    
    def classify_intent(self, query: str, user_id: Optional[str] = None) -> IntentClassificationResult:
        """Classify query intent with Phase 2 enhancements.
        
        Args:
            query: User query to classify
            user_id: Optional user identifier for security logging
            
        Returns:
            IntentClassificationResult with intent, confidence, and rationale
            
        Raises:
            LLMError: If classification fails
        """
        start_time = time.time()
        
        try:
            # Rate limiting
            if not self.rate_limiter.acquire(timeout=5.0):
                raise LLMError("Rate limit exceeded")
            
            # Circuit breaker and retry logic
            def _classify():
                return self._classify_intent_impl(query)
            
            result = self.circuit_breaker.call(
                lambda: self.retry_handler.call(_classify)
            )
            
            # Calculate metrics
            duration = time.time() - start_time
            latency_ms = duration * 1000
            
            # Update result with metrics
            result.latency_ms = latency_ms
            
            # Record metrics
            self._record_metrics(query, result, duration, True, user_id)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failure metrics
            self._record_metrics(query, None, duration, False, user_id)
            
            # Re-raise as LLMError
            if isinstance(e, LLMError):
                raise
            else:
                raise LLMError(f"Classification failed: {str(e)}")
    
    def _record_metrics(self, query: str, result: Optional[IntentClassificationResult], 
                       duration: float, success: bool, user_id: Optional[str]):
        """Record metrics for monitoring and security.
        
        Args:
            query: User query
            result: Classification result
            duration: Request duration
            success: Whether request was successful
            user_id: Optional user identifier
        """
        # Update local statistics
        self.request_count += 1
        if result and result.tokens_used:
            try:
                tokens_used = int(result.tokens_used) if result.tokens_used else 0
                self.total_tokens += tokens_used
            except (TypeError, ValueError):
                # Handle cases where tokens_used is not numeric (e.g., mocks)
                pass
        
        if result and result.cost:
            try:
                cost = float(result.cost) if result.cost else 0.0
                self.total_cost += cost
            except (TypeError, ValueError):
                # Handle cases where cost is not numeric (e.g., mocks)
                pass
        
        # Record monitoring metrics
        provider = self.__class__.__name__.lower()
        tokens_used = 0
        cost = 0.0
        
        if result:
            try:
                tokens_used = int(result.tokens_used) if result.tokens_used else 0
            except (TypeError, ValueError):
                tokens_used = 0
            
            try:
                cost = float(result.cost) if result.cost else 0.0
            except (TypeError, ValueError):
                cost = 0.0
        
        monitoring_service.record_llm_request(
            provider=provider,
            model=self.model,
            duration=duration,
            tokens_used=tokens_used,
            success=success,
            cost=cost
        )
        
        # Record security metrics
        if success and result:
            security_service.log_llm_request(
                user_id=user_id,
                query=query,
                response=str(result),
                provider=provider,
                model=self.model,
                success=success,
                duration=duration
            )
    
    @abstractmethod
    def _classify_intent_impl(self, query: str) -> IntentClassificationResult:
        """Implementation of intent classification (to be implemented by subclasses).
        
        Args:
            query: User query to classify
            
        Returns:
            IntentClassificationResult with intent, confidence, and rationale
            
        Raises:
            LLMError: If classification fails
        """
        pass
    
    def health_check(self) -> bool:
        """Check if the LLM service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Use circuit breaker for health check
            return self.circuit_breaker.call(self._health_check_impl)
        except Exception:
            return False
    
    @abstractmethod
    def _health_check_impl(self) -> bool:
        """Implementation of health check (to be implemented by subclasses).
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "timeout_ms": self.timeout_ms,
            "max_tokens": self.max_tokens,
            "provider": self.__class__.__name__,
            "circuit_breaker_status": self.circuit_breaker.get_status(),
            "rate_limiter_stats": self.rate_limiter.get_stats(),
            "request_stats": {
                "total_requests": self.request_count,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status.
        
        Returns:
            Dictionary with health status information
        """
        return {
            "service_healthy": self.health_check(),
            "circuit_breaker": self.circuit_breaker.get_status(),
            "rate_limiter": self.rate_limiter.get_stats(),
            "model_info": self.get_model_info()
        }


class LLMError(Exception):
    """Exception raised when LLM operations fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None):
        """Initialize LLM error.
        
        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response_text: Response text if applicable
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text 
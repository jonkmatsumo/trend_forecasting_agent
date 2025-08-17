"""
Token Bucket Rate Limiter
Implements token bucket algorithm for rate limiting LLM API calls.
"""

import time
import threading
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    tokens_per_second: float = 10.0      # Tokens added per second
    bucket_size: int = 100               # Maximum tokens in bucket
    cost_per_request: float = 1.0        # Tokens consumed per request
    burst_size: int = 20                 # Maximum burst requests


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        """Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config or RateLimitConfig()
        self.logger = logging.getLogger("rate_limiter")
        
        # Token bucket state
        self.tokens = self.config.bucket_size
        self.last_refill = time.time()
        
        # Thread safety
        self.lock = threading.Lock()
        
        # Statistics
        self.request_count = 0
        self.blocked_count = 0
        self.total_wait_time = 0.0
    
    def acquire(self, tokens: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        """Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire (defaults to cost_per_request)
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        if tokens is None:
            tokens = self.config.cost_per_request
        
        start_time = time.time()
        
        with self.lock:
            while True:
                # Refill tokens
                self._refill_tokens()
                
                # Check if we have enough tokens
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    self.request_count += 1
                    return True
                
                # Check timeout
                if timeout is not None and (time.time() - start_time) >= timeout:
                    self.blocked_count += 1
                    self.logger.warning(f"Rate limit timeout after {timeout}s")
                    return False
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.config.tokens_per_second
                
                # Add small buffer for timing precision
                wait_time = min(wait_time + 0.1, timeout or wait_time)
                
                self.logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
                time.sleep(wait_time)
    
    def _refill_tokens(self):
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Calculate tokens to add
        tokens_to_add = elapsed * self.config.tokens_per_second
        
        # Add tokens, but don't exceed bucket size
        self.tokens = min(
            self.config.bucket_size,
            self.tokens + tokens_to_add
        )
        
        self.last_refill = now
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics.
        
        Returns:
            Dictionary with rate limiter statistics
        """
        with self.lock:
            return {
                "current_tokens": self.tokens,
                "bucket_size": self.config.bucket_size,
                "tokens_per_second": self.config.tokens_per_second,
                "request_count": self.request_count,
                "blocked_count": self.blocked_count,
                "success_rate": (
                    (self.request_count - self.blocked_count) / max(self.request_count, 1)
                ),
                "total_wait_time": self.total_wait_time
            }


class MultiTenantRateLimiter:
    """Rate limiter that supports multiple tenants/users."""
    
    def __init__(self, default_config: Optional[RateLimitConfig] = None):
        """Initialize multi-tenant rate limiter.
        
        Args:
            default_config: Default configuration for new tenants
        """
        self.default_config = default_config or RateLimitConfig()
        self.logger = logging.getLogger("multi_tenant_rate_limiter")
        
        # Per-tenant rate limiters
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.lock = threading.Lock()
    
    def get_limiter(self, tenant_id: str) -> TokenBucketRateLimiter:
        """Get or create rate limiter for tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            
        Returns:
            Rate limiter for the tenant
        """
        with self.lock:
            if tenant_id not in self.limiters:
                self.limiters[tenant_id] = TokenBucketRateLimiter(self.default_config)
                self.logger.info(f"Created rate limiter for tenant: {tenant_id}")
            
            return self.limiters[tenant_id]
    
    def acquire(self, tenant_id: str, tokens: Optional[float] = None, 
                timeout: Optional[float] = None) -> bool:
        """Acquire tokens for a specific tenant.
        
        Args:
            tenant_id: Tenant identifier
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        limiter = self.get_limiter(tenant_id)
        return limiter.acquire(tokens, timeout)
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all tenants.
        
        Returns:
            Dictionary mapping tenant IDs to their statistics
        """
        with self.lock:
            return {
                tenant_id: limiter.get_stats()
                for tenant_id, limiter in self.limiters.items()
            }
    
    def remove_tenant(self, tenant_id: str):
        """Remove rate limiter for a tenant.
        
        Args:
            tenant_id: Tenant identifier to remove
        """
        with self.lock:
            if tenant_id in self.limiters:
                del self.limiters[tenant_id]
                self.logger.info(f"Removed rate limiter for tenant: {tenant_id}")


class CostTrackingRateLimiter(TokenBucketRateLimiter):
    """Rate limiter with cost tracking capabilities."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None, 
                 cost_per_token: float = 0.001):
        """Initialize cost tracking rate limiter.
        
        Args:
            config: Rate limiting configuration
            cost_per_token: Cost per token in dollars
        """
        super().__init__(config)
        self.cost_per_token = cost_per_token
        self.total_cost = 0.0
        self.daily_costs = defaultdict(float)
    
    def acquire(self, tokens: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        """Acquire tokens and track cost.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait
            
        Returns:
            True if tokens were acquired, False if timeout
        """
        if tokens is None:
            tokens = self.config.cost_per_request
        
        success = super().acquire(tokens, timeout)
        
        if success:
            cost = tokens * self.cost_per_token
            self.total_cost += cost
            
            # Track daily costs
            today = time.strftime("%Y-%m-%d")
            self.daily_costs[today] += cost
        
        return success
    
    def get_cost_stats(self) -> Dict[str, Any]:
        """Get cost tracking statistics.
        
        Returns:
            Dictionary with cost statistics
        """
        stats = self.get_stats()
        stats.update({
            "total_cost": self.total_cost,
            "cost_per_token": self.cost_per_token,
            "daily_costs": dict(self.daily_costs),
            "estimated_monthly_cost": self._estimate_monthly_cost()
        })
        return stats
    
    def _estimate_monthly_cost(self) -> float:
        """Estimate monthly cost based on current usage patterns.
        
        Returns:
            Estimated monthly cost in dollars
        """
        if not self.daily_costs:
            return 0.0
        
        # Calculate average daily cost
        avg_daily_cost = sum(self.daily_costs.values()) / len(self.daily_costs)
        
        # Estimate monthly cost (30 days)
        return avg_daily_cost * 30 
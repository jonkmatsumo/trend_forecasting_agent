"""
Monitoring Service
Provides comprehensive monitoring capabilities including health checks, cache stats, performance metrics, and error rates.
"""

import time
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass, field

from app.utils.structured_logger import create_structured_logger
from app.config.adapter_config import create_adapter


@dataclass
class PerformanceMetrics:
    """Performance metrics for an operation."""
    operation: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_duration: float = 0.0
    min_duration: float = float('inf')
    max_duration: float = 0.0
    recent_durations: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests
    
    @property
    def avg_duration(self) -> float:
        """Calculate average duration."""
        if self.total_requests == 0:
            return 0.0
        return self.total_duration / self.total_requests
    
    @property
    def p95_duration(self) -> float:
        """Calculate 95th percentile duration."""
        if not self.recent_durations:
            return 0.0
        sorted_durations = sorted(self.recent_durations)
        index = int(0.95 * len(sorted_durations))
        return sorted_durations[index] if index < len(sorted_durations) else sorted_durations[-1]
    
    def record_request(self, success: bool, duration: float):
        """Record a request."""
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        
        self.total_duration += duration
        self.min_duration = min(self.min_duration, duration)
        self.max_duration = max(self.max_duration, duration)
        self.recent_durations.append(duration)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'operation': self.operation,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'success_rate': round(self.success_rate, 3),
            'error_rate': round(self.error_rate, 3),
            'avg_duration_ms': round(self.avg_duration * 1000, 2),
            'min_duration_ms': round(self.min_duration * 1000, 2) if self.min_duration != float('inf') else 0,
            'max_duration_ms': round(self.max_duration * 1000, 2),
            'p95_duration_ms': round(self.p95_duration * 1000, 2)
        }


@dataclass
class CacheStats:
    """Cache statistics."""
    cache_size: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_evictions: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.cache_hits + self.cache_misses
        if total_requests == 0:
            return 0.0
        return self.cache_hits / total_requests
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'cache_size': self.cache_size,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_evictions': self.cache_evictions,
            'hit_rate': round(self.hit_rate, 3),
            'last_updated': self.last_updated.isoformat()
        }


class MonitoringService:
    """Comprehensive monitoring service for the forecaster application."""
    
    def __init__(self):
        """Initialize the monitoring service."""
        self.logger = create_structured_logger("monitoring")
        self.adapter = create_adapter()
        
        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetrics] = defaultdict(
            lambda: PerformanceMetrics("unknown")
        )
        
        # Cache statistics
        self.cache_stats = CacheStats()
        
        # Health status
        self.health_status = "unknown"
        self.last_health_check = None
        
        # Monitoring thread
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Weekly statistics
        self.weekly_stats = {
            'health_checks': [],
            'cache_stats': [],
            'performance_metrics': [],
            'error_rates': []
        }
    
    def start_monitoring(self, interval_seconds: int = 3600):
        """Start the monitoring service.
        
        Args:
            interval_seconds: Interval between monitoring checks in seconds
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.logger.logger.warning("Monitoring service already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.logger.info("Monitoring service started")
    
    def stop_monitoring_service(self):
        """Stop the monitoring service."""
        self.stop_monitoring.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.logger.info("Monitoring service stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while not self.stop_monitoring.wait(interval_seconds):
            try:
                self._perform_health_check()
                self._update_cache_stats()
                self._log_performance_metrics()
                self._log_error_rates()
                
                # Weekly logging (every 7 days)
                if self._should_log_weekly():
                    self._log_weekly_summary()
                    
            except Exception as e:
                self.logger.logger.error(f"Error in monitoring loop: {str(e)}", exc_info=True)
    
    def _perform_health_check(self):
        """Perform health check on the forecaster service."""
        try:
            start_time = time.time()
            health_response = self.adapter.health()
            duration = time.time() - start_time
            
            status = health_response.get("status", "unknown")
            self.health_status = status
            self.last_health_check = datetime.utcnow()
            
            self.logger.log_health_check(
                service="forecaster",
                status=status,
                duration_ms=round(duration * 1000, 2),
                response=health_response
            )
            
            # Record performance metrics
            self._record_performance("health_check", status == "healthy", duration)
            
        except Exception as e:
            self.health_status = "unhealthy"
            self.last_health_check = datetime.utcnow()
            self.logger.logger.error(f"Health check failed: {str(e)}", exc_info=True)
            self._record_performance("health_check", False, 0.0)
    
    def _update_cache_stats(self):
        """Update cache statistics."""
        try:
            start_time = time.time()
            cache_response = self.adapter.cache_stats()
            duration = time.time() - start_time
            
            if cache_response.get("status") == "success":
                cache_data = cache_response.get("cache_stats", {})
                self.cache_stats.cache_size = cache_data.get("cache_size", 0)
                self.cache_stats.cache_hits = cache_data.get("cache_hits", 0)
                self.cache_stats.cache_misses = cache_data.get("cache_misses", 0)
                self.cache_stats.last_updated = datetime.utcnow()
                
                self.logger.log_cache_stats(
                    cache_size=self.cache_stats.cache_size,
                    cache_hits=self.cache_stats.cache_hits,
                    cache_misses=self.cache_stats.cache_misses,
                    duration_ms=round(duration * 1000, 2)
                )
            
            # Record performance metrics
            self._record_performance("cache_stats", cache_response.get("status") == "success", duration)
            
        except Exception as e:
            self.logger.logger.error(f"Cache stats update failed: {str(e)}", exc_info=True)
            self._record_performance("cache_stats", False, 0.0)
    
    def _log_performance_metrics(self):
        """Log current performance metrics."""
        for operation, metrics in self.performance_metrics.items():
            if metrics.total_requests > 0:
                self.logger.log_performance_metrics(operation, metrics.to_dict())
    
    def _log_error_rates(self):
        """Log current error rates."""
        for operation, metrics in self.performance_metrics.items():
            if metrics.total_requests > 0:
                self.logger.log_error_rate(
                    operation=operation,
                    total_requests=metrics.total_requests,
                    error_count=metrics.failed_requests
                )
    
    def _should_log_weekly(self) -> bool:
        """Check if weekly logging should be performed."""
        # Simple check: log weekly if it's been more than 7 days since last weekly log
        # In a real implementation, you might want to check the actual day of week
        return True  # For demo purposes, always log weekly
    
    def _log_weekly_summary(self):
        """Log weekly summary statistics."""
        # Health summary
        self.logger.log_weekly_health(
            status=self.health_status,
            last_check=self.last_health_check.isoformat() if self.last_health_check else None
        )
        
        # Cache summary
        self.logger.log_weekly_cache_stats(**self.cache_stats.to_dict())
        
        # Performance summary
        performance_summary = {}
        for operation, metrics in self.performance_metrics.items():
            if metrics.total_requests > 0:
                performance_summary[operation] = metrics.to_dict()
        
        self.logger.log_weekly_performance(performance_summary=performance_summary)
        
        # Error rates summary
        error_rates_summary = {}
        for operation, metrics in self.performance_metrics.items():
            if metrics.total_requests > 0:
                error_rates_summary[operation] = {
                    'total_requests': metrics.total_requests,
                    'error_count': metrics.failed_requests,
                    'error_rate': round(metrics.error_rate, 3)
                }
        
        self.logger.log_weekly_error_rates(error_rates=error_rates_summary)
    
    def _record_performance(self, operation: str, success: bool, duration: float):
        """Record performance metrics for an operation."""
        if operation not in self.performance_metrics:
            self.performance_metrics[operation] = PerformanceMetrics(operation)
        
        self.performance_metrics[operation].record_request(success, duration)
    
    def record_request(self, operation: str, success: bool, duration: float):
        """Record a request for performance tracking.
        
        Args:
            operation: Operation name
            success: Whether the request was successful
            duration: Request duration in seconds
        """
        self._record_performance(operation, success, duration)
    
    def get_performance_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance metrics.
        
        Args:
            operation: Specific operation to get metrics for, or None for all
            
        Returns:
            Performance metrics dictionary
        """
        if operation:
            metrics = self.performance_metrics.get(operation)
            return metrics.to_dict() if metrics else {}
        else:
            return {
                op: metrics.to_dict() 
                for op, metrics in self.performance_metrics.items()
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get current cache statistics.
        
        Returns:
            Cache statistics dictionary
        """
        return self.cache_stats.to_dict()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status.
        
        Returns:
            Health status dictionary
        """
        return {
            'status': self.health_status,
            'last_check': self.last_health_check.isoformat() if self.last_health_check else None
        }
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary.
        
        Returns:
            Monitoring summary dictionary
        """
        return {
            'health': self.get_health_status(),
            'cache': self.get_cache_stats(),
            'performance': self.get_performance_metrics(),
            'monitoring_active': self.monitoring_thread and self.monitoring_thread.is_alive()
        }


# Global monitoring service instance
monitoring_service = MonitoringService()


def get_monitoring_service() -> MonitoringService:
    """Get the global monitoring service instance."""
    return monitoring_service


def start_monitoring(interval_seconds: int = 3600):
    """Start the monitoring service.
    
    Args:
        interval_seconds: Interval between monitoring checks in seconds
    """
    monitoring_service.start_monitoring(interval_seconds)


def stop_monitoring():
    """Stop the monitoring service."""
    monitoring_service.stop_monitoring_service()


def record_request(operation: str, success: bool, duration: float):
    """Record a request for performance tracking.
    
    Args:
        operation: Operation name
        success: Whether the request was successful
        duration: Request duration in seconds
    """
    monitoring_service.record_request(operation, success, duration) 
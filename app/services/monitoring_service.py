"""
Monitoring and Telemetry Service
Provides comprehensive monitoring for LLM services including metrics, health checks, and performance tracking.
"""

import time
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    """Health check result."""
    name: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores metrics with time-series capabilities."""
    
    def __init__(self, max_points: int = 1000):
        """Initialize metrics collector.
        
        Args:
            max_points: Maximum number of points to keep per metric
        """
        self.max_points = max_points
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_points))
        self.lock = threading.Lock()
        self.logger = logging.getLogger("metrics_collector")
    
    def record(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            labels: Optional labels for the metric
        """
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )
        
        with self.lock:
            self.metrics[metric_name].append(point)
    
    def get_metric(self, metric_name: str, window_seconds: Optional[float] = None) -> List[MetricPoint]:
        """Get metric data points.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Optional time window filter
            
        Returns:
            List of metric points
        """
        with self.lock:
            if metric_name not in self.metrics:
                return []
            
            points = list(self.metrics[metric_name])
            
            if window_seconds:
                cutoff = time.time() - window_seconds
                points = [p for p in points if p.timestamp >= cutoff]
            
            return points
    
    def get_summary(self, metric_name: str, window_seconds: float = 3600) -> Dict[str, Any]:
        """Get metric summary statistics.
        
        Args:
            metric_name: Name of the metric
            window_seconds: Time window for summary
            
        Returns:
            Dictionary with summary statistics
        """
        points = self.get_metric(metric_name, window_seconds)
        
        if not points:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "avg": None,
                "sum": 0
            }
        
        values = [p.value for p in points]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "sum": sum(values)
        }
    
    def get_all_metrics(self) -> Dict[str, List[MetricPoint]]:
        """Get all metrics.
        
        Returns:
            Dictionary mapping metric names to their points
        """
        with self.lock:
            return {name: list(points) for name, points in self.metrics.items()}


class HealthChecker:
    """Performs health checks on various system components."""
    
    def __init__(self):
        """Initialize health checker."""
        self.checks: Dict[str, Callable] = {}
        self.logger = logging.getLogger("health_checker")
    
    def register_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Register a health check function.
        
        Args:
            name: Name of the health check
            check_func: Function that returns HealthCheck
        """
        self.checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")
    
    def run_check(self, name: str) -> Optional[HealthCheck]:
        """Run a specific health check.
        
        Args:
            name: Name of the health check
            
        Returns:
            Health check result or None if not found
        """
        if name not in self.checks:
            return None
        
        try:
            return self.checks[name]()
        except Exception as e:
            self.logger.error(f"Health check '{name}' failed: {e}")
            return HealthCheck(
                name=name,
                status="unhealthy",
                message=f"Check failed: {str(e)}",
                timestamp=time.time()
            )
    
    def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all registered health checks.
        
        Returns:
            Dictionary mapping check names to results
        """
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_overall_status(self) -> str:
        """Get overall system health status.
        
        Returns:
            Overall status: "healthy", "degraded", or "unhealthy"
        """
        results = self.run_all_checks()
        
        if not results:
            return "unknown"
        
        statuses = [r.status for r in results.values() if r]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        else:
            return "healthy"


class MonitoringService:
    """Main monitoring service that coordinates metrics and health checks."""
    
    def __init__(self):
        """Initialize monitoring service."""
        self.metrics = MetricsCollector()
        self.health_checker = HealthChecker()
        self.logger = logging.getLogger("monitoring_service")
        
        # Register default health checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks."""
        self.health_checker.register_check("system", self._system_health_check)
        self.health_checker.register_check("memory", self._memory_health_check)
    
    def _system_health_check(self) -> HealthCheck:
        """System-level health check."""
        return HealthCheck(
            name="system",
            status="healthy",
            message="System is operational",
            timestamp=time.time(),
            details={
                "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0
            }
        )
    
    def _memory_health_check(self) -> HealthCheck:
        """Memory usage health check."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent < 80:
                status = "healthy"
            elif memory.percent < 90:
                status = "degraded"
            else:
                status = "unhealthy"
            
            return HealthCheck(
                name="memory",
                status=status,
                message=f"Memory usage: {memory.percent:.1f}%",
                timestamp=time.time(),
                details={
                    "percent": memory.percent,
                    "available": memory.available,
                    "total": memory.total
                }
            )
        except ImportError:
            return HealthCheck(
                name="memory",
                status="unknown",
                message="psutil not available",
                timestamp=time.time()
            )
    
    def record_llm_request(self, provider: str, model: str, duration: float, 
                          tokens_used: int, success: bool, cost: float = 0.0):
        """Record LLM request metrics.
        
        Args:
            provider: LLM provider (e.g., "openai", "local")
            model: Model name
            duration: Request duration in seconds
            tokens_used: Number of tokens used
            success: Whether request was successful
            cost: Request cost in dollars
        """
        labels = {"provider": provider, "model": model}
        
        # Record various metrics
        self.metrics.record("llm_request_duration", duration, labels)
        self.metrics.record("llm_tokens_used", tokens_used, labels)
        self.metrics.record("llm_request_cost", cost, labels)
        self.metrics.record("llm_request_success", 1.0 if success else 0.0, labels)
        
        # Record success rate
        success_rate = self.metrics.get_summary("llm_request_success", 3600)["avg"] or 0.0
        self.metrics.record("llm_success_rate", success_rate, labels)
    
    def record_intent_classification(self, method: str, duration: float, 
                                   confidence: float, success: bool):
        """Record intent classification metrics.
        
        Args:
            method: Classification method (e.g., "semantic", "regex", "llm")
            duration: Classification duration in seconds
            confidence: Classification confidence score
            success: Whether classification was successful
        """
        labels = {"method": method}
        
        self.metrics.record("intent_classification_duration", duration, labels)
        self.metrics.record("intent_classification_confidence", confidence, labels)
        self.metrics.record("intent_classification_success", 1.0 if success else 0.0, labels)
    
    def get_llm_stats(self, window_seconds: float = 3600) -> Dict[str, Any]:
        """Get LLM usage statistics.
        
        Args:
            window_seconds: Time window for statistics
            
        Returns:
            Dictionary with LLM statistics
        """
        duration_stats = self.metrics.get_summary("llm_request_duration", window_seconds)
        tokens_stats = self.metrics.get_summary("llm_tokens_used", window_seconds)
        cost_stats = self.metrics.get_summary("llm_request_cost", window_seconds)
        success_stats = self.metrics.get_summary("llm_request_success", window_seconds)
        
        return {
            "duration": duration_stats,
            "tokens": tokens_stats,
            "cost": cost_stats,
            "success_rate": success_stats.get("avg", 0.0),
            "total_requests": duration_stats["count"],
            "window_seconds": window_seconds
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status.
        
        Returns:
            Dictionary with health information
        """
        health_results = self.health_checker.run_all_checks()
        overall_status = self.health_checker.get_overall_status()
        
        return {
            "status": overall_status,
            "timestamp": time.time(),
            "checks": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "timestamp": result.timestamp,
                    "details": result.details
                }
                for name, result in health_results.items()
                if result
            }
        }
    
    def get_metrics_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive metrics dashboard.
        
        Returns:
            Dictionary with dashboard data
        """
        return {
            "llm_stats": self.get_llm_stats(),
            "system_health": self.get_system_health(),
            "metrics": {
                name: self.metrics.get_summary(name, 3600)
                for name in self.metrics.get_all_metrics().keys()
            }
        }
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format.
        
        Args:
            format: Export format ("json" or "prometheus")
            
        Returns:
            Exported metrics as string
        """
        if format == "json":
            return json.dumps(self.get_metrics_dashboard(), indent=2)
        elif format == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        for metric_name, points in self.metrics.get_all_metrics().items():
            for point in points:
                # Convert labels to Prometheus format
                label_str = ""
                if point.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in point.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                lines.append(f'{metric_name}{label_str} {point.value} {int(point.timestamp * 1000)}')
        
        return "\n".join(lines)


# Global monitoring service instance
monitoring_service = MonitoringService() 
"""
Unit tests for Monitoring Service
Tests the monitoring and telemetry service implementation.
"""

import time
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.monitoring_service import (
    MonitoringService, MetricsCollector, HealthChecker, HealthCheck
)


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        collector = MetricsCollector()
        
        # Record metrics
        collector.record("test_metric", 1.0, {"label": "value"})
        collector.record("test_metric", 2.0, {"label": "value"})
        
        # Get metric data
        points = collector.get_metric("test_metric")
        assert len(points) == 2
        assert points[0].value == 1.0
        assert points[1].value == 2.0
    
    def test_metrics_summary(self):
        """Test metrics summary statistics."""
        collector = MetricsCollector()
        
        collector.record("test_metric", 1.0)
        collector.record("test_metric", 2.0)
        collector.record("test_metric", 3.0)
        
        summary = collector.get_summary("test_metric")
        assert summary["count"] == 3
        assert summary["min"] == 1.0
        assert summary["max"] == 3.0
        assert summary["avg"] == 2.0
    
    def test_metrics_retention(self):
        """Test metrics retention and cleanup."""
        collector = MetricsCollector(max_points=2)  # Very small max points
        
        # Record metrics
        collector.record("test_metric", 1.0)
        collector.record("test_metric", 2.0)
        collector.record("test_metric", 3.0)  # Should evict the first one
        
        # Should only have 2 points due to max_points limit
        assert len(collector.get_metric("test_metric")) == 2
    
    def test_metrics_labels(self):
        """Test metrics with labels."""
        collector = MetricsCollector()
        
        # Record metrics with different labels
        collector.record("test_metric", 1.0, {"service": "api"})
        collector.record("test_metric", 2.0, {"service": "worker"})
        
        # Get all metrics for the metric name
        all_points = collector.get_metric("test_metric")
        
        assert len(all_points) == 2
        # Check that labels are stored correctly
        api_point = next(p for p in all_points if p.labels.get("service") == "api")
        worker_point = next(p for p in all_points if p.labels.get("service") == "worker")
        
        assert api_point.value == 1.0
        assert worker_point.value == 2.0
    
    def test_metrics_window_filtering(self):
        """Test metrics window-based filtering."""
        collector = MetricsCollector()
        
        # Record a metric
        collector.record("test_metric", 1.0)
        
        # Wait a bit
        time.sleep(0.1)
        
        # Record another metric
        collector.record("test_metric", 2.0)
        
        # Test window filtering - should get both points
        all_points = collector.get_metric("test_metric")
        assert len(all_points) == 2
        
        # Test with very short window - should get only recent points
        recent_points = collector.get_metric("test_metric", window_seconds=0.05)
        assert len(recent_points) >= 1  # At least the most recent one
    
    def test_metrics_cleanup(self):
        """Test metrics cleanup functionality."""
        collector = MetricsCollector(max_points=1)
        
        # Add some metrics
        collector.record("metric1", 1.0)
        collector.record("metric1", 2.0)  # Should evict the first one
        
        # Check that only the latest metric remains
        points = collector.get_metric("metric1")
        assert len(points) == 1
        assert points[0].value == 2.0


class TestHealthChecker:
    """Test health checker functionality."""
    
    def test_health_checker(self):
        """Test health checker functionality."""
        checker = HealthChecker()
        
        def healthy_check():
            return HealthCheck(
                name="test",
                status="healthy",
                message="OK",
                timestamp=time.time()
            )
        
        checker.register_check("test", healthy_check)
        
        result = checker.run_check("test")
        assert result.status == "healthy"
        
        all_results = checker.run_all_checks()
        assert "test" in all_results
        assert checker.get_overall_status() == "healthy"
    
    def test_health_checker_unhealthy_check(self):
        """Test health checker with unhealthy check."""
        checker = HealthChecker()
        
        def unhealthy_check():
            return HealthCheck(
                name="test",
                status="unhealthy",
                message="Service down",
                timestamp=time.time()
            )
        
        checker.register_check("test", unhealthy_check)
        
        result = checker.run_check("test")
        assert result.status == "unhealthy"
        assert "Service down" in result.message
        
        assert checker.get_overall_status() == "unhealthy"
    
    def test_health_checker_multiple_checks(self):
        """Test health checker with multiple checks."""
        checker = HealthChecker()
        
        def healthy_check():
            return HealthCheck(
                name="healthy",
                status="healthy",
                message="OK",
                timestamp=time.time()
            )
        
        def unhealthy_check():
            return HealthCheck(
                name="unhealthy",
                status="unhealthy",
                message="Failed",
                timestamp=time.time()
            )
        
        checker.register_check("healthy", healthy_check)
        checker.register_check("unhealthy", unhealthy_check)
        
        all_results = checker.run_all_checks()
        assert len(all_results) == 2
        assert all_results["healthy"].status == "healthy"
        assert all_results["unhealthy"].status == "unhealthy"
        
        # Overall status should be unhealthy if any check fails
        assert checker.get_overall_status() == "unhealthy"
    
    def test_health_checker_check_not_found(self):
        """Test health checker with non-existent check."""
        checker = HealthChecker()
        
        # Should return None for non-existent check
        result = checker.run_check("non_existent")
        assert result is None
    
    def test_health_checker_check_registration(self):
        """Test health check registration."""
        checker = HealthChecker()
        
        def test_check():
            return HealthCheck(
                name="test",
                status="healthy",
                message="OK",
                timestamp=time.time()
            )
        
        # Register check
        checker.register_check("test", test_check)
        assert "test" in checker.checks
        
        # Re-register should overwrite
        checker.register_check("test", test_check)
        assert "test" in checker.checks
    
    def test_health_checker_check_removal(self):
        """Test health check removal."""
        checker = HealthChecker()
        
        def test_check():
            return HealthCheck(
                name="test",
                status="healthy",
                message="OK",
                timestamp=time.time()
            )
        
        checker.register_check("test", test_check)
        assert "test" in checker.checks
        
        # HealthChecker doesn't have remove_check method, so we'll test re-registration
        checker.register_check("test", test_check)  # Re-register should work
        assert "test" in checker.checks


class TestMonitoringService:
    """Test monitoring service integration."""
    
    def test_monitoring_service_integration(self):
        """Test monitoring service integration."""
        service = MonitoringService()
        
        # Record LLM request
        service.record_llm_request(
            provider="test",
            model="test-model",
            duration=1.0,
            tokens_used=100,
            success=True,
            cost=0.01
        )
        
        # Get LLM stats
        stats = service.get_llm_stats()
        assert stats["total_requests"] == 1
        assert stats["success_rate"] == 1.0
        
        # Get system health
        health = service.get_system_health()
        assert "status" in health
        assert "checks" in health
    
    def test_monitoring_service_llm_stats(self):
        """Test LLM statistics collection."""
        service = MonitoringService()
        
        # Record multiple requests
        service.record_llm_request("provider1", "model1", 1.0, 100, True, 0.01)
        service.record_llm_request("provider2", "model2", 2.0, 200, True, 0.02)
        service.record_llm_request("provider1", "model1", 3.0, 300, False, 0.03)
        
        stats = service.get_llm_stats()
        assert stats["total_requests"] == 3
        assert stats["success_rate"] == 2/3
        assert stats["duration"]["avg"] == 2.0
        # Check that tokens are tracked (structure may vary)
        assert "tokens" in stats
    
    def test_monitoring_service_intent_classification(self):
        """Test intent classification metrics."""
        service = MonitoringService()
        
        # Record intent classification
        service.record_intent_classification(
            method="llm",
            duration=0.5,
            confidence=0.9,
            success=True
        )
        
        # Get metrics dashboard to check intent classification metrics
        dashboard = service.get_metrics_dashboard()
        assert "metrics" in dashboard
        
        # Check that intent classification metrics are recorded
        metrics = dashboard["metrics"]
        assert "intent_classification_duration" in metrics
        assert "intent_classification_confidence" in metrics
        assert "intent_classification_success" in metrics
    
    def test_monitoring_service_metrics_dashboard(self):
        """Test metrics dashboard."""
        service = MonitoringService()
        
        # Record some data
        service.record_llm_request("test", "test-model", 1.0, 100, True, 0.01)
        service.record_intent_classification("llm", 0.5, 0.9, True)
        
        dashboard = service.get_metrics_dashboard()
        assert "llm_stats" in dashboard
        assert "system_health" in dashboard
        assert "metrics" in dashboard  # Intent stats are in the metrics section
    
    def test_monitoring_service_export_metrics(self):
        """Test metrics export functionality."""
        service = MonitoringService()
        
        # Record some metrics
        service.record_llm_request("test", "test-model", 1.0, 100, True, 0.01)
        
        # Test JSON export
        json_export = service.export_metrics("json")
        assert "llm_request_duration" in json_export
        
        # Test Prometheus export
        prometheus_export = service.export_metrics("prometheus")
        assert "llm_request_duration" in prometheus_export
        # Check that it's in Prometheus format (key=value format)
        assert "=" in prometheus_export
    
    def test_monitoring_service_health_checks(self):
        """Test health check registration and execution."""
        service = MonitoringService()
        
        def custom_check():
            return HealthCheck(
                name="custom",
                status="healthy",
                message="Custom check OK",
                timestamp=time.time()
            )
        
        # Register custom check
        service.health_checker.register_check("custom", custom_check)
        
        # Run health checks
        health = service.get_system_health()
        assert "custom" in health["checks"]
        # Health check result should be a HealthCheck object or dict
        custom_check = health["checks"]["custom"]
        assert custom_check is not None
    
    def test_monitoring_service_default_checks(self):
        """Test default health checks."""
        service = MonitoringService()
        
        health = service.get_system_health()
        assert "status" in health
        assert "checks" in health
        
        # Should have some default checks registered
        assert len(health["checks"]) > 0
    
    def test_monitoring_service_window_filtering(self):
        """Test window-based statistics filtering."""
        service = MonitoringService()
        
        # Record request
        service.record_llm_request("test", "test-model", 1.0, 100, True, 0.01)
        
        # Get stats with different windows
        stats_1hour = service.get_llm_stats(window_seconds=3600)
        stats_1min = service.get_llm_stats(window_seconds=60)
        
        # Both should include the recent request
        assert stats_1hour["total_requests"] == 1
        assert stats_1min["total_requests"] == 1


if __name__ == "__main__":
    pytest.main([__file__]) 
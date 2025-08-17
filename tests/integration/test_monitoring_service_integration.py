"""
Integration tests for Monitoring Service
Tests the integration of monitoring service with all system components.
"""

import pytest
from unittest.mock import Mock, patch

from app.client.llm.llm_client import LLMClient, IntentClassificationResult


class TestMonitoringServiceIntegration:
    """Test monitoring service integration with system components."""
    
    def test_monitoring_service_integration(self):
        """Test that monitoring service is integrated with LLM client."""
        from app.services.monitoring_service import monitoring_service
        
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
        
        # Clear any existing metrics
        monitoring_service.metrics.metrics.clear()
        
        client = MockLLMClient("test-model")
        
        # Perform classification
        client.classify_intent("test query")
        
        # Check that metrics were recorded in monitoring service
        llm_stats = monitoring_service.get_llm_stats()
        assert llm_stats["total_requests"] >= 1
    
    def test_monitoring_metrics_recording(self):
        """Test that monitoring service records LLM metrics."""
        from app.services.monitoring_service import monitoring_service
        
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9,
                    tokens_used=150,
                    cost=0.02
                )
            
            def _health_check_impl(self):
                return True
        
        # Clear any existing metrics
        monitoring_service.metrics.metrics.clear()
        
        client = MockLLMClient("test-model")
        
        # Perform multiple classifications
        client.classify_intent("test query 1")
        client.classify_intent("test query 2")
        
        # Check that metrics were recorded
        llm_stats = monitoring_service.get_llm_stats()
        assert llm_stats["total_requests"] == 2
        assert llm_stats["success_rate"] == 1.0
        assert "tokens" in llm_stats
        assert "cost" in llm_stats
    
    def test_monitoring_health_checks(self):
        """Test that monitoring service includes health checks."""
        from app.services.monitoring_service import monitoring_service
        
        # Get system health
        health = monitoring_service.get_system_health()
        
        # Check that health status is available
        assert "status" in health
        assert "checks" in health
        assert "timestamp" in health
        
        # Check that some default checks are registered
        assert len(health["checks"]) > 0
    
    def test_monitoring_metrics_dashboard(self):
        """Test that monitoring service provides metrics dashboard."""
        from app.services.monitoring_service import monitoring_service
        
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
        
        # Clear any existing metrics
        monitoring_service.metrics.metrics.clear()
        
        client = MockLLMClient("test-model")
        
        # Perform classification
        client.classify_intent("test query")
        
        # Get metrics dashboard
        dashboard = monitoring_service.get_metrics_dashboard()
        
        # Check dashboard structure
        assert "llm_stats" in dashboard
        assert "system_health" in dashboard
        assert "metrics" in dashboard
        
        # Check LLM stats
        llm_stats = dashboard["llm_stats"]
        assert "total_requests" in llm_stats
        assert "success_rate" in llm_stats
    
    def test_monitoring_metrics_export(self):
        """Test that monitoring service can export metrics."""
        from app.services.monitoring_service import monitoring_service
        
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
        
        # Clear any existing metrics
        monitoring_service.metrics.metrics.clear()
        
        client = MockLLMClient("test-model")
        
        # Perform classification
        client.classify_intent("test query")
        
        # Test JSON export
        json_export = monitoring_service.export_metrics("json")
        assert "llm_request_duration" in json_export
        
        # Test Prometheus export
        prometheus_export = monitoring_service.export_metrics("prometheus")
        assert "llm_request_duration" in prometheus_export
        assert "=" in prometheus_export  # Prometheus format


if __name__ == "__main__":
    pytest.main([__file__]) 
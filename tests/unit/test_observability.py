"""
Unit Tests for Observability Components
Tests request context, structured logging, and monitoring service functionality.
"""

import pytest
import json
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from app.utils.request_context import (
    RequestContext, generate_request_id, get_current_request_id,
    get_request_metadata, add_request_metadata, request_context_manager,
    log_request_start, log_request_end, log_request_error, get_request_duration,
    create_request_logger
)
from app.utils.structured_logger import (
    StructuredFormatter, KeywordHasher, StructuredLogger,
    setup_structured_logging, create_structured_logger
)
from app.services.monitoring_service import (
    MonitoringService, MetricsCollector, HealthChecker
)


class TestRequestContext:
    """Test request context management."""
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        request_id = generate_request_id()
        assert isinstance(request_id, str)
        assert len(request_id) > 0
        
        # Should be unique
        request_id2 = generate_request_id()
        assert request_id != request_id2
    
    def test_request_context_thread_local(self):
        """Test that request context is thread-local."""
        context = RequestContext()
        
        # Set values in main thread
        context.request_id = "main-thread-id"
        context.start_time = datetime.utcnow()
        context.metadata = {"key": "value"}
        
        # Verify values are set
        assert context.request_id == "main-thread-id"
        assert context.start_time is not None
        assert context.metadata == {"key": "value"}
        
        # Test in different thread
        def thread_function():
            # Should not see main thread values
            assert context.request_id is None
            assert context.start_time is None
            assert context.metadata == {}
            
            # Set thread-specific values
            context.request_id = "thread-id"
            context.start_time = datetime.utcnow()
            context.metadata = {"thread_key": "thread_value"}
            
            # Verify thread-specific values
            assert context.request_id == "thread-id"
            assert context.start_time is not None
            assert context.metadata == {"thread_key": "thread_value"}
        
        thread = threading.Thread(target=thread_function)
        thread.start()
        thread.join()
        
        # Main thread values should be unchanged
        assert context.request_id == "main-thread-id"
        assert context.metadata == {"key": "value"}
    
    def test_request_context_manager(self):
        """Test request context manager."""
        with request_context_manager() as request_id:
            assert request_id is not None
            assert get_current_request_id() == request_id
            assert get_request_metadata() == {}
        
        # Context should be cleared after exit
        assert get_current_request_id() is None
        assert get_request_metadata() == {}
    
    def test_request_context_manager_with_id(self):
        """Test request context manager with provided ID."""
        custom_id = "custom-request-id"
        with request_context_manager(custom_id) as request_id:
            assert request_id == custom_id
            assert get_current_request_id() == custom_id
    
    def test_add_request_metadata(self):
        """Test adding request metadata."""
        with request_context_manager() as request_id:
            add_request_metadata("test_key", "test_value")
            metadata = get_request_metadata()
            assert metadata["test_key"] == "test_value"
    
    def test_get_request_duration(self):
        """Test request duration calculation."""
        with request_context_manager() as request_id:
            time.sleep(0.1)  # Sleep for 100ms
            duration = get_request_duration()
            assert duration is not None
            assert duration >= 0.1  # Should be at least 100ms


class TestStructuredLogger:
    """Test structured logging functionality."""
    
    def test_structured_formatter(self):
        """Test structured formatter."""
        formatter = StructuredFormatter()
        record = Mock()
        record.levelname = "INFO"
        record.name = "test_logger"
        record.getMessage.return_value = "Test message"
        record.module = "test_module"
        record.funcName = "test_function"
        record.lineno = 42
        record.exc_info = None
        
        # Mock request context
        with patch('app.utils.structured_logger.get_current_request_id') as mock_get_id:
            with patch('app.utils.structured_logger.get_request_metadata') as mock_get_metadata:
                mock_get_id.return_value = "test-request-id"
                mock_get_metadata.return_value = {"key": "value"}
                
                formatted = formatter.format(record)
                log_data = json.loads(formatted)
                
                assert log_data["level"] == "INFO"
                assert log_data["logger"] == "test_logger"
                assert log_data["message"] == "Test message"
                assert log_data["request_id"] == "test-request-id"
                assert log_data["request_metadata"] == {"key": "value"}
    
    def test_keyword_hasher(self):
        """Test keyword hashing functionality."""
        keywords = ["test", "keyword", "list"]
        hashed = KeywordHasher.hash_keywords(keywords)
        
        assert len(hashed) == len(keywords)
        assert all(len(h) == 16 for h in hashed)  # SHA256 truncated to 16 chars
        
        # Same keywords should produce same hashes
        hashed2 = KeywordHasher.hash_keywords(keywords)
        assert hashed == hashed2
        
        # Different keywords should produce different hashes
        different_keywords = ["different", "keywords"]
        different_hashed = KeywordHasher.hash_keywords(different_keywords)
        assert hashed != different_hashed
    
    def test_structured_logger_creation(self):
        """Test structured logger creation."""
        logger = create_structured_logger("test_logger")
        assert isinstance(logger, StructuredLogger)
        assert logger.logger.name == "test_logger"
    
    def test_structured_logger_methods(self):
        """Test structured logger methods."""
        logger = create_structured_logger("test_logger")
        
        # Test intent logging
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_intent("test_intent", 0.95)
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[1]['extra']['intent'] == "test_intent"
            assert call_args[1]['extra']['confidence'] == 0.95
        
        # Test keyword logging
        with patch.object(logger.logger, 'info') as mock_info:
            keywords = ["test", "keywords"]
            logger.log_keywords(keywords, "test_operation")
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[1]['extra']['keyword_count'] == 2
            assert len(call_args[1]['extra']['hashed_keywords']) == 2
        
        # Test outcome logging
        with patch.object(logger.logger, 'info') as mock_info:
            logger.log_outcome("test_operation", True, 1.5)
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert call_args[1]['extra']['operation'] == "test_operation"
            assert call_args[1]['extra']['success'] is True
            assert call_args[1]['extra']['duration_ms'] == 1500.0


class TestMetricsCollector:
    """Test metrics collector functionality."""
    
    def test_metrics_collector_creation(self):
        """Test metrics collector creation."""
        collector = MetricsCollector()
        assert collector.metrics == {}
        assert collector.max_points == 1000
    
    def test_metrics_collector_recording(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        # Record a metric
        collector.record("test_metric", 1.0)
        assert "test_metric" in collector.metrics
        assert len(collector.metrics["test_metric"]) == 1
        assert collector.metrics["test_metric"][0].value == 1.0
    
    def test_metrics_collector_get_metric(self):
        """Test getting metrics."""
        collector = MetricsCollector()
        collector.record("test_metric", 1.0)
        collector.record("test_metric", 2.0)
        
        points = collector.get_metric("test_metric")
        assert len(points) == 2
        assert points[0].value == 1.0
        assert points[1].value == 2.0


class TestHealthChecker:
    """Test health checker functionality."""
    
    def test_health_checker_creation(self):
        """Test health checker creation."""
        checker = HealthChecker()
        assert checker.checks == {}
    
    def test_health_checker_registration(self):
        """Test health check registration."""
        checker = HealthChecker()
        
        def test_check():
            return {"status": "healthy", "message": "OK"}
        
        checker.register_check("test", test_check)
        assert "test" in checker.checks
        assert checker.checks["test"] == test_check
    
    def test_health_checker_run_check(self):
        """Test running health checks."""
        checker = HealthChecker()
        
        def test_check():
            return {"status": "healthy", "message": "OK"}
        
        checker.register_check("test", test_check)
        result = checker.run_check("test")
        assert result["status"] == "healthy"
        assert result["message"] == "OK"


class TestMonitoringService:
    """Test monitoring service functionality."""
    
    def test_monitoring_service_creation(self):
        """Test monitoring service creation."""
        service = MonitoringService()
        assert service.metrics is not None
        assert service.health_checker is not None
    
    def test_monitoring_service_record_llm_request(self):
        """Test recording LLM requests in monitoring service."""
        service = MonitoringService()
        service.record_llm_request("test", "test-model", 1.0, 100, True, 0.01)
        
        stats = service.get_llm_stats()
        assert stats["total_requests"] == 1
        assert stats["success_rate"] == 1.0
    
    def test_monitoring_service_get_metrics_dashboard(self):
        """Test getting metrics dashboard."""
        service = MonitoringService()
        dashboard = service.get_metrics_dashboard()
        
        assert "llm_stats" in dashboard
        assert "system_health" in dashboard
        assert "metrics" in dashboard
    
    def test_monitoring_service_health_checks(self):
        """Test health checks in monitoring service."""
        service = MonitoringService()
        health = service.get_system_health()
        
        assert "status" in health
        assert "checks" in health
        assert "timestamp" in health


class TestMonitoringServiceIntegration:
    """Test monitoring service integration."""
    
    def test_monitoring_service_metrics_integration(self):
        """Test metrics integration in monitoring service."""
        service = MonitoringService()
        
        # Record some metrics
        service.record_llm_request("test", "test-model", 1.0, 100, True, 0.01)
        service.record_intent_classification("llm", 0.5, 0.9, True)
        
        # Check that metrics are properly integrated
        dashboard = service.get_metrics_dashboard()
        assert "llm_stats" in dashboard
        assert "metrics" in dashboard
        
        # Check that LLM stats are available
        llm_stats = dashboard["llm_stats"]
        assert "total_requests" in llm_stats
        assert "success_rate" in llm_stats


class TestRequestLoggingIntegration:
    """Test integration between request context and logging."""
    
    def test_log_request_start_end(self):
        """Test logging request start and end."""
        with patch('app.utils.request_context.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with request_context_manager() as request_id:
                log_request_start("test_operation", param1="value1")
                log_request_end("test_operation", True, 1.5, result="success")
                
                # Verify logging calls
                assert mock_logger.info.call_count == 2
                
                # Check start log
                start_call = mock_logger.info.call_args_list[0]
                assert start_call[1]['extra']['operation'] == "test_operation"
                assert start_call[1]['extra']['parameters']['param1'] == "value1"
                
                # Check end log
                end_call = mock_logger.info.call_args_list[1]
                assert end_call[1]['extra']['operation'] == "test_operation"
                assert end_call[1]['extra']['success'] is True
                assert end_call[1]['extra']['duration_ms'] == 1500.0
    
    def test_log_request_error(self):
        """Test logging request errors."""
        with patch('app.utils.request_context.logging.getLogger') as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with request_context_manager() as request_id:
                error = ValueError("Test error")
                log_request_error("test_operation", error, 1.0)
                
                # Verify error logging
                mock_logger.error.assert_called_once()
                error_call = mock_logger.error.call_args
                assert error_call[1]['extra']['operation'] == "test_operation"
                assert error_call[1]['extra']['error_type'] == "ValueError"
                assert error_call[1]['extra']['error_message'] == "Test error" 
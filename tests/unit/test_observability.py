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
    PerformanceMetrics, CacheStats, MonitoringService,
    get_monitoring_service, start_monitoring, stop_monitoring, record_request
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


class TestPerformanceMetrics:
    """Test performance metrics functionality."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation."""
        metrics = PerformanceMetrics("test_operation")
        assert metrics.operation == "test_operation"
        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
    
    def test_performance_metrics_recording(self):
        """Test recording performance metrics."""
        metrics = PerformanceMetrics("test_operation")
        
        # Record successful request
        metrics.record_request(True, 1.0)
        assert metrics.total_requests == 1
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 0
        assert metrics.success_rate == 1.0
        assert metrics.error_rate == 0.0
        assert metrics.avg_duration == 1.0
        
        # Record failed request
        metrics.record_request(False, 0.5)
        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.5
        assert metrics.error_rate == 0.5
        assert metrics.avg_duration == 0.75
    
    def test_performance_metrics_to_dict(self):
        """Test performance metrics serialization."""
        metrics = PerformanceMetrics("test_operation")
        metrics.record_request(True, 1.0)
        metrics.record_request(False, 0.5)
        
        data = metrics.to_dict()
        assert data["operation"] == "test_operation"
        assert data["total_requests"] == 2
        assert data["successful_requests"] == 1
        assert data["failed_requests"] == 1
        assert data["success_rate"] == 0.5
        assert data["error_rate"] == 0.5
        assert "avg_duration_ms" in data
        assert "min_duration_ms" in data
        assert "max_duration_ms" in data


class TestCacheStats:
    """Test cache statistics functionality."""
    
    def test_cache_stats_creation(self):
        """Test cache stats creation."""
        stats = CacheStats()
        assert stats.cache_size == 0
        assert stats.cache_hits == 0
        assert stats.cache_misses == 0
        assert stats.hit_rate == 0.0
    
    def test_cache_stats_hit_rate(self):
        """Test cache hit rate calculation."""
        stats = CacheStats()
        stats.cache_hits = 80
        stats.cache_misses = 20
        
        assert stats.hit_rate == 0.8
        
        # No requests should return 0
        stats.cache_hits = 0
        stats.cache_misses = 0
        assert stats.hit_rate == 0.0
    
    def test_cache_stats_to_dict(self):
        """Test cache stats serialization."""
        stats = CacheStats()
        stats.cache_size = 100
        stats.cache_hits = 80
        stats.cache_misses = 20
        
        data = stats.to_dict()
        assert data["cache_size"] == 100
        assert data["cache_hits"] == 80
        assert data["cache_misses"] == 20
        assert data["hit_rate"] == 0.8
        assert "last_updated" in data


class TestMonitoringService:
    """Test monitoring service functionality."""
    
    def test_monitoring_service_creation(self):
        """Test monitoring service creation."""
        with patch('app.services.monitoring_service.create_adapter') as mock_create_adapter:
            mock_adapter = Mock()
            mock_create_adapter.return_value = mock_adapter
            
            service = MonitoringService()
            assert service.health_status == "unknown"
            assert service.last_health_check is None
            assert service.monitoring_thread is None
    
    def test_monitoring_service_record_request(self):
        """Test recording requests in monitoring service."""
        with patch('app.services.monitoring_service.create_adapter') as mock_create_adapter:
            mock_adapter = Mock()
            mock_create_adapter.return_value = mock_adapter
            
            service = MonitoringService()
            service.record_request("test_operation", True, 1.0)
            
            metrics = service.get_performance_metrics("test_operation")
            assert metrics["total_requests"] == 1
            assert metrics["successful_requests"] == 1
            assert metrics["failed_requests"] == 0
    
    def test_monitoring_service_get_summary(self):
        """Test getting monitoring summary."""
        with patch('app.services.monitoring_service.create_adapter') as mock_create_adapter:
            mock_adapter = Mock()
            mock_create_adapter.return_value = mock_adapter
            
            service = MonitoringService()
            summary = service.get_monitoring_summary()
            
            assert "health" in summary
            assert "cache" in summary
            assert "performance" in summary
            assert "monitoring_active" in summary
    
    def test_monitoring_service_start_stop(self):
        """Test starting and stopping monitoring service."""
        with patch('app.services.monitoring_service.create_adapter') as mock_create_adapter:
            mock_adapter = Mock()
            mock_create_adapter.return_value = mock_adapter
            
            service = MonitoringService()
            
            # Start monitoring
            service.start_monitoring(interval_seconds=1)
            assert service.monitoring_thread is not None
            assert service.monitoring_thread.is_alive()
            
            # Stop monitoring
            service.stop_monitoring_service()
            assert not service.monitoring_thread.is_alive()


class TestGlobalMonitoringFunctions:
    """Test global monitoring functions."""
    
    def test_get_monitoring_service(self):
        """Test getting global monitoring service."""
        service = get_monitoring_service()
        assert isinstance(service, MonitoringService)
    
    def test_record_request_function(self):
        """Test global record_request function."""
        with patch('app.services.monitoring_service.monitoring_service') as mock_service:
            record_request("test_operation", True, 1.0)
            mock_service.record_request.assert_called_once_with("test_operation", True, 1.0)
    
    def test_start_stop_monitoring_functions(self):
        """Test start and stop monitoring functions."""
        with patch('app.services.monitoring_service.monitoring_service') as mock_service:
            start_monitoring(interval_seconds=3600)
            mock_service.start_monitoring.assert_called_once_with(3600)
            
            stop_monitoring()
            mock_service.stop_monitoring_service.assert_called_once()


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
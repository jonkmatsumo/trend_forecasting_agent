"""
Unit tests for Security Service
Tests the security and compliance service implementation.
"""

import time
import pytest
from unittest.mock import Mock, patch
from datetime import datetime

from app.services.security_service import (
    SecurityService, DataRedactor, AuditLogger, SecurityEvent,
    LogLevel, DataClassification
)


class TestDataRedactor:
    """Test data redactor functionality."""
    
    def test_data_redactor(self):
        """Test data redaction."""
        redactor = DataRedactor()
        
        # Test text redaction with a longer API key that matches the pattern
        text = "My email is test@example.com and API key is abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        redacted = redactor.redact_text(text, ["email", "api_key"])
        
        assert "[EMAIL_REDACTED]" in redacted
        assert "[API_KEY_REDACTED]" in redacted
        assert "test@example.com" not in redacted
    
    def test_data_redactor_dict(self):
        """Test dictionary redaction."""
        redactor = DataRedactor()
        
        data = {
            "user": "test",
            "email": "test@example.com",
            "nested": {
                "api_key": "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
            }
        }
        
        redacted = redactor.redact_dict(data, ["email", "api_key"])
        
        assert redacted["email"] == "[EMAIL_REDACTED]"
        assert redacted["nested"]["api_key"] == "[API_KEY_REDACTED]"
        assert redacted["user"] == "test"  # Should not be redacted
    
    def test_data_redactor_list(self):
        """Test list redaction."""
        redactor = DataRedactor()
        
        # Test redacting individual strings in a list
        data_list = [
            "test@example.com",
            "normal text"
        ]
        
        # Redact the email string
        redacted_email = redactor.redact_text(data_list[0], ["email"])
        assert redacted_email == "[EMAIL_REDACTED]"
        assert redacted_email != "test@example.com"
        
        # Normal text should not be redacted
        redacted_normal = redactor.redact_text(data_list[1], ["email"])
        assert redacted_normal == "normal text"
    
    def test_data_redactor_specific_patterns(self):
        """Test redaction with specific patterns."""
        redactor = DataRedactor()
        
        text = "Email: test@example.com, Phone: 123-456-7890, SSN: 123-45-6789"
        
        # Test email only
        email_redacted = redactor.redact_text(text, ["email"])
        assert "[EMAIL_REDACTED]" in email_redacted
        assert "123-456-7890" in email_redacted  # Should not be redacted
        assert "123-45-6789" in email_redacted  # Should not be redacted
        
        # Test phone only
        phone_redacted = redactor.redact_text(text, ["phone"])
        assert "test@example.com" in phone_redacted  # Should not be redacted
        assert "[PHONE_REDACTED]" in phone_redacted
        assert "123-45-6789" in phone_redacted  # Should not be redacted
    
    def test_data_redactor_pattern_matching(self):
        """Test various pattern matching scenarios."""
        redactor = DataRedactor()
        
        # Test different email formats
        emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "user+tag@example.org"
        ]
        
        for email in emails:
            redacted = redactor.redact_text(f"Email: {email}", ["email"])
            assert "[EMAIL_REDACTED]" in redacted
            assert email not in redacted
        
        # Test different phone formats
        phones = [
            "123-456-7890",
            "123.456.7890",
            "1234567890"
        ]
        
        for phone in phones:
            redacted = redactor.redact_text(f"Phone: {phone}", ["phone"])
            assert "[PHONE_REDACTED]" in redacted
            assert phone not in redacted
    
    def test_data_hashing(self):
        """Test sensitive data hashing."""
        redactor = DataRedactor()
        
        data = "sensitive data"
        hash1 = redactor.hash_sensitive_data(data)
        hash2 = redactor.hash_sensitive_data(data, "salt")
        
        assert hash1 != hash2
        assert len(hash1) == 64  # SHA256 hex length
        assert len(hash2) == 64  # SHA256 hex length
        
        # Same data should produce same hash
        hash3 = redactor.hash_sensitive_data(data)
        assert hash1 == hash3
    
    def test_data_redactor_no_patterns(self):
        """Test redaction with no patterns specified."""
        redactor = DataRedactor()
        
        text = "Email: test@example.com, API key: abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        redacted = redactor.redact_text(text)  # No patterns specified
        
        # Should redact all patterns by default
        assert "[EMAIL_REDACTED]" in redacted
        assert "[API_KEY_REDACTED]" in redacted
        assert "test@example.com" not in redacted
        assert "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz" not in redacted


class TestAuditLogger:
    """Test audit logger functionality."""
    
    def test_audit_logger(self):
        """Test audit logging."""
        logger = AuditLogger()
        
        # Test access logging
        logger.log_access("user1", "resource", "read", True)
        
        # Test data access logging
        logger.log_data_access(
            "user1", "user_data", DataClassification.CONFIDENTIAL, "api"
        )
        
        # Test security event logging
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="test_event",
            level=LogLevel.INFO,
            user_id="user1"
        )
        logger.log_security_event(event)
    
    def test_audit_logger_access_logging(self):
        """Test access attempt logging."""
        logger = AuditLogger()
        
        # Test successful access
        logger.log_access("user1", "api_endpoint", "GET", True, {"ip": "192.168.1.1"})
        
        # Test failed access
        logger.log_access("user2", "api_endpoint", "POST", False, {"reason": "unauthorized"})
    
    def test_audit_logger_data_access_logging(self):
        """Test data access logging."""
        logger = AuditLogger()
        
        # Test different data classifications
        classifications = [
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
            DataClassification.CONFIDENTIAL,
            DataClassification.RESTRICTED
        ]
        
        for classification in classifications:
            logger.log_data_access(
                "user1",
                "test_data",
                classification,
                "api",
                {"details": "test"}
            )
    
    def test_audit_logger_security_event_logging(self):
        """Test security event logging."""
        logger = AuditLogger()
        
        # Test different log levels
        levels = [LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.CRITICAL]
        
        for level in levels:
            event = SecurityEvent(
                timestamp=datetime.utcnow(),
                event_type="test_event",
                level=level,
                user_id="user1",
                session_id="session123",
                ip_address="192.168.1.1",
                details={"test": "data"}
            )
            logger.log_security_event(event)
    
    def test_audit_logger_file_logging(self):
        """Test audit logger with file output."""
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            logger = AuditLogger(temp_filename)
            
            # Log some events
            logger.log_access("user1", "resource", "read", True)
            logger.log_data_access("user1", "data", DataClassification.INTERNAL, "api")
            
            # Check that file was created and contains logs
            assert os.path.exists(temp_filename)
            with open(temp_filename, 'r') as f:
                content = f.read()
                assert "Access attempt" in content
                assert "Data access" in content
        
        finally:
            # Clean up - handle Windows file locking
            try:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
            except PermissionError:
                # File might still be in use, that's okay for this test
                pass


class TestSecurityService:
    """Test security service integration."""
    
    def test_security_service_integration(self):
        """Test security service integration."""
        service = SecurityService()
        
        # Test log data processing
        data = {"query": "test query", "email": "test@example.com"}
        processed = service.process_log_data(
            data, DataClassification.CONFIDENTIAL, redact_sensitive=True
        )
        
        assert processed["_classification"] == "confidential"
        assert "[EMAIL_REDACTED]" in str(processed)
        
        # Test LLM request logging
        service.log_llm_request(
            user_id="user1",
            query="test query",
            response="test response",
            provider="test",
            model="test-model",
            success=True,
            duration=1.0
        )
    
    def test_security_service_log_data_processing(self):
        """Test log data processing with different classifications."""
        service = SecurityService()
        
        data = {
            "query": "test query",
            "email": "test@example.com",
            "api_key": "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        }
        
        # Test with confidential classification
        confidential_processed = service.process_log_data(
            data, DataClassification.CONFIDENTIAL, redact_sensitive=True
        )
        assert "[EMAIL_REDACTED]" in str(confidential_processed)
        assert "[API_KEY_REDACTED]" in str(confidential_processed)
        
        # Test with public classification (no redaction)
        public_processed = service.process_log_data(
            data, DataClassification.PUBLIC, redact_sensitive=False
        )
        assert "test@example.com" in str(public_processed)
        assert "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz" in str(public_processed)
    
    def test_security_service_llm_request_logging(self):
        """Test LLM request logging."""
        service = SecurityService()
        
        # Test successful request
        service.log_llm_request(
            user_id="user1",
            query="How will bitcoin trend?",
            response="Bitcoin is expected to...",
            provider="openai",
            model="gpt-4",
            success=True,
            duration=1.5
        )
        
        # Test failed request
        service.log_llm_request(
            user_id="user2",
            query="What about ethereum?",
            response="",
            provider="openai",
            model="gpt-4",
            success=False,
            duration=0.1
        )
    
    def test_security_service_intent_classification_logging(self):
        """Test intent classification logging."""
        service = SecurityService()
        
        service.log_intent_classification(
            user_id="user1",
            query="How will bitcoin trend next week?",
            intent="forecast",
            confidence=0.95,
            method="llm"
        )
    
    def test_security_service_data_access_logging(self):
        """Test data access logging."""
        service = SecurityService()
        
        # Test different data types
        data_types = ["user_queries", "model_responses", "training_data", "analytics"]
        
        for data_type in data_types:
            service.log_data_access(
                user_id="user1",
                data_type=data_type,
                access_method="api",
                details={"endpoint": "/api/data"}
            )
    
    def test_security_service_sanitize_for_logging(self):
        """Test data sanitization for logging."""
        service = SecurityService()
        
        # Test dictionary sanitization
        data_dict = {"email": "test@example.com", "query": "sensitive query"}
        sanitized_dict = service.sanitize_for_logging(
            data_dict, DataClassification.CONFIDENTIAL
        )
        assert "[EMAIL_REDACTED]" in str(sanitized_dict)
        
        # Test string sanitization
        data_string = "Email: test@example.com, API key: abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        sanitized_string = service.sanitize_for_logging(
            data_string, DataClassification.CONFIDENTIAL
        )
        assert "[EMAIL_REDACTED]" in sanitized_string
        assert "[API_KEY_REDACTED]" in sanitized_string
        
        # Test non-sensitive data
        normal_data = "This is normal data"
        sanitized_normal = service.sanitize_for_logging(
            normal_data, DataClassification.PUBLIC
        )
        assert sanitized_normal == normal_data
    
    def test_security_service_audit_summary(self):
        """Test audit summary functionality."""
        service = SecurityService()
        
        summary = service.get_audit_summary()
        assert "period" in summary
        assert "total_events" in summary
        assert "events_by_type" in summary
        assert "events_by_level" in summary
        assert "data_access_summary" in summary
        
        # Test with time period
        start_time = datetime.utcnow()
        end_time = datetime.utcnow()
        summary_with_period = service.get_audit_summary(start_time, end_time)
        assert summary_with_period["period"]["start"] is not None
        assert summary_with_period["period"]["end"] is not None
    
    def test_security_service_sensitive_operations(self):
        """Test sensitive operations tracking."""
        service = SecurityService()
        
        # Check that sensitive operations are tracked
        assert 'llm_classification' in service.sensitive_operations
        assert 'user_query_processing' in service.sensitive_operations
        assert 'model_training' in service.sensitive_operations
        assert 'data_export' in service.sensitive_operations


if __name__ == "__main__":
    pytest.main([__file__]) 
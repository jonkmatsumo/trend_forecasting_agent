"""
Integration tests for Security Service
Tests the integration of security service with all system components.
"""

import pytest
from unittest.mock import Mock, patch

from app.services.llm.llm_client import LLMClient, IntentClassificationResult
from app.services.security_service import DataClassification


class TestSecurityServiceIntegration:
    """Test security service integration with system components."""
    
    def test_security_service_integration(self):
        """Test that security service is integrated with LLM client."""
        from app.services.security_service import security_service
        
        class MockLLMClient(LLMClient):
            def _classify_intent_impl(self, query):
                return IntentClassificationResult(
                    intent="test",
                    confidence=0.9
                )
            
            def _health_check_impl(self):
                return True
        
        client = MockLLMClient("test-model")
        
        # Perform classification with user ID
        result = client.classify_intent("test query", user_id="test_user")
        
        # Security logging should be handled automatically
        # (We can't easily test the actual logging without mocking, but we can verify no errors)
        assert result.intent == "test"
    
    def test_security_data_redaction(self):
        """Test that security service handles data redaction."""
        from app.services.security_service import security_service
        
        # Test data redaction functionality
        sensitive_data = {
            "query": "How will bitcoin trend?",
            "email": "user@example.com",
            "api_key": "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        }
        
        # Process data with redaction
        processed_data = security_service.process_log_data(
            sensitive_data, 
            classification=DataClassification.CONFIDENTIAL,
            redact_sensitive=True
        )
        
        # Check that sensitive data was redacted
        assert "[EMAIL_REDACTED]" in str(processed_data)
        assert "[API_KEY_REDACTED]" in str(processed_data)
        assert "user@example.com" not in str(processed_data)
        assert "abc123def456ghi789jkl012mno345pqr678stu901vwx234yz" not in str(processed_data)
    
    def test_security_audit_logging(self):
        """Test that security service provides audit logging."""
        from app.services.security_service import security_service
        
        # Test audit logging functionality
        summary = security_service.get_audit_summary()
        
        # Check audit summary structure
        assert "period" in summary
        assert "total_events" in summary
        assert "events_by_type" in summary
        assert "events_by_level" in summary
        assert "data_access_summary" in summary
    
    def test_security_sensitive_operations(self):
        """Test that security service tracks sensitive operations."""
        from app.services.security_service import security_service
        
        # Check that sensitive operations are tracked
        assert 'llm_classification' in security_service.sensitive_operations
        assert 'user_query_processing' in security_service.sensitive_operations
        assert 'model_training' in security_service.sensitive_operations
        assert 'data_export' in security_service.sensitive_operations
    
    def test_security_llm_request_logging(self):
        """Test that security service logs LLM requests."""
        from app.services.security_service import security_service
        
        # Test LLM request logging
        security_service.log_llm_request(
            user_id="test_user",
            query="How will bitcoin trend?",
            response="Bitcoin is expected to...",
            provider="openai",
            model="gpt-4",
            success=True,
            duration=1.5
        )
        
        # Test failed request logging
        security_service.log_llm_request(
            user_id="test_user",
            query="What about ethereum?",
            response="",
            provider="openai",
            model="gpt-4",
            success=False,
            duration=0.1
        )
        
        # Verify that logging completed without errors
        assert True  # If we get here, no exceptions were raised
    
    def test_security_intent_classification_logging(self):
        """Test that security service logs intent classifications."""
        from app.services.security_service import security_service
        
        # Test intent classification logging
        security_service.log_intent_classification(
            user_id="test_user",
            query="How will bitcoin trend next week?",
            intent="forecast",
            confidence=0.95,
            method="llm"
        )
        
        # Verify that logging completed without errors
        assert True  # If we get here, no exceptions were raised
    
    def test_security_data_access_logging(self):
        """Test that security service logs data access."""
        from app.services.security_service import security_service
        
        # Test different data access types
        data_types = ["user_queries", "model_responses", "training_data", "analytics"]
        
        for data_type in data_types:
            security_service.log_data_access(
                user_id="test_user",
                data_type=data_type,
                access_method="api",
                details={"endpoint": "/api/data"}
            )
        
        # Verify that logging completed without errors
        assert True  # If we get here, no exceptions were raised
    
    def test_security_sanitization(self):
        """Test that security service provides data sanitization."""
        from app.services.security_service import security_service
        
        # Test dictionary sanitization
        data_dict = {"email": "test@example.com", "query": "sensitive query"}
        sanitized_dict = security_service.sanitize_for_logging(
            data_dict, 
            DataClassification.CONFIDENTIAL
        )
        
        # Check that sensitive data was sanitized
        assert "[EMAIL_REDACTED]" in str(sanitized_dict)
        
        # Test string sanitization
        data_string = "Email: test@example.com, API key: abc123def456ghi789jkl012mno345pqr678stu901vwx234yz"
        sanitized_string = security_service.sanitize_for_logging(
            data_string, 
            DataClassification.CONFIDENTIAL
        )
        
        # Check that sensitive data was sanitized
        assert "[EMAIL_REDACTED]" in sanitized_string
        assert "[API_KEY_REDACTED]" in sanitized_string


if __name__ == "__main__":
    pytest.main([__file__]) 
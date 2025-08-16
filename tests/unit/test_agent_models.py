"""
Unit Tests for Agent Models
Tests agent request/response models and validation.
"""

import pytest
from datetime import datetime

from app.models.agent_models import (
    AgentRequest, AgentResponse, AgentError, IntentRecognition,
    AgentIntent, create_agent_response, create_agent_error, create_intent_recognition
)


class TestAgentRequest:
    """Test agent request model."""
    
    def test_valid_request(self):
        """Test valid request creation."""
        request = AgentRequest(
            query="What's the health status?",
            context={"user_type": "admin"},
            user_id="user123",
            session_id="session456"
        )
        
        assert request.query == "What's the health status?"
        assert request.context == {"user_type": "admin"}
        assert request.user_id == "user123"
        assert request.session_id == "session456"
    
    def test_empty_query_validation(self):
        """Test empty query validation."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            AgentRequest(query="")
        
        with pytest.raises(ValueError, match="Query cannot be empty"):
            AgentRequest(query="   ")
    
    def test_query_length_validation(self):
        """Test query length validation."""
        long_query = "x" * 1001
        with pytest.raises(ValueError, match="Query too long"):
            AgentRequest(query=long_query)
    
    def test_to_dict(self):
        """Test request serialization."""
        request = AgentRequest(
            query="Test query",
            context={"key": "value"},
            user_id="user123"
        )
        
        data = request.to_dict()
        assert data["query"] == "Test query"
        assert data["context"] == {"key": "value"}
        assert data["user_id"] == "user123"
        assert data["session_id"] is None


class TestAgentResponse:
    """Test agent response model."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        response = AgentResponse(
            text="Service is healthy",
            data={"status": "healthy"},
            metadata={"intent": "health"},
            request_id="req123"
        )
        
        assert response.text == "Service is healthy"
        assert response.data == {"status": "healthy"}
        assert response.metadata == {"intent": "health"}
        assert response.request_id == "req123"
        assert isinstance(response.timestamp, str)
    
    def test_default_values(self):
        """Test default values."""
        response = AgentResponse(text="Test")
        
        assert response.text == "Test"
        assert response.data is None
        assert response.metadata == {}
        assert response.request_id is None
        assert isinstance(response.timestamp, str)
    
    def test_to_dict(self):
        """Test response serialization."""
        response = AgentResponse(
            text="Test response",
            data={"key": "value"},
            metadata={"meta": "data"},
            request_id="req123"
        )
        
        data = response.to_dict()
        assert data["text"] == "Test response"
        assert data["data"] == {"key": "value"}
        assert data["metadata"] == {"meta": "data"}
        assert data["request_id"] == "req123"
        assert "timestamp" in data


class TestAgentError:
    """Test agent error model."""
    
    def test_valid_error(self):
        """Test valid error creation."""
        error = AgentError(
            error_code="VALIDATION_ERROR",
            message="Invalid input",
            details={"field": "query"},
            request_id="req123"
        )
        
        assert error.error_code == "VALIDATION_ERROR"
        assert error.message == "Invalid input"
        assert error.details == {"field": "query"}
        assert error.request_id == "req123"
        assert isinstance(error.timestamp, str)
    
    def test_default_values(self):
        """Test default values."""
        error = AgentError(
            error_code="ERROR",
            message="Test error"
        )
        
        assert error.error_code == "ERROR"
        assert error.message == "Test error"
        assert error.details is None
        assert error.request_id is None
        assert isinstance(error.timestamp, str)
    
    def test_to_dict(self):
        """Test error serialization."""
        error = AgentError(
            error_code="TEST_ERROR",
            message="Test message",
            details={"detail": "info"},
            request_id="req123"
        )
        
        data = error.to_dict()
        assert data["error_code"] == "TEST_ERROR"
        assert data["message"] == "Test message"
        assert data["details"] == {"detail": "info"}
        assert data["request_id"] == "req123"
        assert "timestamp" in data


class TestIntentRecognition:
    """Test intent recognition model."""
    
    def test_valid_intent_recognition(self):
        """Test valid intent recognition creation."""
        intent = IntentRecognition(
            intent=AgentIntent.HEALTH,
            confidence=0.9,
            slots={"keyword": "health"},
            raw_text="What's the health status?"
        )
        
        assert intent.intent == AgentIntent.HEALTH
        assert intent.confidence == 0.9
        assert intent.slots == {"keyword": "health"}
        assert intent.raw_text == "What's the health status?"
    
    def test_confidence_validation(self):
        """Test confidence score validation."""
        # Valid confidence scores
        IntentRecognition(intent=AgentIntent.HEALTH, confidence=0.0)
        IntentRecognition(intent=AgentIntent.HEALTH, confidence=0.5)
        IntentRecognition(intent=AgentIntent.HEALTH, confidence=1.0)
        
        # Invalid confidence scores
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            IntentRecognition(intent=AgentIntent.HEALTH, confidence=-0.1)
        
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            IntentRecognition(intent=AgentIntent.HEALTH, confidence=1.1)
    
    def test_default_values(self):
        """Test default values."""
        intent = IntentRecognition(
            intent=AgentIntent.FORECAST,
            confidence=0.8
        )
        
        assert intent.intent == AgentIntent.FORECAST
        assert intent.confidence == 0.8
        assert intent.slots == {}
        assert intent.raw_text == ""
    
    def test_to_dict(self):
        """Test intent recognition serialization."""
        intent = IntentRecognition(
            intent=AgentIntent.COMPARE,
            confidence=0.85,
            slots={"keywords": ["python", "javascript"]},
            raw_text="Compare python vs javascript"
        )
        
        data = intent.to_dict()
        assert data["intent"] == "compare"
        assert data["confidence"] == 0.85
        assert data["slots"] == {"keywords": ["python", "javascript"]}
        assert data["raw_text"] == "Compare python vs javascript"


class TestAgentIntent:
    """Test agent intent enum."""
    
    def test_intent_values(self):
        """Test intent enum values."""
        assert AgentIntent.FORECAST.value == "forecast"
        assert AgentIntent.COMPARE.value == "compare"
        assert AgentIntent.SUMMARY.value == "summary"
        assert AgentIntent.TRAIN.value == "train"
        assert AgentIntent.EVALUATE.value == "evaluate"
        assert AgentIntent.HEALTH.value == "health"
        assert AgentIntent.CACHE_STATS.value == "cache_stats"
        assert AgentIntent.CACHE_CLEAR.value == "cache_clear"
        assert AgentIntent.UNKNOWN.value == "unknown"
    
    def test_intent_comparison(self):
        """Test intent comparison."""
        assert AgentIntent.HEALTH == AgentIntent.HEALTH
        assert AgentIntent.FORECAST != AgentIntent.COMPARE


class TestFactoryFunctions:
    """Test factory functions."""
    
    def test_create_agent_response(self):
        """Test create_agent_response factory."""
        response = create_agent_response(
            text="Test response",
            data={"key": "value"},
            metadata={"meta": "data"},
            request_id="req123"
        )
        
        assert response.text == "Test response"
        assert response.data == {"key": "value"}
        assert response.metadata == {"meta": "data"}
        assert response.request_id == "req123"
    
    def test_create_agent_response_defaults(self):
        """Test create_agent_response with defaults."""
        response = create_agent_response("Test")
        
        assert response.text == "Test"
        assert response.data == {}
        assert response.metadata == {}
        assert response.request_id is None
    
    def test_create_agent_error(self):
        """Test create_agent_error factory."""
        error = create_agent_error(
            error_code="TEST_ERROR",
            message="Test message",
            details={"detail": "info"},
            request_id="req123"
        )
        
        assert error.error_code == "TEST_ERROR"
        assert error.message == "Test message"
        assert error.details == {"detail": "info"}
        assert error.request_id == "req123"
    
    def test_create_agent_error_defaults(self):
        """Test create_agent_error with defaults."""
        error = create_agent_error("ERROR", "Message")
        
        assert error.error_code == "ERROR"
        assert error.message == "Message"
        assert error.details is None
        assert error.request_id is None
    
    def test_create_intent_recognition(self):
        """Test create_intent_recognition factory."""
        intent = create_intent_recognition(
            intent=AgentIntent.HEALTH,
            confidence=0.9,
            slots={"keyword": "health"},
            raw_text="Health check"
        )
        
        assert intent.intent == AgentIntent.HEALTH
        assert intent.confidence == 0.9
        assert intent.slots == {"keyword": "health"}
        assert intent.raw_text == "Health check"
    
    def test_create_intent_recognition_defaults(self):
        """Test create_intent_recognition with defaults."""
        intent = create_intent_recognition(AgentIntent.FORECAST, 0.8)
        
        assert intent.intent == AgentIntent.FORECAST
        assert intent.confidence == 0.8
        assert intent.slots == {}
        assert intent.raw_text == "" 
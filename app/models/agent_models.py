"""
Agent Models
Data models for the natural language agent interface.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AgentIntent(str, Enum):
    """Supported agent intents."""
    FORECAST = "forecast"
    COMPARE = "compare"
    SUMMARY = "summary"
    TRAIN = "train"
    EVALUATE = "evaluate"
    HEALTH = "health"
    LIST_MODELS = "list_models"
    UNKNOWN = "unknown"


@dataclass
class AgentRequest:
    """Request model for agent queries."""
    query: str
    context: Optional[Dict[str, Any]] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate the request after initialization."""
        if not self.query or not self.query.strip():
            raise ValueError("Query cannot be empty")
        
        if len(self.query.strip()) > 1000:
            raise ValueError("Query too long (max 1000 characters)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'context': self.context,
            'user_id': self.user_id,
            'session_id': self.session_id
        }


@dataclass
class AgentResponse:
    """Response model for agent queries."""
    text: str
    data: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'text': self.text,
            'data': self.data,
            'metadata': self.metadata,
            'timestamp': self.timestamp,
            'request_id': self.request_id
        }


@dataclass
class IntentRecognition:
    """Result of intent recognition."""
    intent: AgentIntent
    confidence: float
    raw_text: Optional[str] = None
    normalized_text: Optional[str] = None
    normalization_stats: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate the intent recognition after initialization."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'intent': self.intent.value,
            'confidence': self.confidence
        }
        if self.raw_text is not None:
            result['raw_text'] = self.raw_text
        if self.normalized_text is not None:
            result['normalized_text'] = self.normalized_text
        if self.normalization_stats is not None:
            result['normalization_stats'] = self.normalization_stats
        return result


@dataclass
class AgentError:
    """Error model for agent responses."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp,
            'request_id': self.request_id
        }


# Factory functions for consistent response creation
def create_agent_response(
    text: str,
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> AgentResponse:
    """Create an agent response with consistent structure."""
    return AgentResponse(
        text=text,
        data=data or {},
        metadata=metadata or {},
        request_id=request_id
    )


def create_agent_error(
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> AgentError:
    """Create an agent error with consistent structure."""
    return AgentError(
        error_code=error_code,
        message=message,
        details=details,
        request_id=request_id
    )


def create_intent_recognition(
    intent: AgentIntent,
    confidence: float,
    raw_text: str = "",
    normalized_text: Optional[str] = None,
    normalization_stats: Optional[Dict[str, Any]] = None
) -> IntentRecognition:
    """Create an intent recognition result with consistent structure."""
    return IntentRecognition(
        intent=intent,
        confidence=confidence,
        raw_text=raw_text,
        normalized_text=normalized_text,
        normalization_stats=normalization_stats
    ) 
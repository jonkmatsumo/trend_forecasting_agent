"""
LLM Client Abstract Base Class
Defines the interface for LLM-based intent classification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class IntentClassificationResult:
    """Result from LLM intent classification."""
    intent: str
    confidence: float
    rationale: Optional[str] = None
    model_version: Optional[str] = None
    latency_ms: Optional[float] = None


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, timeout_ms: int = 2000, max_tokens: int = 128):
        """Initialize LLM client.
        
        Args:
            model: Model name/identifier
            timeout_ms: Request timeout in milliseconds
            max_tokens: Maximum tokens for response
        """
        self.model = model
        self.timeout_ms = timeout_ms
        self.max_tokens = max_tokens
    
    @abstractmethod
    def classify_intent(self, query: str) -> IntentClassificationResult:
        """Classify query intent.
        
        Args:
            query: User query to classify
            
        Returns:
            IntentClassificationResult with intent, confidence, and rationale
            
        Raises:
            LLMError: If classification fails
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the LLM service is healthy.
        
        Returns:
            True if healthy, False otherwise
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "timeout_ms": self.timeout_ms,
            "max_tokens": self.max_tokens,
            "provider": self.__class__.__name__
        }


class LLMError(Exception):
    """Exception raised when LLM operations fail."""
    
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response_text: Optional[str] = None):
        """Initialize LLM error.
        
        Args:
            message: Error message
            status_code: HTTP status code if applicable
            response_text: Response text if applicable
        """
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text 
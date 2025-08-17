"""
Local Client for Intent Classification
Implementation for local models via HTTP API (vLLM/Ollama).
"""

import json
import time
import logging
from typing import Dict, Any, Optional

import requests

from .llm_client import LLMClient, IntentClassificationResult, LLMError
from .prompt_templates import IntentClassificationPrompt


class LocalClient(LLMClient):
    """Local client for intent classification using HTTP API."""
    
    def __init__(self, base_url: str, model: str, timeout_ms: int = 2000, 
                 max_tokens: int = 128, temperature: float = 0.0):
        """Initialize local client.
        
        Args:
            base_url: Base URL for the local model API
            model: Model name/identifier
            timeout_ms: Request timeout in milliseconds
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0.0 for deterministic)
        """
        super().__init__(model, timeout_ms, max_tokens)
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.prompt_template = IntentClassificationPrompt()
        self.logger = logging.getLogger(__name__)
    
    def _classify_intent_impl(self, query: str) -> IntentClassificationResult:
        """Implementation of intent classification using local model.
        
        Args:
            query: User query to classify
            
        Returns:
            IntentClassificationResult with intent, confidence, and rationale
            
        Raises:
            LLMError: If classification fails
        """
        start_time = time.time()
        
        try:
            # Build prompt
            messages = self.prompt_template.build_prompt(query)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stream": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.timeout_ms / 1000.0,  # Convert to seconds
                headers={"Content-Type": "application/json"}
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            latency_ms = (time.time() - start_time) * 1000
            
            # Parse JSON response
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON response: {content}")
                raise LLMError(f"Invalid JSON response: {e}")
            
            # Validate and extract fields
            intent = result.get("intent", "unknown")
            confidence = float(result.get("confidence", 0.0))
            rationale = result.get("rationale", "")
            
            # Validate intent
            if not self.prompt_template.validate_intent(intent):
                self.logger.warning(f"Invalid intent returned: {intent}")
                intent = "unknown"
                confidence = 0.0
            
            # Clamp confidence
            confidence = max(0.0, min(1.0, confidence))
            
            # Calculate tokens used (if available)
            tokens_used = 0
            if "usage" in response_data:
                tokens_used = response_data["usage"].get("total_tokens", 0)
            
            # Local models typically have no cost
            cost = 0.0
            
            return IntentClassificationResult(
                intent=intent,
                confidence=confidence,
                rationale=rationale,
                model_version=self.model,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                cost=cost
            )
            
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Local model request timeout: {e}")
            raise LLMError(f"Request timeout: {e}")
            
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if hasattr(e, 'response') else None
            self.logger.error(f"Local model HTTP error: {e}")
            raise LLMError(f"HTTP error: {e}", status_code=status_code)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Local model request error: {e}")
            raise LLMError(f"Request error: {e}")
            
        except Exception as e:
            self.logger.error(f"Unexpected error in local model classification: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def _health_check_impl(self) -> bool:
        """Implementation of health check for local model service.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Try to get model info or make a simple request
            response = requests.get(
                f"{self.base_url}/v1/models",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"Local model health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "base_url": self.base_url,
            "temperature": self.temperature,
            "provider": "local"
        })
        return info 
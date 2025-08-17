"""
OpenAI Client for Intent Classification
Implementation using OpenAI Chat Completions API.
"""

import json
import time
import logging
from typing import Dict, Any, Optional

import openai
from openai import OpenAI

from .llm_client import LLMClient, IntentClassificationResult, LLMError
from .prompt_templates import IntentClassificationPrompt


class OpenAIClient(LLMClient):
    """OpenAI client for intent classification."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 timeout_ms: int = 2000, max_tokens: int = 128,
                 temperature: float = 0.0):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            timeout_ms: Request timeout in milliseconds
            max_tokens: Maximum tokens for response
            temperature: Sampling temperature (0.0 for deterministic)
        """
        super().__init__(model, timeout_ms, max_tokens)
        self.api_key = api_key
        self.temperature = temperature
        self.client = OpenAI(api_key=api_key)
        self.prompt_template = IntentClassificationPrompt()
        self.logger = logging.getLogger(__name__)
    
    def _classify_intent_impl(self, query: str) -> IntentClassificationResult:
        """Implementation of intent classification using OpenAI.
        
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
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=self.timeout_ms / 1000.0  # Convert to seconds
            )
            
            # Parse response
            content = response.choices[0].message.content
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
            
            # Calculate tokens used (approximate)
            tokens_used = 0
            if hasattr(response, 'usage') and response.usage:
                tokens_used = getattr(response.usage, 'total_tokens', 0)
            
            # Calculate cost (approximate)
            cost = 0.0
            if hasattr(response, 'usage') and response.usage:
                try:
                    # Rough cost estimation for GPT-4o-mini
                    input_cost_per_1k = 0.00015
                    output_cost_per_1k = 0.0006
                    input_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    output_tokens = getattr(response.usage, 'completion_tokens', 0)
                    cost = (input_tokens * input_cost_per_1k / 1000) + (output_tokens * output_cost_per_1k / 1000)
                except (TypeError, AttributeError):
                    # Handle cases where usage attributes are not numeric (e.g., mocks)
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
            
        except openai.RateLimitError as e:
            self.logger.error(f"OpenAI rate limit exceeded: {e}")
            raise LLMError(f"Rate limit exceeded: {e}", status_code=429)
            
        except openai.APITimeoutError as e:
            self.logger.error(f"OpenAI request timeout: {e}")
            raise LLMError(f"Request timeout: {e}")
            
        except openai.APIError as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise LLMError(f"API error: {e}", status_code=getattr(e, 'status_code', None))
            
        except Exception as e:
            self.logger.error(f"Unexpected error in OpenAI classification: {e}")
            raise LLMError(f"Unexpected error: {e}")
    
    def _health_check_impl(self) -> bool:
        """Implementation of health check for OpenAI service.
        
        Returns:
            True if healthy, False otherwise
        """
        try:
            # Simple health check using a minimal request
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=5,
                timeout=5.0
            )
            return True
        except Exception as e:
            self.logger.error(f"OpenAI health check failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model.
        
        Returns:
            Dictionary with model information
        """
        info = super().get_model_info()
        info.update({
            "temperature": self.temperature,
            "provider": "openai"
        })
        return info 
"""
Unit tests for LLM integration components.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import json

from app.client.llm.llm_client import LLMClient, IntentClassificationResult, LLMError
from app.client.llm.openai_client import OpenAIClient
from app.client.llm.local_client import LocalClient
from app.client.llm.prompt_templates import IntentClassificationPrompt
from app.client.llm.intent_cache import IntentCache, hash_query
from app.services.agent.intent_recognizer import IntentRecognizer
from app.models.agent_models import AgentIntent


class TestIntentClassificationPrompt:
    """Test the prompt template system."""
    
    def test_prompt_creation(self):
        """Test that prompts are created correctly."""
        prompt = IntentClassificationPrompt()
        messages = prompt.build_prompt("How will bitcoin trend next week?")
        
        assert len(messages) > 0
        assert messages[0]["role"] == "system"
        assert "intent classification" in messages[0]["content"].lower()
        
        # Check that few-shot examples are included
        user_messages = [m for m in messages if m["role"] == "user"]
        assert len(user_messages) > 1  # At least one example + the actual query
    
    def test_intent_validation(self):
        """Test intent validation."""
        prompt = IntentClassificationPrompt()
        
        assert prompt.validate_intent("forecast")
        assert prompt.validate_intent("compare")
        assert prompt.validate_intent("unknown")
        assert not prompt.validate_intent("invalid_intent")
    
    def test_allowed_intents(self):
        """Test that all required intents are allowed."""
        prompt = IntentClassificationPrompt()
        allowed = prompt.get_allowed_intents()
        
        required_intents = ["forecast", "compare", "summary", "train", 
                           "evaluate", "health", "list_models", "unknown"]
        
        for intent in required_intents:
            assert intent in allowed


class TestIntentCache:
    """Test the caching system."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations."""
        cache = IntentCache(max_size=10, ttl_hours=1)
        
        # Test set and get
        cache.set("test_hash", {"intent": "forecast", "confidence": 0.9})
        result = cache.get("test_hash")
        
        assert result is not None
        assert result["intent"] == "forecast"
        assert result["confidence"] == 0.9
    
    def test_cache_size_limit(self):
        """Test that cache respects size limits."""
        cache = IntentCache(max_size=2, ttl_hours=1)
        
        cache.set("hash1", {"data": "1"})
        cache.set("hash2", {"data": "2"})
        cache.set("hash3", {"data": "3"})  # Should evict hash1
        
        assert cache.get("hash1") is None
        assert cache.get("hash2") is not None
        assert cache.get("hash3") is not None
    
    def test_query_hashing(self):
        """Test query hashing function."""
        hash1 = hash_query("How will bitcoin trend?")
        hash2 = hash_query("How will bitcoin trend?")  # Same query
        hash3 = hash_query("How will ethereum trend?")  # Different query
        
        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 16  # SHA256 truncated to 16 chars


class TestOpenAIClient:
    """Test OpenAI client implementation."""
    
    @patch('app.client.llm.openai_client.OpenAI')
    def test_openai_client_initialization(self, mock_openai):
        """Test OpenAI client initialization."""
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(
            api_key="test_key",
            model="gpt-4o-mini",
            timeout_ms=2000
        )
        
        assert client.model == "gpt-4o-mini"
        assert client.timeout_ms == 2000
        assert client.temperature == 0.0
    
    @patch('app.client.llm.openai_client.OpenAI')
    def test_classify_intent_success(self, mock_openai):
        """Test successful intent classification."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"intent": "forecast", "confidence": 0.9, "rationale": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(api_key="test_key")
        result = client.classify_intent("How will bitcoin trend?")
        
        assert result.intent == "forecast"
        assert result.confidence == 0.9
        assert result.rationale == "test"
    
    @patch('app.client.llm.openai_client.OpenAI')
    def test_classify_intent_invalid_json(self, mock_openai):
        """Test handling of invalid JSON response."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = 'invalid json'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        client = OpenAIClient(api_key="test_key")
        
        with pytest.raises(LLMError):
            client.classify_intent("How will bitcoin trend?")


class TestLocalClient:
    """Test local client implementation."""
    
    @patch('app.client.llm.local_client.requests.post')
    def test_local_client_initialization(self, mock_post):
        """Test local client initialization."""
        client = LocalClient(
            base_url="http://localhost:8000",
            model="llama-3.1-8b",
            timeout_ms=2000
        )
        
        assert client.base_url == "http://localhost:8000"
        assert client.model == "llama-3.1-8b"
        assert client.timeout_ms == 2000
    
    @patch('app.client.llm.local_client.requests.post')
    def test_classify_intent_success(self, mock_post):
        """Test successful intent classification."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"intent": "compare", "confidence": 0.8, "rationale": "test"}'}}]
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response
        
        client = LocalClient(base_url="http://localhost:8000", model="test-model")
        result = client.classify_intent("Compare bitcoin vs ethereum")
        
        assert result.intent == "compare"
        assert result.confidence == 0.8
        assert result.rationale == "test"


class TestIntentRecognizerIntegration:
    """Test LLM integration with IntentRecognizer."""
    
    @patch('app.services.agent.intent_recognizer.Config')
    def test_llm_disabled(self, mock_config):
        """Test that LLM scorer returns zeros when disabled."""
        mock_config.INTENT_LLM_ENABLED = False
        
        recognizer = IntentRecognizer()
        result = recognizer._llm_scorer("How will bitcoin trend?")
        
        assert result.scorer_type.value == "llm"
        assert result.confidence == 0.0
        assert all(score == 0.0 for score in result.scores.values())
    
    @patch('app.services.agent.intent_recognizer.Config')
    @patch('app.client.llm.openai_client.OpenAI')
    def test_llm_enabled_with_openai(self, mock_openai, mock_config):
        """Test LLM scorer when enabled with OpenAI."""
        mock_config.INTENT_LLM_ENABLED = True
        mock_config.INTENT_LLM_PROVIDER = "openai"
        mock_config.INTENT_LLM_API_KEY = "test_key"
        mock_config.INTENT_LLM_MODEL = "gpt-4o-mini"
        mock_config.INTENT_LLM_TIMEOUT_MS = 2000
        mock_config.INTENT_LLM_MAX_TOKENS = 128
        mock_config.INTENT_LLM_TEMPERATURE = 0.0
        mock_config.INTENT_LLM_CACHE_SIZE = 1000
        mock_config.INTENT_LLM_CACHE_TTL_HOURS = 24
        
        # Mock OpenAI response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = '{"intent": "forecast", "confidence": 0.9, "rationale": "test"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        recognizer = IntentRecognizer()
        result = recognizer._llm_scorer("How will bitcoin trend?")
        
        assert result.scorer_type.value == "llm"
        assert result.confidence == 0.9
        assert result.scores[AgentIntent.FORECAST] == 0.9


class TestWeightRedistribution:
    """Test weight redistribution when scorers fail."""
    
    def test_weight_redistribution_when_llm_fails(self):
        """Test that weights are redistributed when LLM scorer fails."""
        from app.services.agent.intent_recognizer import IntentRecognizer, ScorerResult, ScorerType
        from app.models.agent_models import AgentIntent
        
        # Create recognizer
        recognizer = IntentRecognizer()
        
        # Create mock scorer results
        semantic_result = ScorerResult(
            scorer_type=ScorerType.SEMANTIC,
            scores={AgentIntent.FORECAST: 0.8, AgentIntent.COMPARE: 0.2},
            confidence=0.8
        )
        
        regex_result = ScorerResult(
            scorer_type=ScorerType.REGEX,
            scores={AgentIntent.FORECAST: 0.6, AgentIntent.COMPARE: 0.4},
            confidence=0.6
        )
        
        llm_result = ScorerResult(
            scorer_type=ScorerType.LLM,
            scores={AgentIntent.FORECAST: 0.0, AgentIntent.COMPARE: 0.0},
            confidence=0.0,
            valid=False  # LLM failed
        )
        
        # Test ensemble with failed LLM
        intent, confidence = recognizer._ensemble_scores([semantic_result, regex_result, llm_result])
        
        # Should still work and pick the best intent
        assert intent == AgentIntent.FORECAST
        assert confidence > 0.0
        
        # Verify that weights were redistributed (semantic and regex should have higher weights)
        # The exact values depend on the configuration, but they should be higher than original
        original_semantic_weight = recognizer.ensemble_weights[ScorerType.SEMANTIC]
        original_regex_weight = recognizer.ensemble_weights[ScorerType.REGEX]
        original_llm_weight = recognizer.ensemble_weights[ScorerType.LLM]
        
        # Calculate expected redistributed weights
        total_failed_weight = original_llm_weight
        total_working_weight = original_semantic_weight + original_regex_weight
        redistribution_factor = total_failed_weight / total_working_weight
        
        expected_semantic_weight = original_semantic_weight * (1.0 + redistribution_factor)
        expected_regex_weight = original_regex_weight * (1.0 + redistribution_factor)
        
        # The final confidence should reflect the redistributed weights
        expected_forecast_score = (
            expected_semantic_weight * 0.8 +  # semantic score for forecast
            expected_regex_weight * 0.6       # regex score for forecast
        )
        
        # Allow some tolerance for floating point arithmetic
        assert abs(confidence - expected_forecast_score) < 0.01
    
    def test_weight_redistribution_when_multiple_scorers_fail(self):
        """Test weight redistribution when multiple scorers fail."""
        from app.services.agent.intent_recognizer import IntentRecognizer, ScorerResult, ScorerType
        from app.models.agent_models import AgentIntent
        
        # Create recognizer
        recognizer = IntentRecognizer()
        
        # Create mock scorer results where only semantic works
        semantic_result = ScorerResult(
            scorer_type=ScorerType.SEMANTIC,
            scores={AgentIntent.FORECAST: 0.9, AgentIntent.COMPARE: 0.1},
            confidence=0.9
        )
        
        regex_result = ScorerResult(
            scorer_type=ScorerType.REGEX,
            scores={AgentIntent.FORECAST: 0.0, AgentIntent.COMPARE: 0.0},
            confidence=0.0,
            valid=False  # Regex failed
        )
        
        llm_result = ScorerResult(
            scorer_type=ScorerType.LLM,
            scores={AgentIntent.FORECAST: 0.0, AgentIntent.COMPARE: 0.0},
            confidence=0.0,
            valid=False  # LLM failed
        )
        
        # Test ensemble with multiple failures
        intent, confidence = recognizer._ensemble_scores([semantic_result, regex_result, llm_result])
        
        # Should still work and pick the best intent
        assert intent == AgentIntent.FORECAST
        
        # Semantic should get all the weight since it's the only working scorer
        original_semantic_weight = recognizer.ensemble_weights[ScorerType.SEMANTIC]
        original_regex_weight = recognizer.ensemble_weights[ScorerType.REGEX]
        original_llm_weight = recognizer.ensemble_weights[ScorerType.LLM]
        
        total_failed_weight = original_regex_weight + original_llm_weight
        expected_semantic_weight = original_semantic_weight + total_failed_weight
        
        expected_forecast_score = expected_semantic_weight * 0.9
        
        # Allow some tolerance for floating point arithmetic
        assert abs(confidence - expected_forecast_score) < 0.01
    
    def test_no_weight_redistribution_when_all_scorers_work(self):
        """Test that weights remain unchanged when all scorers work."""
        from app.services.agent.intent_recognizer import IntentRecognizer, ScorerResult, ScorerType
        from app.models.agent_models import AgentIntent
        
        # Create recognizer
        recognizer = IntentRecognizer()
        
        # Create mock scorer results where all work
        semantic_result = ScorerResult(
            scorer_type=ScorerType.SEMANTIC,
            scores={AgentIntent.FORECAST: 0.8, AgentIntent.COMPARE: 0.2},
            confidence=0.8
        )
        
        regex_result = ScorerResult(
            scorer_type=ScorerType.REGEX,
            scores={AgentIntent.FORECAST: 0.6, AgentIntent.COMPARE: 0.4},
            confidence=0.6
        )
        
        llm_result = ScorerResult(
            scorer_type=ScorerType.LLM,
            scores={AgentIntent.FORECAST: 0.7, AgentIntent.COMPARE: 0.3},
            confidence=0.7
        )
        
        # Test ensemble with all working
        intent, confidence = recognizer._ensemble_scores([semantic_result, regex_result, llm_result])
        
        # Should use original weights
        original_semantic_weight = recognizer.ensemble_weights[ScorerType.SEMANTIC]
        original_regex_weight = recognizer.ensemble_weights[ScorerType.REGEX]
        original_llm_weight = recognizer.ensemble_weights[ScorerType.LLM]
        
        expected_forecast_score = (
            original_semantic_weight * 0.8 +  # semantic score for forecast
            original_regex_weight * 0.6 +     # regex score for forecast
            original_llm_weight * 0.7         # llm score for forecast
        )
        
        # Allow some tolerance for floating point arithmetic
        assert abs(confidence - expected_forecast_score) < 0.01


if __name__ == "__main__":
    pytest.main([__file__]) 
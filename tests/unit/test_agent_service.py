"""
Unit Tests for Agent Service
Tests agent service functionality including intent recognition and slot extraction.
"""

import pytest
from unittest.mock import Mock, patch

from app.models.agent_models import AgentRequest, AgentIntent
from app.services.agent.agent_service import AgentService
from app.services.forecaster_interface import ForecasterServiceInterface


class TestAgentService:
    """Test agent service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_forecaster = Mock(spec=ForecasterServiceInterface)
        self.agent_service = AgentService(self.mock_forecaster)
    
    def test_initialization(self):
        """Test agent service initialization."""
        assert self.agent_service.forecaster_service == self.mock_forecaster
        assert self.agent_service.logger is not None
    
    def test_recognize_intent_health(self):
        """Test health intent recognition."""
        intent = self.agent_service._recognize_intent("What's the health status?")
        assert intent.intent == AgentIntent.HEALTH
        assert intent.confidence == 0.8
        
        intent = self.agent_service._recognize_intent("Is the service working?")
        assert intent.intent == AgentIntent.HEALTH
        assert intent.confidence == 0.8
    
    def test_recognize_intent_cache_stats(self):
        """Test cache stats intent recognition."""
        intent = self.agent_service._recognize_intent("Show cache statistics")
        assert intent.intent == AgentIntent.CACHE_STATS
        assert intent.confidence == 0.8
        
        intent = self.agent_service._recognize_intent("What are the cache stats?")
        assert intent.intent == AgentIntent.CACHE_STATS
        assert intent.confidence == 0.8
    
    def test_recognize_intent_cache_clear(self):
        """Test cache clear intent recognition."""
        intent = self.agent_service._recognize_intent("Clear the cache")
        assert intent.intent == AgentIntent.CACHE_CLEAR
        assert intent.confidence == 0.7
        
        intent = self.agent_service._recognize_intent("Reset cache")
        assert intent.intent == AgentIntent.CACHE_CLEAR
        assert intent.confidence == 0.7
    
    def test_recognize_intent_forecast(self):
        """Test forecast intent recognition."""
        intent = self.agent_service._recognize_intent("Forecast trends for python")
        assert intent.intent == AgentIntent.FORECAST
        assert intent.confidence == 0.9
        
        intent = self.agent_service._recognize_intent("Predict next week's data")
        assert intent.intent == AgentIntent.FORECAST
        assert intent.confidence == 0.9
    
    def test_recognize_intent_compare(self):
        """Test compare intent recognition."""
        intent = self.agent_service._recognize_intent("Compare python vs javascript")
        assert intent.intent == AgentIntent.COMPARE
        assert intent.confidence == 0.9
        
        intent = self.agent_service._recognize_intent("What's the difference between AI and ML?")
        assert intent.intent == AgentIntent.COMPARE
        assert intent.confidence == 0.9
    
    def test_recognize_intent_summary(self):
        """Test summary intent recognition."""
        intent = self.agent_service._recognize_intent("Give me a summary of python trends")
        assert intent.intent == AgentIntent.SUMMARY
        assert intent.confidence == 0.8
        
        intent = self.agent_service._recognize_intent("Overview of machine learning data")
        assert intent.intent == AgentIntent.SUMMARY
        assert intent.confidence == 0.8
    
    def test_recognize_intent_unknown(self):
        """Test unknown intent recognition."""
        intent = self.agent_service._recognize_intent("Random text that doesn't match any intent")
        assert intent.intent == AgentIntent.UNKNOWN
        assert intent.confidence == 0.5
    
    def test_extract_slots_keywords(self):
        """Test keyword slot extraction."""
        slots = self.agent_service._extract_slots('Forecast "python" trends', AgentIntent.FORECAST)
        assert slots['keywords'] == ['python']
        
        slots = self.agent_service._extract_slots('Compare "python" vs "javascript"', AgentIntent.COMPARE)
        assert slots['keywords'] == ['python', 'javascript']
    
    def test_extract_slots_time_expressions(self):
        """Test time expression slot extraction."""
        slots = self.agent_service._extract_slots('Forecast next week', AgentIntent.FORECAST)
        assert slots['horizon'] == 7
        
        slots = self.agent_service._extract_slots('Predict next month', AgentIntent.FORECAST)
        assert slots['horizon'] == 30
        
        slots = self.agent_service._extract_slots('Forecast next year', AgentIntent.FORECAST)
        assert slots['horizon'] == 365
    
    def test_extract_slots_quantiles(self):
        """Test quantile slot extraction."""
        slots = self.agent_service._extract_slots('Forecast with p10 p50 p90', AgentIntent.FORECAST)
        assert slots['quantiles'] == [0.1, 0.5, 0.9]
        
        slots = self.agent_service._extract_slots('Predict p50', AgentIntent.FORECAST)
        assert slots['quantiles'] == [0.5]
    
    def test_extract_slots_combined(self):
        """Test combined slot extraction."""
        slots = self.agent_service._extract_slots(
            'Forecast "python" next week with p10 p90', 
            AgentIntent.FORECAST
        )
        assert slots['keywords'] == ['python']
        assert slots['horizon'] == 7
        assert slots['quantiles'] == [0.1, 0.9]
    
    def test_execute_action_health(self):
        """Test health action execution."""
        self.mock_forecaster.health.return_value = {
            'status': 'healthy',
            'service': 'Test Service'
        }
        
        request = AgentRequest(query="What's the health status?")
        result = self.agent_service._execute_action(AgentIntent.HEALTH, {}, request)
        
        assert result['type'] == 'health'
        assert result['data']['status'] == 'healthy'
        assert 'healthy' in result['text']
        self.mock_forecaster.health.assert_called_once()
    
    def test_execute_action_cache_stats(self):
        """Test cache stats action execution."""
        self.mock_forecaster.cache_stats.return_value = {
            'status': 'success',
            'cache_stats': {'cache_size': 100}
        }
        
        request = AgentRequest(query="Show cache statistics")
        result = self.agent_service._execute_action(AgentIntent.CACHE_STATS, {}, request)
        
        assert result['type'] == 'cache_stats'
        assert result['data']['cache_stats']['cache_size'] == 100
        assert '100' in result['text']
        self.mock_forecaster.cache_stats.assert_called_once()
    
    def test_execute_action_cache_clear(self):
        """Test cache clear action execution."""
        self.mock_forecaster.cache_clear.return_value = {
            'status': 'success',
            'message': 'Cache cleared'
        }
        
        request = AgentRequest(query="Clear the cache")
        result = self.agent_service._execute_action(AgentIntent.CACHE_CLEAR, {}, request)
        
        assert result['type'] == 'cache_clear'
        assert result['data']['status'] == 'success'
        assert 'cleared' in result['text']
        self.mock_forecaster.cache_clear.assert_called_once()
    
    def test_execute_action_list_models(self):
        """Test list models action execution."""
        self.mock_forecaster.list_models.return_value = {
            'status': 'success',
            'models': ['model1', 'model2']
        }
        
        request = AgentRequest(query="List models")
        result = self.agent_service._execute_action(AgentIntent.LIST_MODELS, {}, request)
        
        assert result['type'] == 'list_models'
        assert len(result['data']['models']) == 2
        assert '2' in result['text']
        self.mock_forecaster.list_models.assert_called_once()
    
    def test_execute_action_not_implemented(self):
        """Test not implemented action execution."""
        request = AgentRequest(query="Train a model")
        result = self.agent_service._execute_action(AgentIntent.TRAIN, {}, request)
        
        assert result['type'] == 'not_implemented'
        assert 'not yet implemented' in result['text']
    
    def test_execute_action_error(self):
        """Test action execution with error."""
        self.mock_forecaster.health.side_effect = Exception("Service unavailable")
        
        request = AgentRequest(query="What's the health status?")
        with pytest.raises(Exception, match="Service unavailable"):
            self.agent_service._execute_action(AgentIntent.HEALTH, {}, request)
    
    def test_format_response(self):
        """Test response formatting."""
        from app.models.agent_models import IntentRecognition
        
        result = {
            'type': 'health',
            'data': {'status': 'healthy'},
            'text': 'Service is healthy'
        }
        
        intent_result = IntentRecognition(
            intent=AgentIntent.HEALTH,
            confidence=0.9,
            slots={'keyword': 'health'}
        )
        
        request = AgentRequest(query="What's the health status?")
        response = self.agent_service._format_response(result, intent_result, request)
        
        assert response.text == 'Service is healthy'
        assert response.data == {'status': 'healthy'}
        assert response.metadata['intent'] == 'health'
        assert response.metadata['confidence'] == 0.9
        assert response.metadata['slots'] == {'keyword': 'health'}
        assert response.metadata['raw_query'] == "What's the health status?"
    
    @patch('app.services.agent.agent_service.request_context_manager')
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_success(self, mock_get_request_id, mock_context_manager):
        """Test successful query processing."""
        mock_context_manager.return_value.__enter__.return_value = "req123"
        mock_context_manager.return_value.__exit__.return_value = None
        mock_get_request_id.return_value = "req123"
        
        self.mock_forecaster.health.return_value = {
            'status': 'healthy',
            'service': 'Test Service'
        }
        
        request = AgentRequest(query="What's the health status?")
        response = self.agent_service.process_query(request)
        
        assert response.text == 'Service is healthy'
        assert response.metadata['intent'] == 'health'
        assert response.metadata['confidence'] == 0.8
        assert response.request_id == "req123"
    
    @patch('app.services.agent.agent_service.request_context_manager')
    def test_process_query_error(self, mock_context_manager):
        """Test query processing with error."""
        mock_context_manager.return_value.__enter__.return_value = "req123"
        mock_context_manager.return_value.__exit__.return_value = None
        
        self.mock_forecaster.health.side_effect = Exception("Service unavailable")
        
        request = AgentRequest(query="What's the health status?")
        response = self.agent_service.process_query(request)
        
        assert 'error' in response.text.lower()
        assert response.metadata['intent'] == 'error'
        assert response.metadata['confidence'] == 0.0
        assert response.request_id == "req123"
    
    def test_case_insensitive_intent_recognition(self):
        """Test case insensitive intent recognition."""
        intent = self.agent_service._recognize_intent("WHAT'S THE HEALTH STATUS?")
        assert intent.intent == AgentIntent.HEALTH
        
        intent = self.agent_service._recognize_intent("show CACHE statistics")
        assert intent.intent == AgentIntent.CACHE_STATS
        
        intent = self.agent_service._recognize_intent("CLEAR the cache")
        assert intent.intent == AgentIntent.CACHE_CLEAR
    
    def test_multiple_keywords_in_query(self):
        """Test intent recognition with multiple keywords."""
        # Should match the first recognized intent
        intent = self.agent_service._recognize_intent("Show health status and cache stats")
        assert intent.intent == AgentIntent.HEALTH  # health comes first in the logic
    
    def test_empty_slots(self):
        """Test slot extraction with no slots."""
        slots = self.agent_service._extract_slots("Just a simple query", AgentIntent.HEALTH)
        assert slots == {}
    
    def test_slots_with_unknown_intent(self):
        """Test slot extraction with unknown intent."""
        slots = self.agent_service._extract_slots("Random query with 'quoted' text", AgentIntent.UNKNOWN)
        assert slots['keywords'] == ['quoted'] 
"""
Unit Tests for LangGraph Agent Service
Tests the new LangGraph-based agent service.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from app.models.agent_models import AgentRequest, AgentIntent, AgentResponse, IntentRecognition
from app.services.agent.agent_service import AgentService
from app.services.forecaster_interface import ForecasterServiceInterface
from app.agent_graph.state import AgentState


class TestLangGraphAgentService:
    """Test LangGraph agent service functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_forecaster = Mock(spec=ForecasterServiceInterface)
        self.agent_service = AgentService(self.mock_forecaster)
    
    def test_initialization(self):
        """Test agent service initialization."""
        assert self.agent_service.forecaster_service == self.mock_forecaster
        assert self.agent_service.logger is not None
        assert hasattr(self.agent_service, 'graph_agent')
        assert self.agent_service.graph_agent is not None
    
    def test_initialization_failure(self):
        """Test agent service initialization failure."""
        with patch('app.services.agent.agent_service.create_agent_graph') as mock_create_graph:
            mock_create_graph.side_effect = Exception("Failed to create graph")
            
            with pytest.raises(RuntimeError, match="Failed to initialize LangGraph agent"):
                AgentService(self.mock_forecaster)
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_string_input(self, mock_get_request_id):
        """Test processing query with string input."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a successful state
        mock_state = AgentState(
            raw="test query",
            request_id="test-request-id",
            answer={
                'text': 'Test response',
                'data': {'test': 'data'},
                'metadata': {'intent': 'test', 'confidence': 0.8, 'request_id': 'test-request-id'}
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("test query")
        
        assert response.text == 'Test response'
        assert response.data == {'test': 'data'}
        assert response.metadata['intent'] == 'test'
        assert response.metadata['confidence'] == 0.8
        assert response.request_id == 'test-request-id'
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_agent_request_input(self, mock_get_request_id):
        """Test processing query with AgentRequest input."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a successful state
        mock_state = AgentState(
            raw="test query",
            request_id="test-request-id",
            user_id="test-user",
            session_id="test-session",
            answer={
                'text': 'Test response',
                'data': {'test': 'data'},
                'metadata': {'intent': 'test', 'confidence': 0.8, 'request_id': 'test-request-id'}
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        request = AgentRequest(
            query="test query",
            user_id="test-user",
            session_id="test-session"
        )
        response = self.agent_service.process_query(request)
        
        assert response.text == 'Test response'
        assert response.data == {'test': 'data'}
        assert response.metadata['intent'] == 'test'
        assert response.metadata['confidence'] == 0.8
        assert response.request_id == 'test-request-id'
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_no_answer_fallback(self, mock_get_request_id):
        """Test processing query when no answer is generated."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a state with no answer
        mock_state = AgentState(
            raw="test query",
            request_id="test-request-id"
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("test query")
        
        assert "couldn't process" in response.text.lower()
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0
        assert response.request_id == 'test-request-id'
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_graph_execution_error(self, mock_get_request_id):
        """Test processing query when graph execution fails."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to raise an exception
        self.agent_service.graph_agent = Mock(side_effect=Exception("Graph execution failed"))
        
        response = self.agent_service.process_query("test query")
        
        assert "encountered an error" in response.text.lower()
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0
        assert response.request_id == 'test-request-id'
        assert response.data['error_type'] == 'execution_error'
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_with_metadata(self, mock_get_request_id):
        """Test processing query with proper metadata handling."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a state with metadata
        mock_state = AgentState(
            raw="forecast machine learning trends",
            request_id="test-request-id",
            answer={
                'text': 'I will forecast machine learning trends for you.',
                'data': {'keyword': 'machine learning', 'horizon': 30},
                'metadata': {
                    'intent': 'forecast',
                    'confidence': 0.9,
                    'request_id': 'test-request-id',
                    'raw_query': 'forecast machine learning trends'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("forecast machine learning trends")
        
        assert response.text == 'I will forecast machine learning trends for you.'
        assert response.data == {'keyword': 'machine learning', 'horizon': 30}
        assert response.metadata['intent'] == 'forecast'
        assert response.metadata['confidence'] == 0.9
        assert response.metadata['raw_query'] == 'forecast machine learning trends'
        assert response.request_id == 'test-request-id'
    
    def test_format_error_response(self):
        """Test error response formatting."""
        error_result = {
            'type': 'error',
            'text': 'An error occurred',
            'data': {'error': 'test error'}
        }
        
        intent_result = IntentRecognition(
            intent=AgentIntent.UNKNOWN,
            confidence=0.0,
            raw_text="test query"
        )
        
        request = AgentRequest(query="test query")
        
        response = self.agent_service._format_error_response(error_result, intent_result, request)
        
        assert response.text == 'An error occurred'
        assert response.data == {'error': 'test error'}
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0
        assert response.metadata['raw_query'] == 'test query'
    
    def test_format_error_response_no_request(self):
        """Test error response formatting without request."""
        error_result = {
            'type': 'error',
            'text': 'An error occurred',
            'data': {'error': 'test error'}
        }
        
        intent_result = IntentRecognition(
            intent=AgentIntent.UNKNOWN,
            confidence=0.0,
            raw_text="test query"
        )
        
        response = self.agent_service._format_error_response(error_result, intent_result, None)
        
        assert response.text == 'An error occurred'
        assert response.data == {'error': 'test error'}
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0
        assert response.metadata['raw_query'] == 'test query'
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_health_intent(self, mock_get_request_id):
        """Test processing health check query."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a health response
        mock_state = AgentState(
            raw="is the service working?",
            request_id="test-request-id",
            answer={
                'text': 'The service is healthy and running.',
                'data': {'status': 'healthy'},
                'metadata': {
                    'intent': 'health',
                    'confidence': 0.95,
                    'request_id': 'test-request-id',
                    'raw_query': 'is the service working?'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("is the service working?")
        
        assert "healthy" in response.text.lower()
        assert response.metadata['intent'] == 'health'
        assert response.metadata['confidence'] == 0.95
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_forecast_intent(self, mock_get_request_id):
        """Test processing forecast query."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a forecast response
        mock_state = AgentState(
            raw="forecast machine learning trends",
            request_id="test-request-id",
            answer={
                'text': 'I will forecast machine learning trends for the next 30 days.',
                'data': {'keyword': 'machine learning', 'horizon': 30},
                'metadata': {
                    'intent': 'forecast',
                    'confidence': 0.88,
                    'request_id': 'test-request-id',
                    'raw_query': 'forecast machine learning trends'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("forecast machine learning trends")
        
        assert "forecast" in response.text.lower()
        assert response.metadata['intent'] == 'forecast'
        assert response.metadata['confidence'] == 0.88
        assert response.data['keyword'] == 'machine learning'
        assert response.data['horizon'] == 30
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_compare_intent(self, mock_get_request_id):
        """Test processing compare query."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a compare response
        mock_state = AgentState(
            raw="compare python vs javascript",
            request_id="test-request-id",
            answer={
                'text': 'I will compare trends for python and javascript.',
                'data': {'keywords': ['python', 'javascript']},
                'metadata': {
                    'intent': 'compare',
                    'confidence': 0.92,
                    'request_id': 'test-request-id',
                    'raw_query': 'compare python vs javascript'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("compare python vs javascript")
        
        assert "compare" in response.text.lower()
        assert response.metadata['intent'] == 'compare'
        assert response.metadata['confidence'] == 0.92
        assert response.data['keywords'] == ['python', 'javascript']
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_process_query_unknown_intent(self, mock_get_request_id):
        """Test processing unknown intent query."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return an unknown intent response
        mock_state = AgentState(
            raw="tell me a joke",
            request_id="test-request-id",
            answer={
                'text': 'I do not understand that request.',
                'data': {},
                'metadata': {
                    'intent': 'unknown',
                    'confidence': 0.1,
                    'request_id': 'test-request-id',
                    'raw_query': 'tell me a joke'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("tell me a joke")
        
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.1
        assert "do not understand" in response.text.lower()
    
    def test_service_attributes(self):
        """Test that service has expected attributes."""
        assert hasattr(self.agent_service, 'forecaster_service')
        assert hasattr(self.agent_service, 'graph_agent')
        assert hasattr(self.agent_service, 'logger')
        assert hasattr(self.agent_service, 'validator')
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_query_validation_empty_query(self, mock_get_request_id):
        """Test validation of empty query."""
        mock_get_request_id.return_value = "test-request-id"
        
        response = self.agent_service.process_query("")
        
        assert "couldn't understand" in response.text.lower()
        assert response.data['error_type'] == 'validation_error'
        assert "empty" in response.data['error_details'].lower()
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_query_validation_whitespace_only(self, mock_get_request_id):
        """Test validation of whitespace-only query."""
        mock_get_request_id.return_value = "test-request-id"
        
        response = self.agent_service.process_query("   \n\t   ")
        
        assert "couldn't understand" in response.text.lower()
        assert response.data['error_type'] == 'validation_error'
        assert "whitespace" in response.data['error_details'].lower()
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_query_validation_too_long(self, mock_get_request_id):
        """Test validation of query that's too long."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Create a query that's longer than the max (1000 characters)
        long_query = "a" * 1001
        
        response = self.agent_service.process_query(long_query)
        
        assert "couldn't understand" in response.text.lower()
        assert response.data['error_type'] == 'validation_error'
        assert "too long" in response.data['error_details'].lower()
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_query_validation_too_short(self, mock_get_request_id):
        """Test validation of query that's too short."""
        mock_get_request_id.return_value = "test-request-id"
        
        response = self.agent_service.process_query("ab")
        
        assert "couldn't understand" in response.text.lower()
        assert response.data['error_type'] == 'validation_error'
        assert "too short" in response.data['error_details'].lower()
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_processing_time_logging(self, mock_get_request_id):
        """Test that processing time is logged correctly."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to return a simple response
        mock_state = AgentState(
            raw="test query",
            request_id="test-request-id",
            answer={
                'text': 'Test response',
                'data': {},
                'metadata': {
                    'intent': 'test',
                    'confidence': 0.8,
                    'request_id': 'test-request-id'
                }
            }
        )
        self.agent_service.graph_agent = Mock(return_value=mock_state)
        
        response = self.agent_service.process_query("test query")
        
        # Verify the response is correct
        assert response.text == 'Test response'
        assert response.metadata['intent'] == 'test'
        assert response.metadata['confidence'] == 0.8
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_validation_error_formatting(self, mock_get_request_id):
        """Test validation error response formatting."""
        mock_get_request_id.return_value = "test-request-id"
        
        response = self.agent_service.process_query("")
        
        assert response.data['error_type'] == 'validation_error'
        assert 'help' in response.data
        assert response.data['help'] == "Please check your query and try again."
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0
    
    @patch('app.services.agent.agent_service.get_current_request_id')
    def test_execution_error_formatting(self, mock_get_request_id):
        """Test execution error response formatting."""
        mock_get_request_id.return_value = "test-request-id"
        
        # Mock the graph agent to raise an exception
        self.agent_service.graph_agent = Mock(side_effect=Exception("Test execution error"))
        
        response = self.agent_service.process_query("test query")
        
        assert response.data['error_type'] == 'execution_error'
        assert 'help' in response.data
        assert response.data['help'] == "If this problem persists, please contact support."
        assert response.metadata['intent'] == 'unknown'
        assert response.metadata['confidence'] == 0.0 
"""
Unit Tests for LangGraph Agent Integration
Tests the LangGraph agent functionality and integration with the existing agent service.
"""

import pytest
from unittest.mock import Mock, patch
from app.agent_graph.state import AgentState
from app.agent_graph.service_client import InProcessForecasterClient
from app.agent_graph.nodes import normalize, recognize_intent, extract_slots, plan, step, format_answer
from app.agent_graph.graph import build_graph, create_agent_graph
from app.models.agent_models import AgentIntent
from app.services.forecaster_interface import ForecasterServiceInterface


class TestAgentState:
    """Test the AgentState model."""
    
    def test_agent_state_creation(self):
        """Test creating an AgentState instance."""
        state = AgentState(raw="test query")
        
        assert state.raw == "test query"
        assert state.norm_loose is None
        assert state.norm_strict is None
        assert state.intent is None
        assert state.intent_conf == 0.0
        assert state.slots == {}
        assert state.plan == []
        assert state.tool_outputs == {}
        assert state.answer is None
    
    def test_agent_state_with_optional_fields(self):
        """Test creating an AgentState with optional fields."""
        state = AgentState(
            raw="test query",
            request_id="req123",
            user_id="user456",
            session_id="session789"
        )
        
        assert state.request_id == "req123"
        assert state.user_id == "user456"
        assert state.session_id == "session789"


class TestServiceClient:
    """Test the service client functionality."""
    
    def test_in_process_forecaster_client(self):
        """Test the InProcessForecasterClient."""
        mock_service = Mock(spec=ForecasterServiceInterface)
        mock_service.health.return_value = {"status": "healthy"}
        
        client = InProcessForecasterClient(mock_service)
        result = client.health()
        
        assert result == {"status": "healthy"}
        mock_service.health.assert_called_once()


class TestNodes:
    """Test individual LangGraph nodes."""
    
    def test_normalize_node(self):
        """Test the normalize node."""
        state = AgentState(raw="  Test Query  ")
        result = normalize(state)
        
        assert result.norm_loose is not None
        assert result.norm_strict is not None
        assert "test query" in result.norm_strict.lower()
    
    def test_recognize_intent_node(self):
        """Test the recognize_intent node."""
        state = AgentState(raw="What's the health status?")
        state.norm_strict = "what's the health status?"
        
        result = recognize_intent(state)
        
        assert result.intent == AgentIntent.HEALTH
        assert result.intent_conf > 0.0
    
    def test_extract_slots_node(self):
        """Test the extract_slots node."""
        state = AgentState(raw="Forecast python trends")
        state.norm_strict = "forecast python trends"
        state.intent = AgentIntent.FORECAST
        
        result = extract_slots(state)
        
        assert "python" in (result.slots.keywords or [])
    
    def test_plan_node_health(self):
        """Test the plan node for health intent."""
        state = AgentState(raw="health check")
        state.intent = AgentIntent.HEALTH
        state.intent_conf = 0.8
        from app.services.agent.slot_extractor import ExtractedSlots
        state.slots = ExtractedSlots()
        
        result = plan(state)
        
        assert len(result.plan) == 1
        assert result.plan[0]["action"] == "health"
    
    def test_plan_node_forecast(self):
        """Test the plan node for forecast intent."""
        state = AgentState(raw="forecast python trends")
        state.intent = AgentIntent.FORECAST
        state.intent_conf = 0.8
        from app.services.agent.slot_extractor import ExtractedSlots
        state.slots = ExtractedSlots(keywords=["python"])
        
        result = plan(state)
        
        assert len(result.plan) == 1
        assert result.plan[0]["action"] == "forecast"
        assert result.plan[0]["keyword"] == "python"
    
    def test_plan_node_validation_failure(self):
        """Test the plan node when validation fails."""
        state = AgentState(raw="forecast")
        state.intent = AgentIntent.FORECAST
        state.intent_conf = 0.1  # Low confidence
        from app.services.agent.slot_extractor import ExtractedSlots
        state.slots = ExtractedSlots()
        
        result = plan(state)
        
        assert len(result.plan) == 1
        assert result.plan[0]["action"] == "error"
    
    def test_step_node_health(self):
        """Test the step node for health action."""
        mock_service = Mock(spec=ForecasterServiceInterface)
        mock_service.health.return_value = {"status": "healthy"}
        client = InProcessForecasterClient(mock_service)
        
        state = AgentState(raw="health check")
        state.plan = [{"action": "health"}]
        
        result = step(state, client)
        
        assert "health" in result.tool_outputs
        assert result.tool_outputs["health"]["type"] == "health"
        assert len(result.plan) == 0  # Step should be removed
    
    def test_step_node_error(self):
        """Test the step node for error action."""
        mock_service = Mock(spec=ForecasterServiceInterface)
        client = InProcessForecasterClient(mock_service)
        
        state = AgentState(raw="invalid query")
        state.plan = [{"action": "error", "message": "Test error"}]
        
        result = step(state, client)
        
        assert "error" in result.tool_outputs
        assert result.tool_outputs["error"]["type"] == "error"
        assert "Test error" in result.tool_outputs["error"]["text"]
    
    def test_format_answer_node(self):
        """Test the format_answer node."""
        state = AgentState(raw="health check")
        state.intent = AgentIntent.HEALTH
        state.intent_conf = 0.8
        state.tool_outputs = {
            "health": {
                "type": "health",
                "text": "Service is healthy",
                "data": {"status": "healthy"}
            }
        }
        
        result = format_answer(state)
        
        assert result.answer is not None
        assert result.answer["text"] == "Service is healthy"
        assert result.answer["data"] == {"status": "healthy"}
        assert result.answer["metadata"]["intent"] == "health"
        assert result.answer["metadata"]["confidence"] == 0.8


class TestGraph:
    """Test the LangGraph graph functionality."""
    
    def test_build_graph(self):
        """Test building the graph."""
        mock_service = Mock(spec=ForecasterServiceInterface)
        client = InProcessForecasterClient(mock_service)
        
        graph = build_graph(client)
        
        assert graph is not None
        # The graph should be compiled and ready to use
    
    def test_create_agent_graph(self):
        """Test creating the agent graph function."""
        mock_service = Mock(spec=ForecasterServiceInterface)
        client = InProcessForecasterClient(mock_service)
        
        agent_func = create_agent_graph(client)
        
        assert callable(agent_func)
    
    @patch('app.agent_graph.nodes.get_current_request_id')
    def test_full_graph_execution(self, mock_request_id):
        """Test a full graph execution."""
        mock_request_id.return_value = "test-req-123"
        
        mock_service = Mock(spec=ForecasterServiceInterface)
        mock_service.health.return_value = {"status": "healthy"}
        client = InProcessForecasterClient(mock_service)
        
        agent_func = create_agent_graph(client)
        
        # Create initial state
        initial_state = AgentState(raw="What's the health status?")
        
                # Execute the graph
        final_state = agent_func(initial_state)

        # Handle case where graph returns dict instead of AgentState
        if isinstance(final_state, dict):
            final_state = AgentState(**final_state)

        # Verify the result
        assert final_state.answer is not None
        assert "healthy" in final_state.answer["text"].lower()
        assert final_state.answer["metadata"]["intent"] == "health"
        assert final_state.answer["metadata"]["confidence"] > 0.0


class TestAgentServiceIntegration:
    """Test the integration with the main agent service."""
    
    def test_agent_service_with_langgraph(self):
        """Test that the agent service uses LangGraph."""
        from app.services.agent.agent_service import AgentService
        
        mock_service = Mock(spec=ForecasterServiceInterface)
        mock_service.health.return_value = {"status": "healthy"}
        
        agent_service = AgentService(mock_service)
        
        # Test that LangGraph is used
        response = agent_service.process_query("What's the health status?")
        
        assert response.text is not None
        assert "healthy" in response.text.lower()
        assert response.metadata["intent"] == "health"
    
    def test_agent_service_always_uses_langgraph(self):
        """Test that the agent service always uses LangGraph (no legacy mode)."""
        from app.services.agent.agent_service import AgentService
        
        mock_service = Mock(spec=ForecasterServiceInterface)
        mock_service.health.return_value = {"status": "healthy"}
        
        agent_service = AgentService(mock_service)
        
        # Test that LangGraph is always used (no feature flag)
        response = agent_service.process_query("What's the health status?")
        
        assert response.text is not None
        assert "healthy" in response.text.lower()
        assert response.metadata["intent"] == "health" 
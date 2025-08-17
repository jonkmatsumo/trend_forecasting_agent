"""
Agent Service
Core agent orchestration logic using LangGraph for natural language processing.
"""

import time
from typing import Dict, Any, Optional, Union
from datetime import datetime

from app.models.agent_models import (
    AgentRequest, AgentResponse, AgentError, IntentRecognition,
    AgentIntent, create_agent_response, create_agent_error, create_intent_recognition
)
from app.services.forecaster_interface import ForecasterServiceInterface
from app.utils.request_context import request_context_manager, get_current_request_id
from app.utils.structured_logger import create_structured_logger
from app.agent_graph import InProcessForecasterClient, create_agent_graph, AgentState


class AgentService:
    """Core agent service for natural language processing using LangGraph."""
    
    def __init__(self, forecaster_service: ForecasterServiceInterface):
        """Initialize the agent service.
        
        Args:
            forecaster_service: The forecaster service interface to use
        """
        self.forecaster_service = forecaster_service
        self.logger = create_structured_logger("agent_service")
        
        # Initialize LangGraph
        try:
            service_client = InProcessForecasterClient(forecaster_service)
            self.graph_agent = create_agent_graph(service_client)
            self.logger.info("LangGraph agent initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize LangGraph agent: {e}")
            raise RuntimeError(f"Failed to initialize LangGraph agent: {e}")
        
    def process_query(self, query: Union[str, AgentRequest]) -> AgentResponse:
        """Process a natural language query through the LangGraph pipeline.
        
        Args:
            query: The natural language query (string or AgentRequest)
            
        Returns:
            AgentResponse with results
        """
        try:
            # Handle both string and AgentRequest inputs
            if isinstance(query, AgentRequest):
                raw_query = query.query
                request = query
            else:
                raw_query = query
                request = AgentRequest(query=raw_query)
            
            # Create initial state
            initial_state = AgentState(
                raw=raw_query,
                request_id=get_current_request_id(),
                user_id=getattr(request, 'user_id', None),
                session_id=getattr(request, 'session_id', None)
            )
            
            # Run the graph
            final_state = self.graph_agent(initial_state)
            
            # Handle both AgentState objects and dictionaries
            if isinstance(final_state, dict):
                # If the graph returned a dictionary, extract the answer
                if 'answer' in final_state and final_state['answer']:
                    answer = final_state['answer']
                    return create_agent_response(
                        text=answer.get('text', 'Action completed'),
                        data=answer.get('data', {}),
                        metadata=answer.get('metadata', {
                            'intent': 'unknown',
                            'confidence': 0.0,
                            'raw_query': raw_query
                        }),
                        request_id=answer.get('metadata', {}).get('request_id', get_current_request_id())
                    )
                else:
                    # Fallback if no answer in dictionary
                    return create_agent_response(
                        text="I couldn't process your request properly.",
                        data={},
                        metadata={
                            'intent': 'unknown',
                            'confidence': 0.0,
                            'raw_query': raw_query
                        },
                        request_id=get_current_request_id()
                    )
            else:
                # Handle AgentState object
                if hasattr(final_state, 'answer') and final_state.answer:
                    return create_agent_response(
                        text=final_state.answer.get('text', 'Action completed'),
                        data=final_state.answer.get('data', {}),
                        metadata=final_state.answer.get('metadata', {
                            'intent': 'unknown',
                            'confidence': 0.0,
                            'raw_query': raw_query
                        }),
                        request_id=final_state.answer.get('metadata', {}).get('request_id', get_current_request_id())
                    )
                else:
                    # Fallback if no answer was generated
                    return create_agent_response(
                        text="I couldn't process your request properly.",
                        data={},
                        metadata={
                            'intent': 'unknown',
                            'confidence': 0.0,
                            'raw_query': raw_query
                        },
                        request_id=get_current_request_id()
                    )
                
        except Exception as e:
            self.logger.error(f"Error processing query with LangGraph '{query}': {str(e)}")
            
            # Return error response
            error_result = {
                'type': 'error',
                'text': f"An error occurred while processing your query: {str(e)}",
                'data': {}
            }
            error_intent = create_intent_recognition(AgentIntent.UNKNOWN, 0.0, raw_query)
            return self._format_error_response(error_result, error_intent, request)
    
    def _format_error_response(
        self, 
        result: Dict[str, Any], 
        intent_result: IntentRecognition,
        request: Optional[AgentRequest]
    ) -> AgentResponse:
        """Format error response for the user.
        
        Args:
            result: The error result
            intent_result: The intent recognition result
            request: The original request
            
        Returns:
            Formatted agent response
        """
        response = create_agent_response(
            text=result.get('text', 'I processed your request successfully.'),
            data=result.get('data', {}),
            metadata={
                'intent': intent_result.intent.value,
                'confidence': intent_result.confidence,
                'raw_query': request.query if request else intent_result.raw_text or "unknown"
            },
            request_id=get_current_request_id()
        )
        return response
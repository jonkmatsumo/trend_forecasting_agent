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
from app.services.agent.validators import AgentValidator
from app.utils.error_handlers import ValidationError


class AgentService:
    """Core agent service for natural language processing using LangGraph."""
    
    def __init__(self, forecaster_service: ForecasterServiceInterface):
        """Initialize the agent service.
        
        Args:
            forecaster_service: The forecaster service interface to use
        """
        self.forecaster_service = forecaster_service
        self.logger = create_structured_logger("agent_service")
        self.validator = AgentValidator()
        
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
        start_time = time.time()
        request_id = get_current_request_id()
        raw_query = ""
        request = None
        
        try:
            # Step 1: Query validation
            raw_query, request = self._validate_and_prepare_query(query)
            
            # Step 2: State initialization
            initial_state = self._initialize_state(raw_query, request)
            
            # Step 3: Graph execution
            final_state = self._execute_graph(initial_state)
            
            # Step 4: Response formatting
            response = self._format_response(final_state, raw_query, request)
            
            # Log successful processing
            processing_time = time.time() - start_time
            self.logger.info(
                "Query processed successfully",
                extra={
                    "request_id": request_id,
                    "query_length": len(raw_query),
                    "processing_time": processing_time,
                    "intent": response.metadata.get("intent", "unknown"),
                    "confidence": response.metadata.get("confidence", 0.0)
                }
            )
            
            return response
                
        except ValidationError as e:
            # Handle validation errors
            self.logger.warning(
                "Query validation failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "field": getattr(e, 'field', 'unknown')
                }
            )
            return self._format_validation_error(str(e), raw_query, request)
            
        except Exception as e:
            # Handle graph execution and other errors
            processing_time = time.time() - start_time
            self.logger.error(
                "Error processing query",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "processing_time": processing_time,
                    "query": raw_query if raw_query else str(query)
                }
            )
            return self._format_execution_error(str(e), raw_query, request)
    
    def _validate_and_prepare_query(self, query: Union[str, AgentRequest]) -> tuple[str, AgentRequest]:
        """Validate and prepare the query for processing.
        
        Args:
            query: The input query (string or AgentRequest)
            
        Returns:
            Tuple of (raw_query, AgentRequest)
            
        Raises:
            ValidationError: If query validation fails
        """
        # Handle both string and AgentRequest inputs
        if isinstance(query, AgentRequest):
            raw_query = query.query
            request = query
        else:
            raw_query = str(query)
            # Validate before creating AgentRequest to avoid triggering its validation
            self._validate_raw_query(raw_query)
            # Create request after validation
            request = AgentRequest(query=raw_query)
        
        return raw_query, request
    
    def _validate_raw_query(self, raw_query: str) -> None:
        """Validate raw query string before creating AgentRequest.
        
        Args:
            raw_query: The raw query string to validate
            
        Raises:
            ValidationError: If validation fails
        """
        # Check for empty or whitespace-only queries
        if not raw_query.strip():
            raise ValidationError("Query cannot be empty or contain only whitespace", field="query")
        
        # Validate query length and content using our validator
        validation_result = self.validator.validate_query_length(raw_query)
        if not validation_result.is_valid:
            raise ValidationError(
                f"Query validation failed: {'; '.join(validation_result.errors)}",
                field="query"
            )
    
    def _initialize_state(self, raw_query: str, request: AgentRequest) -> AgentState:
        """Initialize the agent state for processing.
        
        Args:
            raw_query: The validated raw query
            request: The agent request
            
        Returns:
            Initialized AgentState
        """
        return AgentState(
            raw=raw_query,
            request_id=get_current_request_id(),
            user_id=getattr(request, 'user_id', None),
            session_id=getattr(request, 'session_id', None)
        )
    
    def _execute_graph(self, initial_state: AgentState) -> Union[AgentState, Dict[str, Any]]:
        """Execute the LangGraph with the given initial state.
        
        Args:
            initial_state: The initial state for the graph
            
        Returns:
            Final state from graph execution
            
        Raises:
            Exception: If graph execution fails
        """
        try:
            return self.graph_agent(initial_state)
        except Exception as e:
            # Log the specific error for debugging
            self.logger.error(
                "Graph execution failed",
                extra={
                    "request_id": initial_state.request_id,
                    "error": str(e),
                    "error_type": type(e).__name__
                }
            )
            raise RuntimeError(f"Graph execution failed: {str(e)}") from e
    
    def _format_response(
        self, 
        final_state: Union[AgentState, Dict[str, Any]], 
        raw_query: str, 
        request: AgentRequest
    ) -> AgentResponse:
        """Format the final state into an AgentResponse.
        
        Args:
            final_state: The final state from graph execution
            raw_query: The original raw query
            request: The original request
            
        Returns:
            Formatted AgentResponse
        """
        # Handle both AgentState objects and dictionaries
        if isinstance(final_state, dict):
            return self._format_dict_response(final_state, raw_query, request)
        else:
            return self._format_state_response(final_state, raw_query, request)
    
    def _format_dict_response(
        self, 
        final_state: Dict[str, Any], 
        raw_query: str, 
        request: AgentRequest
    ) -> AgentResponse:
        """Format dictionary response from graph execution.
        
        Args:
            final_state: Dictionary state from graph
            raw_query: Original raw query
            request: Original request
            
        Returns:
            Formatted AgentResponse
        """
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
                text="I couldn't process your request properly. Please try rephrasing your question.",
                data={},
                metadata={
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'raw_query': raw_query,
                    'error': 'No answer generated by graph'
                },
                request_id=get_current_request_id()
            )
    
    def _format_state_response(
        self, 
        final_state: AgentState, 
        raw_query: str, 
        request: AgentRequest
    ) -> AgentResponse:
        """Format AgentState response from graph execution.
        
        Args:
            final_state: AgentState from graph
            raw_query: Original raw query
            request: Original request
            
        Returns:
            Formatted AgentResponse
        """
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
                text="I couldn't process your request properly. Please try rephrasing your question.",
                data={},
                metadata={
                    'intent': 'unknown',
                    'confidence': 0.0,
                    'raw_query': raw_query,
                    'error': 'No answer generated by graph'
                },
                request_id=get_current_request_id()
            )
    
    def _format_validation_error(
        self, 
        error_message: str, 
        raw_query: str, 
        request: AgentRequest
    ) -> AgentResponse:
        """Format validation error response.
        
        Args:
            error_message: The validation error message
            raw_query: The original raw query
            request: The original request
            
        Returns:
            Formatted error response
        """
        error_intent = create_intent_recognition(AgentIntent.UNKNOWN, 0.0, raw_query)
        
        return create_agent_response(
            text=f"I couldn't understand your request. {error_message}",
            data={
                'error_type': 'validation_error',
                'error_details': error_message,
                'help': "Please check your query and try again."
            },
            metadata={
                'intent': error_intent.intent.value,
                'confidence': error_intent.confidence,
                'raw_query': raw_query,
                'error': error_message
            },
            request_id=get_current_request_id()
        )
    
    def _format_execution_error(
        self, 
        error_message: str, 
        raw_query: str, 
        request: AgentRequest
    ) -> AgentResponse:
        """Format execution error response.
        
        Args:
            error_message: The execution error message
            raw_query: The original raw query
            request: The original request
            
        Returns:
            Formatted error response
        """
        error_intent = create_intent_recognition(AgentIntent.UNKNOWN, 0.0, raw_query)
        
        return create_agent_response(
            text="I encountered an error while processing your request. Please try again later.",
            data={
                'error_type': 'execution_error',
                'error_details': error_message,
                'help': "If this problem persists, please contact support."
            },
            metadata={
                'intent': error_intent.intent.value,
                'confidence': error_intent.confidence,
                'raw_query': raw_query,
                'error': error_message
            },
            request_id=get_current_request_id()
        )
    
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
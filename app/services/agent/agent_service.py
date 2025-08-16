"""
Agent Service
Core agent orchestration logic for natural language processing.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime

from app.models.agent_models import (
    AgentRequest, AgentResponse, AgentError, IntentRecognition,
    AgentIntent, create_agent_response, create_agent_error, create_intent_recognition
)
from app.services.forecaster_interface import ForecasterServiceInterface
from app.utils.request_context import request_context_manager, get_current_request_id
from app.utils.structured_logger import create_structured_logger


class AgentService:
    """Core agent service for natural language processing."""
    
    def __init__(self, forecaster_service: ForecasterServiceInterface):
        """Initialize the agent service.
        
        Args:
            forecaster_service: The forecaster service interface to use
        """
        self.forecaster_service = forecaster_service
        self.logger = create_structured_logger("agent_service")
        
    def process_query(self, request: AgentRequest) -> AgentResponse:
        """Process a natural language query.
        
        Args:
            request: The agent request containing the query
            
        Returns:
            AgentResponse with the processed result
        """
        start_time = time.time()
        
        with request_context_manager() as request_id:
            try:
                # Log the incoming query
                self.logger.log_intent(
                    intent="query_received",
                    confidence=1.0,
                    query_length=len(request.query),
                    user_id=request.user_id,
                    session_id=request.session_id
                )
                
                # Step 1: Recognize intent
                intent_result = self._recognize_intent(request.query)
                
                # Step 2: Extract slots/parameters
                slots = self._extract_slots(request.query, intent_result.intent)
                
                # Step 3: Execute the action
                result = self._execute_action(intent_result.intent, slots, request)
                
                # Step 4: Format the response
                response = self._format_response(result, intent_result, request)
                
                # Log successful processing
                duration = time.time() - start_time
                self.logger.log_outcome(
                    operation="query_processing",
                    success=True,
                    duration=duration,
                    intent=intent_result.intent.value,
                    confidence=intent_result.confidence
                )
                
                return response
                
            except Exception as e:
                # Log error
                duration = time.time() - start_time
                self.logger.logger.error(
                    f"Error processing query: {str(e)}",
                    extra={
                        'request_id': request_id,
                        'query': request.query,
                        'error_type': type(e).__name__,
                        'duration_ms': round(duration * 1000, 2)
                    },
                    exc_info=True
                )
                
                # Return error response
                return create_agent_response(
                    text=f"I'm sorry, I encountered an error while processing your request: {str(e)}",
                    data={'error': str(e)},
                    metadata={
                        'intent': 'error',
                        'confidence': 0.0,
                        'duration_ms': round(duration * 1000, 2)
                    },
                    request_id=request_id
                )
    
    def _recognize_intent(self, query: str) -> IntentRecognition:
        """Recognize the intent from the natural language query.
        
        Args:
            query: The natural language query
            
        Returns:
            IntentRecognition result
        """
        query_lower = query.lower().strip()
        
        # Simple keyword-based intent recognition
        # In a real implementation, this would use a more sophisticated NLP model
        
        if any(word in query_lower for word in ['forecast', 'predict', 'future', 'next']):
            return create_intent_recognition(AgentIntent.FORECAST, 0.9, raw_text=query)
        
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference']):
            return create_intent_recognition(AgentIntent.COMPARE, 0.9, raw_text=query)
        
        elif any(word in query_lower for word in ['summary', 'overview', 'trends', 'data']):
            return create_intent_recognition(AgentIntent.SUMMARY, 0.8, raw_text=query)
        
        elif any(word in query_lower for word in ['train', 'model', 'learn']):
            return create_intent_recognition(AgentIntent.TRAIN, 0.9, raw_text=query)
        
        elif any(word in query_lower for word in ['evaluate', 'performance', 'accuracy']):
            return create_intent_recognition(AgentIntent.EVALUATE, 0.9, raw_text=query)
        
        elif any(word in query_lower for word in ['health', 'status', 'working']):
            return create_intent_recognition(AgentIntent.HEALTH, 0.8, raw_text=query)
        
        elif any(word in query_lower for word in ['clear', 'reset']) and 'cache' in query_lower:
            return create_intent_recognition(AgentIntent.CACHE_CLEAR, 0.7, raw_text=query)
        
        elif any(word in query_lower for word in ['cache', 'stats', 'statistics']):
            return create_intent_recognition(AgentIntent.CACHE_STATS, 0.8, raw_text=query)
        
        else:
            return create_intent_recognition(AgentIntent.UNKNOWN, 0.5, raw_text=query)
    
    def _extract_slots(self, query: str, intent: AgentIntent) -> Dict[str, Any]:
        """Extract slots/parameters from the query.
        
        Args:
            query: The natural language query
            intent: The recognized intent
            
        Returns:
            Dictionary of extracted slots
        """
        slots = {}
        query_lower = query.lower()
        
        # Extract keywords (simple approach - in real implementation would use NER)
        # Look for quoted strings or common keyword patterns
        import re
        
        # Extract quoted keywords
        quoted_keywords = re.findall(r'"([^"]*)"', query)
        if quoted_keywords:
            slots['keywords'] = quoted_keywords
        
        # Extract single quoted keywords as fallback
        single_quoted_keywords = re.findall(r"'([^']*)'", query)
        if single_quoted_keywords and 'keywords' not in slots:
            slots['keywords'] = single_quoted_keywords
        
        # Extract time expressions
        if any(word in query_lower for word in ['next week', 'next month', 'next year']):
            if 'next week' in query_lower:
                slots['horizon'] = 7
            elif 'next month' in query_lower:
                slots['horizon'] = 30
            elif 'next year' in query_lower:
                slots['horizon'] = 365
        
        # Extract quantile expressions
        if 'p10' in query_lower or 'p50' in query_lower or 'p90' in query_lower:
            quantiles = []
            if 'p10' in query_lower:
                quantiles.append(0.1)
            if 'p50' in query_lower:
                quantiles.append(0.5)
            if 'p90' in query_lower:
                quantiles.append(0.9)
            slots['quantiles'] = quantiles
        
        return slots
    
    def _execute_action(
        self, 
        intent: AgentIntent, 
        slots: Dict[str, Any], 
        request: AgentRequest
    ) -> Dict[str, Any]:
        """Execute the action based on the recognized intent.
        
        Args:
            intent: The recognized intent
            slots: Extracted slots/parameters
            request: The original request
            
        Returns:
            Dictionary with the action result
        """
        try:
            if intent == AgentIntent.HEALTH:
                result = self.forecaster_service.health()
                return {
                    'type': 'health',
                    'data': result,
                    'text': f"Service is {result.get('status', 'unknown')}"
                }
            
            elif intent == AgentIntent.CACHE_STATS:
                result = self.forecaster_service.cache_stats()
                return {
                    'type': 'cache_stats',
                    'data': result,
                    'text': f"Cache has {result.get('cache_stats', {}).get('cache_size', 0)} items"
                }
            
            elif intent == AgentIntent.CACHE_CLEAR:
                result = self.forecaster_service.cache_clear()
                return {
                    'type': 'cache_clear',
                    'data': result,
                    'text': "Cache has been cleared successfully"
                }
            
            elif intent == AgentIntent.LIST_MODELS:
                result = self.forecaster_service.list_models()
                models = result.get('models', [])
                return {
                    'type': 'list_models',
                    'data': result,
                    'text': f"Found {len(models)} trained models"
                }
            
            else:
                # For more complex intents, return a placeholder
                return {
                    'type': 'not_implemented',
                    'data': {'intent': intent.value, 'slots': slots},
                    'text': f"I understand you want to {intent.value}, but this feature is not yet implemented."
                }
                
        except Exception as e:
            self.logger.logger.error(f"Error executing action for intent {intent}: {str(e)}")
            raise
    
    def _format_response(
        self, 
        result: Dict[str, Any], 
        intent_result: IntentRecognition,
        request: AgentRequest
    ) -> AgentResponse:
        """Format the response for the user.
        
        Args:
            result: The action result
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
                'slots': intent_result.slots,
                'raw_query': request.query
            },
            request_id=get_current_request_id()
        )
        return response 
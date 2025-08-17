"""
Agent Service
Core agent orchestration logic for natural language processing.
"""

import time
from typing import Dict, Any, Optional, Union
from datetime import datetime

from app.models.agent_models import (
    AgentRequest, AgentResponse, AgentError, IntentRecognition,
    AgentIntent, create_agent_response, create_agent_error, create_intent_recognition
)
from app.services.forecaster_interface import ForecasterServiceInterface
from app.services.agent.intent_recognizer import IntentRecognizer
from app.services.agent.slot_extractor import SlotExtractor, ExtractedSlots
from app.services.agent.validators import AgentValidator, ValidationResult
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
        self.intent_recognizer = IntentRecognizer()
        self.slot_extractor = SlotExtractor()
        self.validator = AgentValidator()
        
    def process_query(self, query: Union[str, AgentRequest]) -> AgentResponse:
        """Process a natural language query through the full pipeline.
        
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
                request = None
            
            # Normalize text for processing
            from app.utils.text_normalizer import normalize_with_ftfy
            normalized_query = normalize_with_ftfy(raw_query)
            
            # Step 1: Intent Recognition
            intent_result = self.intent_recognizer.recognize_intent(
                normalized_query, 
                raw_text=raw_query
            )
            
            # Step 2: Slot Extraction (use normalized text for extraction)
            slots = self.slot_extractor.extract_slots(normalized_query, intent_result.intent)
            
            # Step 3: Validation
            validation_result = self.validator.validate_slots(slots, intent_result.intent, intent_result.confidence)
            
            if not validation_result.is_valid:
                error_result = {
                    'type': 'error',
                    'text': '; '.join(validation_result.errors) if validation_result.errors else 'Validation failed',
                    'data': {'warnings': validation_result.warnings} if validation_result.warnings else {}
                }
                return self._format_response(error_result, intent_result, request)
            
            # Step 4: Action Execution
            action_result = self._execute_action(intent_result.intent, slots, request)
            
            # Step 5: Format Response
            return self._format_response(action_result, intent_result, request)
            
        except Exception as e:
            # Log the error
            self.logger.error(f"Error processing query '{query}': {str(e)}")
            
            # Return error response
            error_result = {
                'type': 'error',
                'text': f"An error occurred while processing your query: {str(e)}",
                'data': {}
            }
            # Create a default intent result for error case
            from app.models.agent_models import create_intent_recognition
            error_intent = create_intent_recognition(AgentIntent.UNKNOWN, 0.0, raw_query)
            # Create a dummy request if none exists
            if request is None:
                request = AgentRequest(query=raw_query)
            return self._format_response(error_result, error_intent, request)
    
    def _recognize_intent(self, query: str) -> IntentRecognition:
        """Recognize intent from natural language query.
        
        Args:
            query: The natural language query
            
        Returns:
            IntentRecognition with intent and confidence
        """
        return self.intent_recognizer.recognize_intent(query)
    
    def _extract_slots(self, query: str, intent: AgentIntent) -> ExtractedSlots:
        """Extract slots from natural language query.
        
        Args:
            query: The natural language query
            intent: The recognized intent
            
        Returns:
            ExtractedSlots with extracted parameters
        """
        slots = self.slot_extractor.extract_slots(query, intent)
        
        # If no keywords found, return empty slots
        if slots.keywords is None:
            slots.keywords = []
            
        return slots

    def _execute_action(
        self, 
        intent: AgentIntent, 
        slots: Union[ExtractedSlots, Dict[str, Any]],
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
            # Handle both ExtractedSlots objects and dictionaries
            if isinstance(slots, dict):
                # Convert dict to ExtractedSlots-like object
                class DictSlots:
                    def __init__(self, data):
                        self.keywords = data.get('keywords')
                        self.horizon = data.get('horizon')
                        self.quantiles = data.get('quantiles')
                        self.date_range = data.get('date_range')
                        self.model_id = data.get('model_id')
                        self.geo = data.get('geo')
                        self.category = data.get('category')
                    
                    def to_dict(self):
                        return {k: v for k, v in self.__dict__.items() if v is not None}
                
                slots = DictSlots(slots)
            
            if intent == AgentIntent.HEALTH:
                result = self.forecaster_service.health()
                return {
                    'type': 'health',
                    'data': result,
                    'text': f"Service is {result.get('status', 'unknown')}"
                }
            
            elif intent == AgentIntent.LIST_MODELS:
                result = self.forecaster_service.list_models()
                models = result.get('models', [])
                return {
                    'type': 'list_models',
                    'data': result,
                    'text': f"Found {len(models)} trained models"
                }
            
            elif intent == AgentIntent.FORECAST:
                if not slots.keywords:
                    return {
                        'type': 'error',
                        'data': {'error': 'No keywords provided for forecasting'},
                        'text': "Please provide keywords to forecast. For example: 'Forecast machine learning trends'"
                    }
                
                # Use first keyword for now (could be enhanced to handle multiple)
                keyword = slots.keywords[0]
                horizon = slots.horizon or 30  # Default to 30 days
                quantiles = slots.quantiles or [0.1, 0.5, 0.9]  # Default quantiles
                
                return {
                    'type': 'forecast',
                    'data': {
                        'keyword': keyword,
                        'horizon': horizon,
                        'quantiles': quantiles
                    },
                    'text': f"I'll forecast trends for '{keyword}' over the next {horizon} days with quantiles {quantiles}."
                }
            
            elif intent == AgentIntent.SUMMARY:
                if not slots.keywords:
                    return {
                        'type': 'error',
                        'data': {'error': 'No keywords provided for summary'},
                        'text': "Please provide keywords to summarize. For example: 'Give me a summary of python programming'"
                    }
                
                keyword = slots.keywords[0]
                date_range = slots.date_range or {}
                
                return {
                    'type': 'summary',
                    'data': {
                        'keyword': keyword,
                        'date_range': date_range
                    },
                    'text': f"I'll provide a summary of '{keyword}' trends."
                }
            
            elif intent == AgentIntent.COMPARE:
                if not slots.keywords or len(slots.keywords) < 2:
                    return {
                        'type': 'error',
                        'data': {'error': 'Need at least 2 keywords for comparison'},
                        'text': "Please provide at least 2 keywords to compare. For example: 'Compare machine learning vs artificial intelligence'"
                    }
                
                return {
                    'type': 'compare',
                    'data': {
                        'keywords': slots.keywords
                    },
                    'text': f"I'll compare trends for: {', '.join(slots.keywords)}"
                }
            
            elif intent == AgentIntent.TRAIN:
                if not slots.keywords:
                    return {
                        'type': 'error',
                        'data': {'error': 'No keywords provided for training'},
                        'text': "Please provide keywords to train on. For example: 'Train a model for machine learning'"
                    }
                
                keyword = slots.keywords[0]
                horizon = slots.horizon or 30
                
                return {
                    'type': 'train',
                    'data': {
                        'keyword': keyword,
                        'horizon': horizon
                    },
                    'text': f"I'll train a forecasting model for '{keyword}' with {horizon}-day horizon."
                }
            
            elif intent == AgentIntent.EVALUATE:
                model_id = slots.model_id
                
                return {
                    'type': 'evaluate',
                    'data': {
                        'model_id': model_id
                    },
                    'text': f"I'll evaluate model performance{f' for model {model_id}' if model_id else ' for all models'}."
                }
            
            else:
                # For unknown intents, return a placeholder
                return {
                    'type': 'not_implemented',
                    'data': {'intent': intent.value, 'slots': slots.to_dict()},
                    'text': f"I understand you want to {intent.value}, but this feature is not yet implemented."
                }
                
        except Exception as e:
            self.logger.logger.error(f"Error executing action for intent {intent}: {str(e)}")
            return {
                'type': 'error',
                'data': {'error': str(e)},
                'text': f"I encountered an error while processing your request: {str(e)}"
            }
    
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
                'raw_query': request.query
            },
            request_id=get_current_request_id()
        )
        return response 
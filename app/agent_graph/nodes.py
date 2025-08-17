"""
LangGraph Nodes
Individual processing nodes for the agent workflow.
"""

from typing import Dict, Any, List
from app.agent_graph.state import AgentState
from app.agent_graph.service_client import ForecasterClient
from app.services.agent.intent_recognizer import IntentRecognizer
from app.services.agent.slot_extractor import SlotExtractor
from app.services.agent.validators import AgentValidator
from app.models.agent_models import AgentIntent
from app.utils.text_normalizer import normalize_with_ftfy, normalize_views
from app.utils.request_context import get_current_request_id


def normalize(state: AgentState) -> AgentState:
    """Normalize the raw input text.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with normalized text
    """
    if not state.raw or not state.raw.strip():
        return state
    
    # Use the existing normalization function
    norm_loose, norm_strict, _ = normalize_views(state.raw)
    
    state.norm_loose = norm_loose
    state.norm_strict = norm_strict
    
    return state


def recognize_intent(state: AgentState) -> AgentState:
    """Recognize intent from the normalized text.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with recognized intent
    """
    if not state.norm_strict:
        return state
    
    # Use the existing intent recognizer
    recognizer = IntentRecognizer()
    intent_result = recognizer.recognize_intent(state.norm_strict, raw_text=state.raw)
    
    state.intent = intent_result.intent
    state.intent_conf = intent_result.confidence
    
    return state


def extract_slots(state: AgentState) -> AgentState:
    """Extract slots from the normalized text.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with extracted slots
    """
    if not state.intent or not state.norm_strict:
        return state
    
    # Use the existing slot extractor
    extractor = SlotExtractor()
    extracted_slots = extractor.extract_slots(state.norm_strict, state.intent)
    
    # Keep as ExtractedSlots object for validator compatibility
    state.slots = extracted_slots
    
    return state


def plan(state: AgentState) -> AgentState:
    """Create an execution plan based on the intent and slots.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with execution plan
    """
    if not state.intent:
        return state
    
    # Validate slots first
    validator = AgentValidator()
    validation_result = validator.validate_slots(state.slots, state.intent, state.intent_conf)
    
    if not validation_result.is_valid:
        # If validation fails, create an error plan
        state.plan = [{
            'action': 'error',
            'message': '; '.join(validation_result.errors) if validation_result.errors else 'Validation failed',
            'warnings': validation_result.warnings or []
        }]
        return state
    
    # Create execution plan based on intent
    if state.intent == AgentIntent.HEALTH:
        state.plan = [{'action': 'health'}]
    
    elif state.intent == AgentIntent.LIST_MODELS:
        state.plan = [{'action': 'list_models'}]
    
    elif state.intent == AgentIntent.FORECAST:
        if state.slots.keywords:
            state.plan = [
                {'action': 'forecast', 'keyword': state.slots.keywords[0]}
            ]
        else:
            state.plan = [{'action': 'error', 'message': 'No keywords provided for forecasting'}]
    
    elif state.intent == AgentIntent.SUMMARY:
        if state.slots.keywords:
            state.plan = [
                {'action': 'trends_summary', 'keywords': state.slots.keywords}
            ]
        else:
            state.plan = [{'action': 'error', 'message': 'No keywords provided for summary'}]
    
    elif state.intent == AgentIntent.COMPARE:
        if state.slots.keywords and len(state.slots.keywords) >= 2:
            state.plan = [
                {'action': 'compare', 'keywords': state.slots.keywords}
            ]
        else:
            state.plan = [{'action': 'error', 'message': 'Need at least 2 keywords for comparison'}]
    
    elif state.intent == AgentIntent.TRAIN:
        if state.slots.keywords:
            state.plan = [
                {'action': 'train', 'keyword': state.slots.keywords[0]}
            ]
        else:
            state.plan = [{'action': 'error', 'message': 'No keywords provided for training'}]
    
    elif state.intent == AgentIntent.EVALUATE:
        model_id = state.slots.model_id
        state.plan = [
            {'action': 'evaluate', 'model_id': model_id}
        ]
    
    else:
        # Unknown intent
        state.plan = [{'action': 'error', 'message': f'Intent {state.intent.value} not yet implemented'}]
    
    return state


def step(state: AgentState, service_client: ForecasterClient) -> AgentState:
    """Execute the next step in the plan.
    
    Args:
        state: Current agent state
        service_client: Client for calling forecaster services
        
    Returns:
        Updated state with step results
    """
    if not state.plan:
        return state
    
    # Get the next step
    current_step = state.plan[0]
    action = current_step.get('action')
    
    try:
        if action == 'health':
            result = service_client.health()
            state.tool_outputs['health'] = {
                'type': 'health',
                'data': result,
                'text': f"Service is {result.get('status', 'unknown')}"
            }
        
        elif action == 'list_models':
            result = service_client.list_models()
            models = result.get('models', [])
            state.tool_outputs['list_models'] = {
                'type': 'list_models',
                'data': result,
                'text': f"Found {len(models)} trained models"
            }
        
        elif action == 'forecast':
            keyword = current_step.get('keyword')
            horizon = getattr(state.slots, 'horizon', 30)
            quantiles = getattr(state.slots, 'quantiles', [0.1, 0.5, 0.9])
            
            # For now, return a placeholder since we need a model_id for prediction
            state.tool_outputs['forecast'] = {
                'type': 'forecast',
                'data': {
                    'keyword': keyword,
                    'horizon': horizon,
                    'quantiles': quantiles
                },
                'text': f"I'll forecast trends for '{keyword}' over the next {horizon} days with quantiles {quantiles}."
            }
        
        elif action == 'trends_summary':
            keywords = current_step.get('keywords', [])
            timeframe = getattr(state.slots, 'timeframe', 'today 12-m')
            geo = getattr(state.slots, 'geo', '')
            
            result = service_client.trends_summary(keywords, timeframe, geo)
            state.tool_outputs['trends_summary'] = {
                'type': 'trends_summary',
                'data': result,
                'text': f"Summary for keywords: {', '.join(keywords)}"
            }
        
        elif action == 'compare':
            keywords = current_step.get('keywords', [])
            timeframe = getattr(state.slots, 'timeframe', 'today 12-m')
            geo = getattr(state.slots, 'geo', '')
            
            result = service_client.compare(keywords, timeframe, geo)
            state.tool_outputs['compare'] = {
                'type': 'compare',
                'data': result,
                'text': f"Comparison for keywords: {', '.join(keywords)}"
            }
        
        elif action == 'train':
            keyword = current_step.get('keyword')
            horizon = getattr(state.slots, 'horizon', 30)
            
            # For now, return a placeholder since we need time series data
            state.tool_outputs['train'] = {
                'type': 'train',
                'data': {
                    'keyword': keyword,
                    'horizon': horizon
                },
                'text': f"I'll train a forecasting model for '{keyword}' with {horizon}-day horizon."
            }
        
        elif action == 'evaluate':
            model_id = current_step.get('model_id')
            
            if model_id:
                result = service_client.evaluate(model_id)
                state.tool_outputs['evaluate'] = {
                    'type': 'evaluate',
                    'data': result,
                    'text': f"Evaluation results for model {model_id}"
                }
            else:
                state.tool_outputs['evaluate'] = {
                    'type': 'evaluate',
                    'data': {},
                    'text': "I'll evaluate model performance for all models."
                }
        
        elif action == 'error':
            state.tool_outputs['error'] = {
                'type': 'error',
                'text': current_step.get('message', 'An error occurred'),
                'data': {'warnings': current_step.get('warnings', [])}
            }
        
        else:
            state.tool_outputs['unknown'] = {
                'type': 'error',
                'text': f"Unknown action: {action}",
                'data': {}
            }
    
    except Exception as e:
        state.tool_outputs['error'] = {
            'type': 'error',
            'text': f"Error executing {action}: {str(e)}",
            'data': {'error': str(e)}
        }
    
    # Remove the completed step
    state.plan.pop(0)
    
    return state


def format_answer(state: AgentState) -> AgentState:
    """Format the final answer from tool outputs.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with formatted answer
    """
    # Get the most recent tool output
    if not state.tool_outputs:
        state.answer = {
            'text': 'No action was executed',
            'data': {},
            'metadata': {
                'intent': state.intent.value if hasattr(state.intent, 'value') else str(state.intent) if state.intent else 'unknown',
                'confidence': state.intent_conf,
                'raw_query': state.raw
            }
        }
        return state
    
    # Get the last tool output
    last_output = list(state.tool_outputs.values())[-1]
    
    state.answer = {
        'text': last_output.get('text', 'Action completed successfully'),
        'data': last_output.get('data', {}),
                    'metadata': {
                'intent': state.intent.value if hasattr(state.intent, 'value') else str(state.intent) if state.intent else 'unknown',
                'confidence': state.intent_conf,
                'raw_query': state.raw,
                'request_id': get_current_request_id()
            }
    }
    
    return state 
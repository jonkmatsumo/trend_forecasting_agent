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
    intent_result = recognizer.recognize_intent(state.norm_strict, raw_text=state.raw, is_normalized=True)
    
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


import time
import logging
import threading
import queue
from typing import Dict, Any, List, Optional
from functools import wraps

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator to retry operations on failure with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}. "
                                     f"Retrying in {current_delay} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}: {str(e)}")
            
            raise last_exception
        return wrapper
    return decorator


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
            result = _execute_with_retry(service_client.health)
            state.tool_outputs['health'] = {
                'type': 'health',
                'data': result,
                'text': f"Service is {result.get('status', 'unknown')}"
            }
        
        elif action == 'list_models':
            # Get filters from slots if available
            keyword_filter = getattr(state.slots, 'keywords', [None])[0] if getattr(state.slots, 'keywords', []) else None
            model_type_filter = getattr(state.slots, 'model_type', None)
            
            result = _execute_with_retry(
                service_client.list_models,
                keyword=keyword_filter,
                model_type=model_type_filter,
                limit=50,
                offset=0
            )
            models = result.get('models', [])
            state.tool_outputs['list_models'] = {
                'type': 'list_models',
                'data': result,
                'text': f"Found {len(models)} trained models"
            }
        
        elif action == 'forecast':
            keyword = current_step.get('keyword')
            horizon = getattr(state.slots, 'horizon', 30)
            
            # First, try to find a model for this keyword
            models_result = _execute_with_retry(
                service_client.list_models,
                keyword=keyword,
                limit=1,
                offset=0
            )
            
            models = models_result.get('models', [])
            if not models:
                # No model found, create an error response
                state.tool_outputs['forecast'] = {
                    'type': 'error',
                    'data': {
                        'keyword': keyword,
                        'error': 'No trained model found for this keyword'
                    },
                    'text': f"No trained model found for '{keyword}'. Please train a model first."
                }
            else:
                # Use the first available model
                model_id = models[0]['model_id']
                result = _execute_with_retry(
                    service_client.predict,
                    model_id=model_id,
                    forecast_horizon=horizon
                )
                state.tool_outputs['forecast'] = {
                    'type': 'forecast',
                    'data': result,
                    'text': f"Forecast generated for '{keyword}' using model {model_id} over {horizon} periods."
                }
        
        elif action == 'trends_summary':
            keywords = current_step.get('keywords', [])
            timeframe = getattr(state.slots, 'timeframe', 'today 12-m')
            geo = getattr(state.slots, 'geo', '')
            
            result = _execute_with_retry(
                service_client.trends_summary,
                keywords=keywords,
                timeframe=timeframe,
                geo=geo
            )
            state.tool_outputs['trends_summary'] = {
                'type': 'trends_summary',
                'data': result,
                'text': f"Summary generated for keywords: {', '.join(keywords)}"
            }
        
        elif action == 'compare':
            keywords = current_step.get('keywords', [])
            timeframe = getattr(state.slots, 'timeframe', 'today 12-m')
            geo = getattr(state.slots, 'geo', '')
            
            result = _execute_with_retry(
                service_client.compare,
                keywords=keywords,
                timeframe=timeframe,
                geo=geo
            )
            state.tool_outputs['compare'] = {
                'type': 'compare',
                'data': result,
                'text': f"Comparison generated for keywords: {', '.join(keywords)}"
            }
        
        elif action == 'train':
            keyword = current_step.get('keyword')
            horizon = getattr(state.slots, 'horizon', 30)
            model_type = getattr(state.slots, 'model_type', 'prophet')  # Default to prophet
            
            # For training, we need time series data. Since we don't have it in the slots,
            # we'll need to get it from the trends service first
            try:
                # Get trends data for the keyword
                trends_result = _execute_with_retry(
                    service_client.trends_summary,
                    keywords=[keyword],
                    timeframe='today 12-m',
                    geo=''
                )
                
                # Extract time series data from trends result
                # This is a simplified approach - in practice, you'd need to parse the trends data
                # and convert it to the format expected by the train method
                time_series_data = trends_result.get('data', {}).get('interest_over_time', [])
                dates = trends_result.get('data', {}).get('dates', [])
                
                if not time_series_data or len(time_series_data) < 52:
                    state.tool_outputs['train'] = {
                        'type': 'error',
                        'data': {
                            'keyword': keyword,
                            'error': 'Insufficient data for training'
                        },
                        'text': f"Insufficient data for training model for '{keyword}'. Need at least 52 data points."
                    }
                else:
                    # Train the model
                    result = _execute_with_retry(
                        service_client.train,
                        keyword=keyword,
                        time_series_data=time_series_data,
                        dates=dates,
                        model_type=model_type,
                        forecast_horizon=horizon
                    )
                    state.tool_outputs['train'] = {
                        'type': 'train',
                        'data': result,
                        'text': f"Model training completed for '{keyword}' with {model_type} model."
                    }
                    
            except Exception as e:
                state.tool_outputs['train'] = {
                    'type': 'error',
                    'data': {
                        'keyword': keyword,
                        'error': str(e)
                    },
                    'text': f"Failed to train model for '{keyword}': {str(e)}"
                }
        
        elif action == 'evaluate':
            model_id = current_step.get('model_id')
            
            if model_id:
                result = _execute_with_retry(
                    service_client.evaluate,
                    model_id=model_id
                )
                state.tool_outputs['evaluate'] = {
                    'type': 'evaluate',
                    'data': result,
                    'text': f"Evaluation completed for model {model_id}"
                }
            else:
                # Evaluate all models
                models_result = _execute_with_retry(
                    service_client.list_models,
                    limit=50,
                    offset=0
                )
                
                models = models_result.get('models', [])
                evaluation_results = []
                
                for model in models[:5]:  # Limit to first 5 models to avoid timeout
                    try:
                        eval_result = _execute_with_retry(
                            service_client.evaluate,
                            model_id=model['model_id']
                        )
                        evaluation_results.append({
                            'model_id': model['model_id'],
                            'evaluation': eval_result
                        })
                    except Exception as e:
                        evaluation_results.append({
                            'model_id': model['model_id'],
                            'error': str(e)
                        })
                
                state.tool_outputs['evaluate'] = {
                    'type': 'evaluate',
                    'data': {
                        'evaluations': evaluation_results,
                        'total_models': len(models)
                    },
                    'text': f"Evaluated {len(evaluation_results)} models out of {len(models)} total models."
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
        logger.error(f"Error executing action '{action}': {str(e)}")
        state.tool_outputs['error'] = {
            'type': 'error',
            'text': f"Error executing {action}: {str(e)}",
            'data': {'error': str(e)}
        }
    
    # Remove the completed step
    state.plan.pop(0)
    
    return state


@retry_on_failure(max_retries=3, delay=1.0, backoff=2.0)
def _execute_with_retry(func, *args, **kwargs):
    """Execute a function with retry logic and timeout handling."""
    import threading
    import queue
    
    def execute_with_timeout():
        try:
            result = func(*args, **kwargs)
            result_queue.put(('success', result))
        except Exception as e:
            result_queue.put(('error', e))
    
    # Use threading for timeout (cross-platform)
    result_queue = queue.Queue()
    thread = threading.Thread(target=execute_with_timeout)
    thread.daemon = True
    thread.start()
    
    try:
        # Wait for result with timeout (30 seconds)
        result_type, result = result_queue.get(timeout=30)
        if result_type == 'success':
            return result
        else:
            raise result
    except queue.Empty:
        raise TimeoutError(f"Operation {func.__name__} timed out after 30 seconds")


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
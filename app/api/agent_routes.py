"""
Agent API Routes
Natural language agent interface endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
from flask_limiter.util import get_remote_address
from datetime import datetime
import logging

from app.models.agent_models import (
    AgentRequest, AgentResponse, AgentError, create_agent_response, create_agent_error
)
from app.services.agent.agent_service import AgentService
from app.config.adapter_config import create_adapter
from app.utils.request_context import request_context_manager, get_current_request_id
from app.utils.structured_logger import create_structured_logger

# Create blueprint
agent_bp = Blueprint('agent', __name__)

# Get logger
logger = create_structured_logger("agent_api")


def validate_agent_request(data: dict) -> AgentRequest:
    """Validate and create an agent request from JSON data.
    
    Args:
        data: JSON data from request
        
    Returns:
        Validated AgentRequest object
        
    Raises:
        ValueError: If validation fails
    """
    if not data:
        raise ValueError("Request body is required")
    
    if 'query' not in data:
        raise ValueError("'query' field is required")
    
    query = data.get('query', '').strip()
    if not query:
        raise ValueError("Query cannot be empty")
    
    if len(query) > 1000:
        raise ValueError("Query too long (max 1000 characters)")
    
    return AgentRequest(
        query=query,
        context=data.get('context', {}),
        user_id=data.get('user_id'),
        session_id=data.get('session_id')
    )


@agent_bp.route('/ask', methods=['POST'])
def ask_agent():
    """
    Natural language agent endpoint.
    
    Accepts natural language queries and returns structured responses.
    
    Request body:
    {
        "query": "What's the health status?",
        "context": {"optional": "context"},
        "user_id": "optional_user_id",
        "session_id": "optional_session_id"
    }
    
    Response:
    {
        "text": "Service is healthy",
        "data": {...},
        "metadata": {...},
        "timestamp": "2024-01-01T00:00:00Z",
        "request_id": "uuid"
    }
    """
    start_time = datetime.utcnow()
    
    try:
        # Check content type
        if not request.is_json:
            return jsonify(create_agent_error(
                error_code="VALIDATION_ERROR",
                message="Content-Type must be application/json"
            ).to_dict()), 400
        
        # Parse JSON
        try:
            data = request.get_json()
        except Exception as e:
            return jsonify(create_agent_error(
                error_code="VALIDATION_ERROR",
                message=f"Invalid JSON in request body: {str(e)}"
            ).to_dict()), 400
        
        # Validate request
        try:
            agent_request = validate_agent_request(data)
        except ValueError as e:
            return jsonify(create_agent_error(
                error_code="VALIDATION_ERROR",
                message=str(e)
            ).to_dict()), 400
        
        # Log incoming request
        logger.log_intent(
            intent="agent_request_received",
            confidence=1.0,
            query_length=len(agent_request.query),
            user_id=agent_request.user_id,
            session_id=agent_request.session_id
        )
        
        # Create forecaster service and agent service
        forecaster_service = create_adapter()
        agent_service = AgentService(forecaster_service)
        
        # Process the query
        with request_context_manager() as request_id:
            response = agent_service.process_query(agent_request)
            
            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            
            # Log successful response
            logger.log_outcome(
                operation="agent_query_processing",
                success=True,
                duration=duration,
                intent=response.metadata.get('intent', 'unknown'),
                confidence=response.metadata.get('confidence', 0.0)
            )
            
            # Return response
            return jsonify(response.to_dict()), 200
    
    except Exception as e:
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Log error
        logger.logger.error(
            f"Error in agent ask endpoint: {str(e)}",
            extra={
                'request_id': get_current_request_id(),
                'error_type': type(e).__name__,
                'duration_ms': round(duration * 1000, 2)
            },
            exc_info=True
        )
        
        # Return error response
        return jsonify(create_agent_error(
            error_code="INTERNAL_ERROR",
            message="An internal error occurred while processing your request",
            details={"error": str(e)} if current_app.debug else None,
            request_id=get_current_request_id()
        ).to_dict()), 500


@agent_bp.route('/health', methods=['GET'])
def agent_health():
    """
    Agent health check endpoint.
    """
    try:
        # Test that we can create the agent service
        forecaster_service = create_adapter()
        agent_service = AgentService(forecaster_service)
        
        return jsonify({
            'status': 'healthy',
            'service': 'Agent API',
            'version': current_app.config.get('API_VERSION', 'v1'),
            'timestamp': datetime.utcnow().isoformat(),
            'capabilities': [
                'natural_language_processing',
                'intent_recognition',
                'slot_extraction',
                'forecaster_integration'
            ]
        }), 200
        
    except Exception as e:
        logger.logger.error(f"Agent health check failed: {str(e)}", exc_info=True)
        
        return jsonify({
            'status': 'unhealthy',
            'service': 'Agent API',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@agent_bp.route('/capabilities', methods=['GET'])
def agent_capabilities():
    """
    Get agent capabilities and supported intents.
    """
    return jsonify({
        'capabilities': {
            'intents': [
                {
                    'name': 'forecast',
                    'description': 'Generate forecasts for keywords',
                    'keywords': ['forecast', 'predict', 'future', 'next'],
                    'supported': True
                },
                {
                    'name': 'compare',
                    'description': 'Compare trends between keywords',
                    'keywords': ['compare', 'versus', 'vs', 'difference'],
                    'supported': True
                },
                {
                    'name': 'summary',
                    'description': 'Get trend summaries',
                    'keywords': ['summary', 'overview', 'trends', 'data'],
                    'supported': True
                },
                {
                    'name': 'train',
                    'description': 'Train forecasting models',
                    'keywords': ['train', 'model', 'learn'],
                    'supported': False
                },
                {
                    'name': 'evaluate',
                    'description': 'Evaluate model performance',
                    'keywords': ['evaluate', 'performance', 'accuracy'],
                    'supported': False
                },
                {
                    'name': 'health',
                    'description': 'Check service health',
                    'keywords': ['health', 'status', 'working'],
                    'supported': True
                },
                {
                    'name': 'cache_stats',
                    'description': 'Get cache statistics',
                    'keywords': ['cache', 'stats', 'statistics'],
                    'supported': True
                },
                {
                    'name': 'cache_clear',
                    'description': 'Clear cache',
                    'keywords': ['clear', 'reset', 'cache'],
                    'supported': True
                }
            ],
            'slot_extraction': {
                'keywords': 'Quoted strings ("keyword")',
                'time_expressions': 'next week, next month, next year',
                'quantiles': 'p10, p50, p90'
            },
            'response_format': {
                'text': 'Human-readable response',
                'data': 'Structured data',
                'metadata': 'Processing metadata',
                'timestamp': 'ISO timestamp',
                'request_id': 'Unique request identifier'
            }
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200 
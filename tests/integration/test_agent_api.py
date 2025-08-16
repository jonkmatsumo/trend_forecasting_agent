"""
Integration Tests for Agent API
Tests the complete agent API request/response flow.
"""

import pytest
import json
from unittest.mock import patch, Mock

from app import create_app
from app.models.agent_models import AgentIntent


class TestAgentAPI:
    """Test agent API endpoints."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.app = create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
    
    def test_ask_endpoint_success(self):
        """Test successful agent ask request."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.health.return_value = {
                'status': 'healthy',
                'service': 'Test Service',
                'version': 'v1'
            }
            mock_create_adapter.return_value = mock_forecaster
            
            # Make request
            response = self.client.post('/agent/ask', 
                json={'query': "What's the health status?"},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['text'] == 'Service is healthy'
            assert data['data']['status'] == 'healthy'
            assert data['metadata']['intent'] == 'health'
            assert data['metadata']['confidence'] == 0.8
            assert 'request_id' in data
            assert 'timestamp' in data
    
    def test_ask_endpoint_cache_stats(self):
        """Test cache stats agent request."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.cache_stats.return_value = {
                'status': 'success',
                'cache_stats': {
                    'cache_size': 150,
                    'cache_hits': 100,
                    'cache_misses': 50
                }
            }
            mock_create_adapter.return_value = mock_forecaster
            
            # Make request
            response = self.client.post('/agent/ask', 
                json={'query': "Show cache statistics"},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert '150' in data['text']
            assert data['metadata']['intent'] == 'cache_stats'
            assert data['metadata']['confidence'] == 0.8
    
    def test_ask_endpoint_cache_clear(self):
        """Test cache clear agent request."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.cache_clear.return_value = {
                'status': 'success',
                'message': 'Cache cleared successfully'
            }
            # Ensure the mock returns a proper dict, not a Mock object
            mock_forecaster.cache_clear.return_value = {
                'status': 'success',
                'message': 'Cache cleared successfully'
            }
            mock_create_adapter.return_value = mock_forecaster
            
            # Make request
            response = self.client.post('/agent/ask', 
                json={'query': "Clear the cache"},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert 'cleared' in data['text'].lower()
            assert data['metadata']['intent'] == 'cache_clear'
            assert data['metadata']['confidence'] == 0.7
    
    def test_ask_endpoint_validation_error_empty_query(self):
        """Test validation error for empty query."""
        response = self.client.post('/agent/ask', 
            json={'query': ''},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'empty' in data['message'].lower()
    
    def test_ask_endpoint_validation_error_missing_query(self):
        """Test validation error for missing query."""
        response = self.client.post('/agent/ask', 
            json={'context': {'key': 'value'}},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'required' in data['message'].lower()
    
    def test_ask_endpoint_validation_error_long_query(self):
        """Test validation error for query too long."""
        long_query = "x" * 1001
        response = self.client.post('/agent/ask', 
            json={'query': long_query},
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'too long' in data['message'].lower()
    
    def test_ask_endpoint_invalid_json(self):
        """Test error for invalid JSON."""
        response = self.client.post('/agent/ask', 
            data='invalid json',
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'JSON' in data['message']
    
    def test_ask_endpoint_wrong_content_type(self):
        """Test error for wrong content type."""
        response = self.client.post('/agent/ask', 
            data=json.dumps({'query': 'test'}),
            content_type='text/plain'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'Content-Type' in data['message']
    
    def test_ask_endpoint_service_error(self):
        """Test error handling when service fails."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service to raise an exception
            mock_forecaster = Mock()
            mock_forecaster.health.side_effect = Exception("Service unavailable")
            mock_create_adapter.return_value = mock_forecaster
            
            # Make request
            response = self.client.post('/agent/ask', 
                json={'query': "What's the health status?"},
                content_type='application/json'
            )
            
            assert response.status_code == 200  # Agent handles errors gracefully
            data = json.loads(response.data)
            
            assert 'error' in data['text'].lower()
            assert data['metadata']['intent'] == 'error'
            assert data['metadata']['confidence'] == 0.0
    
    def test_ask_endpoint_with_context(self):
        """Test agent request with context."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.health.return_value = {
                'status': 'healthy',
                'service': 'Test Service'
            }
            mock_create_adapter.return_value = mock_forecaster
            
            # Make request with context
            response = self.client.post('/agent/ask', 
                json={
                    'query': "What's the health status?",
                    'context': {'user_type': 'admin', 'session': 'test'},
                    'user_id': 'user123',
                    'session_id': 'session456'
                },
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['text'] == 'Service is healthy'
            assert data['metadata']['intent'] == 'health'
    
    def test_health_endpoint_success(self):
        """Test agent health endpoint."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_create_adapter.return_value = mock_forecaster
            
            response = self.client.get('/agent/health')
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            assert data['status'] == 'healthy'
            assert data['service'] == 'Agent API'
            assert 'capabilities' in data
            assert 'natural_language_processing' in data['capabilities']
    
    def test_health_endpoint_error(self):
        """Test agent health endpoint with error."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service to raise an exception
            mock_create_adapter.side_effect = Exception("Service unavailable")
            
            response = self.client.get('/agent/health')
            
            assert response.status_code == 500
            data = json.loads(response.data)
            
            assert data['status'] == 'unhealthy'
            assert data['service'] == 'Agent API'
            assert 'error' in data
    
    def test_capabilities_endpoint(self):
        """Test agent capabilities endpoint."""
        response = self.client.get('/agent/capabilities')
        
        assert response.status_code == 200
        data = json.loads(response.data)
        
        assert 'capabilities' in data
        assert 'intents' in data['capabilities']
        assert 'slot_extraction' in data['capabilities']
        assert 'response_format' in data['capabilities']
        
        # Check that we have the expected intents
        intent_names = [intent['name'] for intent in data['capabilities']['intents']]
        assert 'health' in intent_names
        assert 'cache_stats' in intent_names
        assert 'cache_clear' in intent_names
        assert 'forecast' in intent_names
        assert 'compare' in intent_names
        assert 'summary' in intent_names
    
    def test_intent_recognition_variations(self):
        """Test various intent recognition patterns."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.health.return_value = {'status': 'healthy'}
            mock_forecaster.cache_stats.return_value = {
                'status': 'success',
                'cache_stats': {'cache_size': 100}
            }
            mock_forecaster.cache_clear.return_value = {
                'status': 'success',
                'message': 'Cache cleared'
            }
            mock_create_adapter.return_value = mock_forecaster
            
            # Test health variations
            health_queries = [
                "What's the health status?",
                "Is the service working?",
                "Check system health",
                "HEALTH STATUS",
                "health check"
            ]
            
            for query in health_queries:
                response = self.client.post('/agent/ask', 
                    json={'query': query},
                    content_type='application/json'
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['metadata']['intent'] == 'health'
            
            # Test cache stats variations
            cache_stats_queries = [
                "Show cache statistics",
                "What are the cache stats?",
                "Cache information",
                "cache stats"
            ]
            
            for query in cache_stats_queries:
                response = self.client.post('/agent/ask', 
                    json={'query': query},
                    content_type='application/json'
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['metadata']['intent'] == 'cache_stats'
            
            # Test cache clear variations
            cache_clear_queries = [
                "Clear the cache",
                "Reset cache",
                "clear cache"
            ]
            
            for query in cache_clear_queries:
                response = self.client.post('/agent/ask', 
                    json={'query': query},
                    content_type='application/json'
                )
                assert response.status_code == 200
                data = json.loads(response.data)
                assert data['metadata']['intent'] == 'cache_clear'
    
    def test_slot_extraction(self):
        """Test slot extraction from queries."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.health.return_value = {'status': 'healthy'}
            mock_create_adapter.return_value = mock_forecaster
            
            # Test keyword extraction
            response = self.client.post('/agent/ask', 
                json={'query': 'Forecast "python" trends'},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['metadata']['intent'] == 'forecast'
            # Note: Slot extraction is tested in unit tests, here we just verify the flow works
    
    def test_unknown_intent(self):
        """Test handling of unknown intents."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_create_adapter.return_value = mock_forecaster
            
            response = self.client.post('/agent/ask', 
                json={'query': 'Random text that does not match any intent'},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            assert data['metadata']['intent'] == 'unknown'
            assert data['metadata']['confidence'] == 0.5
    
    def test_request_tracking(self):
        """Test that request IDs are properly tracked."""
        with patch('app.api.agent_routes.create_adapter') as mock_create_adapter:
            # Mock the forecaster service
            mock_forecaster = Mock()
            mock_forecaster.health.return_value = {'status': 'healthy'}
            mock_create_adapter.return_value = mock_forecaster
            
            response = self.client.post('/agent/ask', 
                json={'query': "What's the health status?"},
                content_type='application/json'
            )
            
            assert response.status_code == 200
            data = json.loads(response.data)
            
            # Check that request ID is present and valid
            assert 'request_id' in data
            assert data['request_id'] is not None
            assert len(data['request_id']) > 0
            
            # Check that timestamp is present and valid
            assert 'timestamp' in data
            assert data['timestamp'] is not None 
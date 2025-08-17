"""
Unit Tests for HTTP Client
Tests for the HTTP client implementation.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests
from requests.exceptions import RequestException

from app.agent_graph.http.http_client import HTTPClient
from app.agent_graph.http.http_config import HTTPClientConfig
from app.agent_graph.http.http_models import HTTPError, HealthRequest


class TestHTTPClient:
    """Test HTTP client functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = HTTPClientConfig(
            base_url="http://localhost:5000",
            timeout=30,
            max_retries=3,
            enable_request_logging=False
        )
        self.client = HTTPClient(self.config)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.client.close()
    
    def test_initialization(self):
        """Test client initialization."""
        assert self.client.config == self.config
        assert self.client.session is not None
        assert self.client.logger is not None
    
    def test_create_session(self):
        """Test session creation."""
        session = self.client._create_session()
        assert session is not None
        assert session.headers.get("Content-Type") == "application/json"
        assert session.headers.get("User-Agent") == "TrendForecaster-HTTPClient/1.0"
    
    def test_create_retry_strategy(self):
        """Test retry strategy creation."""
        retry = self.client._create_retry_strategy()
        assert retry.total == 3
        assert retry.backoff_factor == 1.0
        assert 429 in retry.status_forcelist
        assert 500 in retry.status_forcelist
    
    @patch('requests.Session.request')
    def test_health_success(self, mock_request):
        """Test successful health check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"status": "healthy"}
        mock_response.url = "http://localhost:5000/health"
        mock_request.return_value = mock_response
        
        result = self.client.health()
        
        assert result == {"status": "healthy"}
        mock_request.assert_called_once()
    
    @patch('requests.Session.request')
    def test_health_error(self, mock_request):
        """Test health check with error."""
        # Mock error response
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.url = "http://localhost:5000/health"
        mock_response.headers = {}
        mock_request.return_value = mock_response
        
        with pytest.raises(HTTPError) as exc_info:
            self.client.health()
        
        assert exc_info.value.status_code == 500
        assert "Internal Server Error" in str(exc_info.value)
    
    @patch('requests.Session.request')
    def test_network_error(self, mock_request):
        """Test network error handling."""
        # Mock network error
        mock_request.side_effect = RequestException("Connection failed")
        
        with pytest.raises(HTTPError) as exc_info:
            self.client.health()
        
        assert exc_info.value.status_code == 0
        assert "Connection failed" in str(exc_info.value)
    
    @patch('requests.Session.request')
    def test_trends_summary(self, mock_request):
        """Test trends summary request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"summary": "test data"}
        mock_response.url = "http://localhost:5000/api/trends/summary"
        mock_request.return_value = mock_response
        
        result = self.client.trends_summary(["python", "javascript"])
        
        assert result == {"summary": "test data"}
        # Verify request was made with correct data
        call_args = mock_request.call_args
        assert call_args[1]["json"]["keywords"] == ["python", "javascript"]
    
    @patch('requests.Session.request')
    def test_predict(self, mock_request):
        """Test predict request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"predictions": [1, 2, 3]}
        mock_response.url = "http://localhost:5000/api/models/test-model/predict"
        mock_request.return_value = mock_response
        
        result = self.client.predict("test-model", forecast_horizon=10)
        
        assert result == {"predictions": [1, 2, 3]}
        # Verify request was made with correct data
        call_args = mock_request.call_args
        assert call_args[1]["json"]["forecast_horizon"] == 10
    
    @patch('requests.Session.request')
    def test_train(self, mock_request):
        """Test train request."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"model_id": "test-model-123"}
        mock_response.url = "http://localhost:5000/api/models/train"
        mock_request.return_value = mock_response
        
        result = self.client.train(
            keyword="python",
            time_series_data=[1.0, 2.0, 3.0],
            dates=["2023-01-01", "2023-01-02", "2023-01-03"],
            model_type="prophet"
        )
        
        assert result == {"model_id": "test-model-123"}
        # Verify request was made with correct data
        call_args = mock_request.call_args
        assert call_args[1]["json"]["keyword"] == "python"
        assert call_args[1]["json"]["model_type"] == "prophet"
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with self.client as client:
            assert client == self.client
        
        # Session should be closed
        assert hasattr(self.client.session, 'close')


class TestHTTPForecasterClient:
    """Test HTTP forecaster client wrapper."""
    
    def setup_method(self):
        """Set up test fixtures."""
        with patch('app.agent_graph.service_client.HTTPClient') as mock_http_client:
            self.mock_http_client = Mock()
            mock_http_client.return_value = self.mock_http_client
            from app.agent_graph.service_client import HTTPForecasterClient
            self.client = HTTPForecasterClient("http://localhost:5000")
    
    def test_health(self):
        """Test health check."""
        self.mock_http_client.health.return_value = {"status": "healthy"}
        
        result = self.client.health()
        
        assert result == {"status": "healthy"}
        self.mock_http_client.health.assert_called_once()
    
    def test_health_error(self):
        """Test health check with error."""
        self.mock_http_client.health.side_effect = Exception("Connection failed")
        
        result = self.client.health()
        
        assert result["status"] == "unhealthy"
        assert "Connection failed" in result["error"]
    
    def test_trends_summary(self):
        """Test trends summary."""
        self.mock_http_client.trends_summary.return_value = {"summary": "test"}
        
        result = self.client.trends_summary(["python"])
        
        assert result == {"summary": "test"}
        self.mock_http_client.trends_summary.assert_called_once_with(["python"], "today 12-m", "")
    
    def test_compare(self):
        """Test compare functionality."""
        self.mock_http_client.compare.return_value = {"comparison": "test"}
        
        result = self.client.compare(["python", "javascript"])
        
        assert result == {"comparison": "test"}
        self.mock_http_client.compare.assert_called_once_with(["python", "javascript"], "today 12-m", "")
    
    def test_list_models(self):
        """Test list models functionality."""
        self.mock_http_client.list_models.return_value = {"models": []}
        
        result = self.client.list_models(keyword="python", limit=10)
        
        assert result == {"models": []}
        self.mock_http_client.list_models.assert_called_once_with("python", None, 10, 0)
    
    def test_predict(self):
        """Test predict functionality."""
        self.mock_http_client.predict.return_value = {"predictions": [1, 2, 3]}
        
        result = self.client.predict("test-model", forecast_horizon=5)
        
        assert result == {"predictions": [1, 2, 3]}
        self.mock_http_client.predict.assert_called_once_with("test-model", 5)
    
    def test_train(self):
        """Test train functionality."""
        self.mock_http_client.train.return_value = {"model_id": "test-123"}
        
        result = self.client.train(
            keyword="python",
            time_series_data=[1.0, 2.0],
            dates=["2023-01-01", "2023-01-02"],
            model_type="prophet"
        )
        
        assert result == {"model_id": "test-123"}
        self.mock_http_client.train.assert_called_once()
    
    def test_evaluate(self):
        """Test evaluate functionality."""
        self.mock_http_client.evaluate.return_value = {"metrics": {"mae": 0.1}}
        
        result = self.client.evaluate("test-model")
        
        assert result == {"metrics": {"mae": 0.1}}
        self.mock_http_client.evaluate.assert_called_once_with("test-model")
    
    def test_context_manager(self):
        """Test context manager functionality."""
        with self.client as client:
            assert client == self.client
        
        self.mock_http_client.close.assert_called_once()


class TestHTTPClientConfig:
    """Test HTTP client configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HTTPClientConfig(base_url="http://localhost:5000")
        
        assert config.base_url == "http://localhost:5000"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.max_retry_delay == 60.0
        assert config.pool_connections == 10
        assert config.pool_maxsize == 20
        assert config.enable_request_logging is True
        assert config.log_request_body is False
        assert config.log_response_body is False
        assert "Content-Type" in config.default_headers
        assert "User-Agent" in config.default_headers
    
    def test_config_with_api_key(self):
        """Test configuration with API key."""
        config = HTTPClientConfig(
            base_url="http://localhost:5000",
            api_key="test-api-key"
        )
        
        assert config.default_headers["Authorization"] == "Bearer test-api-key"
    
    def test_load_http_config(self):
        """Test loading configuration from environment variables."""
        with patch.dict('os.environ', {
            'FORECASTER_API_URL': 'http://test:8080',
            'HTTP_TIMEOUT': '60',
            'HTTP_MAX_RETRIES': '5',
            'HTTP_LOG_REQUESTS': 'false'
        }):
            from app.agent_graph.http.http_config import load_http_config
            config = load_http_config()
            
            assert config.base_url == "http://test:8080"
            assert config.timeout == 60
            assert config.max_retries == 5
            assert config.enable_request_logging is False


class TestHTTPModels:
    """Test HTTP request/response models."""
    
    def test_http_request_validation(self):
        """Test HTTP request validation."""
        from app.agent_graph.http.http_models import HTTPRequest, HTTPMethod
        
        # Valid request
        request = HTTPRequest(
            method=HTTPMethod.GET,
            url="http://localhost:5000/test",
            json={"test": "data"}
        )
        assert request.method == HTTPMethod.GET
        assert request.json == {"test": "data"}
        
        # Invalid request - both data and json
        with pytest.raises(ValueError, match="Cannot specify both data and json"):
            HTTPRequest(
                method=HTTPMethod.POST,
                url="http://localhost:5000/test",
                data={"test": "data"},
                json={"test": "data"}
            )
    
    def test_http_response_properties(self):
        """Test HTTP response properties."""
        from app.agent_graph.http.http_models import HTTPResponse
        
        # Success response
        response = HTTPResponse(status_code=200, headers={})
        assert response.is_success is True
        assert response.is_client_error is False
        assert response.is_server_error is False
        
        # Client error response
        response = HTTPResponse(status_code=404, headers={})
        assert response.is_success is False
        assert response.is_client_error is True
        assert response.is_server_error is False
        
        # Server error response
        response = HTTPResponse(status_code=500, headers={})
        assert response.is_success is False
        assert response.is_client_error is False
        assert response.is_server_error is True
    
    def test_http_error(self):
        """Test HTTP error exception."""
        from app.agent_graph.http.http_models import HTTPError
        
        error = HTTPError(status_code=404, message="Not Found")
        assert str(error) == "HTTP 404: Not Found"
        assert error.status_code == 404
        assert error.message == "Not Found"
    
    def test_specific_request_models(self):
        """Test specific request models."""
        from app.agent_graph.http.http_models import (
            HealthRequest, TrendsSummaryRequest, CompareRequest,
            ListModelsRequest, PredictRequest, TrainRequest, EvaluateRequest
        )
        
        # Health request
        health_req = HealthRequest("http://localhost:5000")
        assert health_req.url == "http://localhost:5000/health"
        
        # Trends summary request
        trends_req = TrendsSummaryRequest("http://localhost:5000", ["python", "javascript"])
        assert trends_req.url == "http://localhost:5000/api/trends/summary"
        assert trends_req.json["keywords"] == ["python", "javascript"]
        
        # Compare request
        compare_req = CompareRequest("http://localhost:5000", ["python", "javascript"])
        assert compare_req.url == "http://localhost:5000/api/trends/compare"
        
        # List models request
        list_req = ListModelsRequest("http://localhost:5000", keyword="python", limit=10)
        assert list_req.url == "http://localhost:5000/api/models"
        assert list_req.params["keyword"] == "python"
        assert list_req.params["limit"] == 10
        
        # Predict request
        predict_req = PredictRequest("http://localhost:5000", "test-model", forecast_horizon=5)
        assert predict_req.url == "http://localhost:5000/api/models/test-model/predict"
        assert predict_req.json["forecast_horizon"] == 5
        
        # Train request
        train_req = TrainRequest(
            "http://localhost:5000",
            "python",
            [1.0, 2.0],
            ["2023-01-01", "2023-01-02"],
            "prophet"
        )
        assert train_req.url == "http://localhost:5000/api/models/train"
        assert train_req.json["keyword"] == "python"
        assert train_req.json["model_type"] == "prophet"
        
        # Evaluate request
        eval_req = EvaluateRequest("http://localhost:5000", "test-model")
        assert eval_req.url == "http://localhost:5000/api/models/test-model/evaluate" 
"""
Basic tests for Flask application structure
"""

import pytest
from app import create_app


@pytest.fixture
def app():
    """Create and configure a new app instance for each test."""
    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


def test_app_creation(app):
    """Test that the app can be created."""
    assert app is not None
    assert app.config['TESTING'] is True


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get('/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'Google Trends Quantile Forecaster API' in data['service']


def test_api_health_endpoint(client):
    """Test the API health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = response.get_json()
    assert data['status'] == 'healthy'
    assert 'Google Trends Quantile Forecaster API' in data['service']


def test_404_error_handler(client):
    """Test the 404 error handler."""
    response = client.get('/nonexistent')
    assert response.status_code == 404
    
    data = response.get_json()
    assert data['error'] == 'Not Found'
    assert data['status_code'] == 400


def test_app_configuration(app):
    """Test that the app has the expected configuration."""
    assert 'SECRET_KEY' in app.config
    assert 'API_VERSION' in app.config
    assert 'MLFLOW_TRACKING_URI' in app.config 
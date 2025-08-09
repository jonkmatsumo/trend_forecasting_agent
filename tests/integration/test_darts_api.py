"""
Integration tests for Darts-based API endpoints
"""

import pytest
import json
from unittest.mock import patch, Mock
from datetime import datetime, timedelta
import uuid

from app import create_app
from app.models.darts.darts_models import (
    ModelType, ModelTrainingRequest, ModelEvaluationMetrics, ForecastResult
)


class TestDartsAPI:
    """Integration tests for Darts-based API endpoints"""
    
    @pytest.fixture
    def app(self):
        """Create Flask app for testing"""
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    @pytest.fixture
    def mock_darts_services(self):
        """Mock Darts services responses"""
        with patch('app.services.darts.training_service.TrainingService') as mock_training_service, \
             patch('app.services.darts.prediction_service.PredictionService') as mock_prediction_service:
            
            # Mock training service
            mock_training_instance = Mock()
            mock_training_instance.train_model.return_value = (
                "test_model_123",
                ModelEvaluationMetrics(
                    model_id="test_model_123",
                    keyword="python",
                    model_type=ModelType.LSTM,
                    train_mae=0.15,
                    train_rmse=0.25,
                    train_mape=0.12,
                    test_mae=0.18,
                    test_rmse=0.28,
                    test_mape=0.15,
                    directional_accuracy=0.75,
                    coverage_95=0.92,
                    train_samples=80,
                    test_samples=20,
                    total_samples=100,
                    training_time_seconds=45.5,
                    mlflow_run_id="mlflow_run_456"
                )
            )
            
            mock_training_instance.get_model_info.return_value = {
                "model_id": "test_model_123",
                "keyword": "python",
                "model_type": "lstm",
                "created_at": "2023-12-01T10:00:00",
                "status": "trained",
                "metrics": {
                    "test_mae": 0.18,
                    "test_rmse": 0.28,
                    "directional_accuracy": 0.75
                }
            }
            
            mock_training_instance.list_models.return_value = [
                {
                    "model_id": "test_model_123",
                    "keyword": "python",
                    "model_type": "lstm",
                    "created_at": "2023-12-01T10:00:00",
                    "status": "trained"
                },
                {
                    "model_id": "test_model_456",
                    "keyword": "javascript",
                    "model_type": "tcn",
                    "created_at": "2023-12-02T11:00:00",
                    "status": "trained"
                }
            ]
            
            # Mock prediction service
            mock_prediction_instance = Mock()
            mock_prediction_instance.generate_forecast.return_value = ForecastResult(
                model_id="test_model_123",
                keyword="python",
                forecast_values=[75.2, 76.1, 77.3, 78.5, 79.2],
                forecast_dates=[
                    datetime.now() + timedelta(weeks=i) 
                    for i in range(1, 6)
                ],
                confidence_intervals={
                    "95%": {
                        "lower": [70.1, 71.2, 72.4, 73.6, 74.3],
                        "upper": [80.3, 81.0, 82.2, 83.4, 84.1]
                    }
                },
                model_metrics=ModelEvaluationMetrics(
                    model_id="test_model_123",
                    keyword="python",
                    model_type=ModelType.LSTM,
                    train_mae=0.15,
                    train_rmse=0.25,
                    train_mape=0.12,
                    test_mae=0.18,
                    test_rmse=0.28,
                    test_mape=0.15,
                    directional_accuracy=0.75,
                    coverage_95=0.92,
                    train_samples=80,
                    test_samples=20,
                    total_samples=100,
                    training_time_seconds=45.5
                ),
                forecast_horizon=5
            )
            
            mock_training_service.return_value = mock_training_instance
            mock_prediction_service.return_value = mock_prediction_instance
            
            yield {
                'training_service': mock_training_instance,
                'prediction_service': mock_prediction_instance
            }
    
    def test_train_model_endpoint_success(self, client, mock_darts_services):
        """Test successful model training"""
        # Prepare request data with valid values (0-100 range)
        request_data = {
            "keyword": "python",
            "time_series_data": [50.0 + i * 0.5 for i in range(52)],  # 52 data points, 50-75 range
            "dates": [f"2023-{i:02d}-01" for i in range(1, 53)],
            "model_type": "lstm",
            "train_test_split": 0.8,
            "forecast_horizon": 25,
            "model_parameters": {
                "input_chunk_length": 12,
                "output_chunk_length": 1
            },
            "validation_strategy": "holdout"
        }
        
        # Make request
        response = client.post('/api/models/train', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 201
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['message'] == 'Model trained successfully'
        assert data['model_id'] == 'test_model_123'
        assert data['keyword'] == 'python'
        assert 'training_metrics' in data
        assert data['training_metrics']['model_id'] == 'test_model_123'
        assert data['training_metrics']['model_type'] == 'lstm'
        
        # Verify service was called
        mock_darts_services['training_service'].train_model.assert_called_once()
        call_args = mock_darts_services['training_service'].train_model.call_args[0][0]
        assert isinstance(call_args, ModelTrainingRequest)
        assert call_args.keyword == 'python'
        assert call_args.model_type == ModelType.LSTM
    
    def test_train_model_endpoint_validation_error(self, client):
        """Test model training with validation error"""
        # Prepare invalid request data (too few data points)
        request_data = {
            "keyword": "python",
            "time_series_data": [50.0, 51.0],  # Too few data points
            "model_type": "lstm"
        }
        
        # Make request
        response = client.post('/api/models/train', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'At least 10 data points required' in data['message']
    
    def test_train_model_endpoint_missing_body(self, client):
        """Test model training with missing request body"""
        # Make request without body
        response = client.post('/api/models/train')
        
        # Verify response - should be 500 due to content-type handling
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_generate_prediction_endpoint_success(self, client, mock_darts_services):
        """Test successful prediction generation"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Prepare request data
        request_data = {
            "prediction_weeks": 5
        }
        
        # Make request
        response = client.post(f'/api/models/{valid_model_id}/predict', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['model_id'] == valid_model_id
        assert data['prediction_weeks'] == 5
        assert 'result' in data
        assert data['result']['model_id'] == 'test_model_123'
        assert data['result']['keyword'] == 'python'
        assert 'forecast' in data['result']
        assert len(data['result']['forecast']['values']) == 5
        assert 'confidence_intervals' in data['result']['forecast']
        
        # Verify service was called
        mock_darts_services['prediction_service'].generate_forecast.assert_called_once_with(
            valid_model_id, 5
        )
    
    def test_generate_prediction_endpoint_validation_error(self, client):
        """Test prediction with validation error"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Prepare invalid request data
        request_data = {
            "prediction_weeks": -5  # Invalid negative value
        }
        
        # Make request
        response = client.post(f'/api/models/{valid_model_id}/predict', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
    
    def test_generate_prediction_endpoint_model_not_found(self, client, mock_darts_services):
        """Test prediction with non-existent model"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Mock service to raise Exception
        mock_darts_services['prediction_service'].generate_forecast.side_effect = \
            Exception("Model not found")
        
        request_data = {
            "prediction_weeks": 5
        }
        
        # Make request
        response = client.post(f'/api/models/{valid_model_id}/predict', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_get_model_info_endpoint_success(self, client, mock_darts_services):
        """Test successful model info retrieval"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Make request
        response = client.get(f'/api/models/{valid_model_id}')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'model_info' in data
        assert data['model_info']['model_id'] == 'test_model_123'
        assert data['model_info']['keyword'] == 'python'
        assert data['model_info']['model_type'] == 'lstm'
        assert data['model_info']['status'] == 'trained'
        
        # Verify service was called
        mock_darts_services['training_service'].get_model_info.assert_called_once_with(valid_model_id)
    
    def test_get_model_info_endpoint_not_found(self, client, mock_darts_services):
        """Test model info with non-existent model"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Mock service to return None (model not found)
        mock_darts_services['training_service'].get_model_info.return_value = None
        
        # Make request
        response = client.get(f'/api/models/{valid_model_id}')
        
        # Verify response
        assert response.status_code == 404
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'NOT_FOUND'
        assert 'not found' in data['message'].lower()
    
    def test_list_models_endpoint_success(self, client, mock_darts_services):
        """Test successful models listing"""
        # Make request
        response = client.get('/api/models')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert 'models' in data
        assert data['count'] == 2
        assert len(data['models']) == 2
        
        # Check first model
        first_model = data['models'][0]
        assert first_model['model_id'] == 'test_model_123'
        assert first_model['keyword'] == 'python'
        assert first_model['model_type'] == 'lstm'
        
        # Check second model
        second_model = data['models'][1]
        assert second_model['model_id'] == 'test_model_456'
        assert second_model['keyword'] == 'javascript'
        assert second_model['model_type'] == 'tcn'
        
        # Verify service was called
        mock_darts_services['training_service'].list_models.assert_called_once()
    
    def test_list_models_endpoint_empty(self, client, mock_darts_services):
        """Test models listing with no models"""
        # Mock service to return empty list
        mock_darts_services['training_service'].list_models.return_value = []
        
        # Make request
        response = client.get('/api/models')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'success'
        assert data['models'] == []
        assert data['count'] == 0
    
    def test_health_endpoint(self, client):
        """Test API health endpoint"""
        # Make request
        response = client.get('/api/health')
        
        # Verify response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'Google Trends Quantile Forecaster API'
        assert 'version' in data
        assert 'timestamp' in data
    
    def test_train_model_endpoint_invalid_json(self, client):
        """Test model training with invalid JSON"""
        # Make request with invalid JSON
        response = client.post('/api/models/train', 
                             data='invalid json',
                             content_type='application/json')
        
        # Verify response - should be 500 due to JSON parsing error
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_generate_prediction_endpoint_invalid_json(self, client):
        """Test prediction with invalid JSON"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Make request with invalid JSON
        response = client.post(f'/api/models/{valid_model_id}/predict', 
                             data='invalid json',
                             content_type='application/json')
        
        # Verify response - should be 500 due to JSON parsing error
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_train_model_endpoint_model_error(self, client, mock_darts_services):
        """Test model training with model error"""
        # Mock service to raise Exception
        mock_darts_services['training_service'].train_model.side_effect = \
            Exception("Training failed")
        
        request_data = {
            "keyword": "python",
            "time_series_data": [50.0 + i * 0.5 for i in range(52)],  # Valid range
            "dates": [f"2023-{i:02d}-01" for i in range(1, 53)],
            "model_type": "lstm"
        }
        
        # Make request
        response = client.post('/api/models/train', 
                             data=json.dumps(request_data),
                             content_type='application/json')
        
        # Verify response
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_generate_prediction_endpoint_missing_body(self, client):
        """Test prediction with missing request body"""
        # Use a valid UUID format for model ID
        valid_model_id = str(uuid.uuid4())
        
        # Make request without body
        response = client.post(f'/api/models/{valid_model_id}/predict')
        
        # Verify response - should be 500 due to content-type handling
        assert response.status_code == 500
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'INTERNAL_ERROR'
    
    def test_invalid_model_id_format(self, client):
        """Test endpoints with invalid model ID format"""
        # Test prediction endpoint with invalid model ID
        response = client.post('/api/models/invalid_id/predict', 
                             data=json.dumps({"prediction_weeks": 5}),
                             content_type='application/json')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'UUID format' in data['message']
        
        # Test get model info endpoint with invalid model ID
        response = client.get('/api/models/invalid_id')
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert data['status'] == 'error'
        assert data['error_code'] == 'VALIDATION_ERROR'
        assert 'UUID format' in data['message'] 
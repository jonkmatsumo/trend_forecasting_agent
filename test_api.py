"""
Simple test script to demonstrate the API functionality
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:5000"

def test_health_endpoints():
    """Test health check endpoints"""
    print("Testing health endpoints...")
    
    # Test main health endpoint
    response = requests.get(f"{BASE_URL}/health")
    print(f"Main health: {response.status_code} - {response.json()}")
    
    # Test API health endpoint
    response = requests.get(f"{BASE_URL}/api/health")
    print(f"API health: {response.status_code} - {response.json()}")
    print()

def test_trends_endpoint():
    """Test trends endpoint with a sample keyword"""
    print("Testing trends endpoint...")
    
    data = {
        "keyword": "python",
        "timeframe": "today 12-m",
        "geo": ""
    }
    
    response = requests.post(f"{BASE_URL}/api/trends", json=data)
    print(f"Trends request: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Data points: {result['data']['data_points']}")
        print(f"Keyword: {result['keyword']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_model_training():
    """Test model training with sample data"""
    print("Testing model training...")
    
    # Generate sample time series data
    import numpy as np
    np.random.seed(42)
    sample_data = np.random.randint(0, 100, 50).tolist()
    
    data = {
        "keyword": "test_keyword",
        "time_series_data": sample_data,
        "model_params": {
            "epochs": 10,  # Reduced for testing
            "batch_size": 5,
            "lstm_units": 4
        }
    }
    
    response = requests.post(f"{BASE_URL}/api/models/train", json=data)
    print(f"Training request: {response.status_code}")
    
    if response.status_code == 201:
        result = response.json()
        print(f"Model ID: {result['model_id']}")
        print(f"Keyword: {result['keyword']}")
        print(f"Metrics: {result['training_metrics']}")
        return result['model_id']
    else:
        print(f"Error: {response.json()}")
        return None
    print()

def test_model_prediction(model_id):
    """Test model prediction"""
    if not model_id:
        print("No model ID available for prediction test")
        return
    
    print("Testing model prediction...")
    
    data = {
        "prediction_weeks": 10
    }
    
    response = requests.post(f"{BASE_URL}/api/models/{model_id}/predict", json=data)
    print(f"Prediction request: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Predictions: {len(result['predictions'])} values")
        print(f"Prediction weeks: {result['prediction_weeks']}")
        print(f"Sample predictions: {result['predictions'][:5]}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_model_info(model_id):
    """Test getting model information"""
    if not model_id:
        print("No model ID available for info test")
        return
    
    print("Testing model info...")
    
    response = requests.get(f"{BASE_URL}/api/models/{model_id}")
    print(f"Model info request: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Model keyword: {result['model_info']['keyword']}")
        print(f"Model status: {result['model_info']['status']}")
        print(f"Data points: {result['model_info']['data_points']}")
    else:
        print(f"Error: {response.json()}")
    print()

def test_list_models():
    """Test listing all models"""
    print("Testing list models...")
    
    response = requests.get(f"{BASE_URL}/api/models")
    print(f"List models request: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Total models: {result['count']}")
        if result['models']:
            print(f"Latest model: {result['models'][0]['keyword']}")
    else:
        print(f"Error: {response.json()}")
    print()

def main():
    """Run all tests"""
    print("=" * 50)
    print("Google Trends Quantile Forecaster API Test")
    print("=" * 50)
    print()
    
    try:
        # Test health endpoints
        test_health_endpoints()
        
        # Test trends endpoint
        test_trends_endpoint()
        
        # Test model training
        model_id = test_model_training()
        
        # Test model info
        test_model_info(model_id)
        
        # Test model prediction
        test_model_prediction(model_id)
        
        # Test list models
        test_list_models()
        
        print("=" * 50)
        print("All tests completed!")
        print("=" * 50)
        
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API server.")
        print("Make sure the Flask application is running on http://localhost:5000")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main() 
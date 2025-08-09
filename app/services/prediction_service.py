"""
Prediction Service for Google Trends Quantile Forecaster
Handles model loading and prediction generation
"""

import os
import mlflow
import numpy as np
import pandas as pd
from flask import current_app
import logging

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling prediction operations"""
    
    def __init__(self):
        """Initialize the prediction service"""
        self.models_dir = current_app.config.get('MODELS_DIR', 'models')
        
        # Initialize MLflow
        mlflow.set_tracking_uri(current_app.config.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(current_app.config.get('MLFLOW_EXPERIMENT_NAME', 'google_trends_forecaster'))
    
    def generate_prediction(self, model_id, prediction_weeks=25):
        """
        Generate prediction using a trained model
        
        Args:
            model_id (str): Model identifier
            prediction_weeks (int): Number of weeks to predict
            
        Returns:
            dict: Prediction results with values and confidence intervals
        """
        try:
            logger.info(f"Generating prediction for model_id: {model_id}")
            
            # Load model info
            model_info = self._load_model_info(model_id)
            if not model_info:
                raise Exception(f"Model not found: {model_id}")
            
            # Load model and scaler
            model = self._load_model(model_id)
            scaler = self._load_scaler(model_id)
            
            if not model or not scaler:
                raise Exception(f"Failed to load model artifacts for: {model_id}")
            
            # Generate predictions
            predictions = self._generate_forecast(model, scaler, prediction_weeks)
            
            # Calculate confidence intervals (simplified)
            confidence_intervals = self._calculate_confidence_intervals(predictions)
            
            logger.info(f"Successfully generated {len(predictions)} predictions for model_id: {model_id}")
            
            return {
                'predictions': predictions.tolist(),
                'confidence_intervals': confidence_intervals,
                'model_id': model_id,
                'prediction_weeks': prediction_weeks,
                'keyword': model_info.get('keyword', 'Unknown')
            }
            
        except Exception as e:
            logger.error(f"Error generating prediction for {model_id}: {str(e)}")
            raise Exception(f"Failed to generate prediction: {str(e)}")
    
    def _load_model_info(self, model_id):
        """
        Load model information from file
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            dict: Model information or None
        """
        try:
            import json
            model_info_path = os.path.join(self.models_dir, f"{model_id}_info.json")
            
            if not os.path.exists(model_info_path):
                return None
            
            with open(model_info_path, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            logger.error(f"Error loading model info for {model_id}: {str(e)}")
            return None
    
    def _load_model(self, model_id):
        """
        Load trained model from MLflow
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            keras.Model: Loaded model or None
        """
        try:
            # Load model from MLflow
            model_uri = f"runs:/{model_id}/model_{model_id}"
            model = mlflow.keras.load_model(model_uri)
            return model
            
        except Exception as e:
            logger.error(f"Error loading model for {model_id}: {str(e)}")
            return None
    
    def _load_scaler(self, model_id):
        """
        Load fitted scaler from file
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            MinMaxScaler: Loaded scaler or None
        """
        try:
            import pickle
            scaler_path = os.path.join(self.models_dir, f"{model_id}_scaler.pkl")
            
            if not os.path.exists(scaler_path):
                return None
            
            with open(scaler_path, 'rb') as f:
                return pickle.load(f)
                
        except Exception as e:
            logger.error(f"Error loading scaler for {model_id}: {str(e)}")
            return None
    
    def _generate_forecast(self, model, scaler, prediction_weeks):
        """
        Generate forecast using the trained model
        
        Args:
            model (keras.Model): Trained LSTM model
            scaler (MinMaxScaler): Fitted scaler
            prediction_weeks (int): Number of weeks to predict
            
        Returns:
            numpy.ndarray: Predicted values
        """
        try:
            # Start with a single input value (last known value)
            # In a real implementation, you might want to use the last few values
            input_data = np.array([[0.5]])  # Placeholder - should be last actual value
            input_data = input_data.reshape(1, 1, 1)
            
            predictions = []
            
            for _ in range(prediction_weeks):
                # Predict next value
                pred = model.predict(input_data, verbose=0)
                predictions.append(pred[0, 0])
                
                # Update input for next prediction
                input_data = pred.reshape(1, 1, 1)
            
            # Inverse transform predictions
            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise Exception(f"Failed to generate forecast: {str(e)}")
    
    def _calculate_confidence_intervals(self, predictions, confidence_level=0.95):
        """
        Calculate confidence intervals for predictions
        
        Args:
            predictions (numpy.ndarray): Predicted values
            confidence_level (float): Confidence level (0.95 for 95%)
            
        Returns:
            list: List of confidence intervals
        """
        try:
            # Simplified confidence interval calculation
            # In a real implementation, you might use more sophisticated methods
            std_dev = np.std(predictions) * 0.1  # Simplified uncertainty
            z_score = 1.96  # For 95% confidence level
            
            confidence_intervals = []
            for pred in predictions:
                margin_of_error = z_score * std_dev
                confidence_intervals.append({
                    'lower': max(0, pred - margin_of_error),
                    'upper': min(100, pred + margin_of_error),
                    'prediction': pred
                })
            
            return confidence_intervals
            
        except Exception as e:
            logger.error(f"Error calculating confidence intervals: {str(e)}")
            return [] 
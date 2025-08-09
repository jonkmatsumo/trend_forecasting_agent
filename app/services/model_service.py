"""
Model Service for Google Trends Quantile Forecaster
Handles LSTM model training, saving, and management
"""

import os
import uuid
import mlflow
import mlflow.keras
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from flask import current_app
import logging

logger = logging.getLogger(__name__)


class ModelService:
    """Service for handling LSTM model operations"""
    
    def __init__(self):
        """Initialize the model service"""
        self.models_dir = current_app.config.get('MODELS_DIR', 'models')
        self.data_dir = current_app.config.get('DATA_DIR', 'data')
        
        # Ensure directories exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri(current_app.config.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db'))
        mlflow.set_experiment(current_app.config.get('MLFLOW_EXPERIMENT_NAME', 'google_trends_forecaster'))
    
    def train_model(self, time_series_data, keyword, model_params=None):
        """
        Train a new LSTM model with provided time series data
        
        Args:
            time_series_data (list): List of interest values
            keyword (str): Keyword associated with the model
            model_params (dict): Model training parameters
            
        Returns:
            dict: Model information including model_id and metrics
        """
        try:
            logger.info(f"Starting model training for keyword: {keyword}")
            
            # Generate unique model ID
            model_id = str(uuid.uuid4())
            
            # Set default parameters
            default_params = {
                'batch_size': current_app.config.get('DEFAULT_BATCH_SIZE', 5),
                'epochs': current_app.config.get('DEFAULT_EPOCHS', 150),
                'lstm_units': current_app.config.get('DEFAULT_LSTM_UNITS', 4),
                'optimizer': 'adam',
                'loss': 'mean_squared_error'
            }
            
            # Update with provided parameters
            if model_params:
                default_params.update(model_params)
            
            # Prepare data
            training_data, scaler = self._prepare_training_data(time_series_data)
            
            # Create and train model
            model = self._create_lstm_model(default_params['lstm_units'])
            
            # Train the model
            history = model.fit(
                training_data['X_train'],
                training_data['y_train'],
                batch_size=default_params['batch_size'],
                epochs=default_params['epochs'],
                verbose=1
            )
            
            # Log model with MLflow
            with mlflow.start_run() as run:
                # Log parameters
                mlflow.log_params(default_params)
                mlflow.log_param("keyword", keyword)
                mlflow.log_param("model_id", model_id)
                
                # Log metrics
                final_loss = history.history['loss'][-1]
                mlflow.log_metric("final_loss", final_loss)
                
                # Log model
                mlflow.keras.log_model(model, f"model_{model_id}")
                
                # Save model metadata
                model_info = {
                    'model_id': model_id,
                    'keyword': keyword,
                    'run_id': run.info.run_id,
                    'metrics': {
                        'final_loss': final_loss,
                        'epochs_trained': len(history.history['loss'])
                    },
                    'parameters': default_params,
                    'data_points': len(time_series_data),
                    'created_at': pd.Timestamp.now().isoformat()
                }
                
                # Save scaler and model info
                self._save_model_artifacts(model_id, scaler, model_info)
            
            logger.info(f"Model training completed for keyword: {keyword}, model_id: {model_id}")
            
            return {
                'model_id': model_id,
                'keyword': keyword,
                'metrics': model_info['metrics'],
                'run_id': run.info.run_id
            }
            
        except Exception as e:
            logger.error(f"Error training model for {keyword}: {str(e)}")
            raise Exception(f"Failed to train model: {str(e)}")
    
    def get_model_info(self, model_id):
        """
        Get information about a specific model
        
        Args:
            model_id (str): Model identifier
            
        Returns:
            dict: Model information or None if not found
        """
        try:
            # Load model info from file
            model_info_path = os.path.join(self.models_dir, f"{model_id}_info.json")
            
            if not os.path.exists(model_info_path):
                logger.warning(f"Model info not found for model_id: {model_id}")
                return None
            
            import json
            with open(model_info_path, 'r') as f:
                model_info = json.load(f)
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error retrieving model info for {model_id}: {str(e)}")
            return None
    
    def list_models(self):
        """
        List all available models
        
        Returns:
            list: List of model information dictionaries
        """
        try:
            models = []
            
            # Scan models directory for info files
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_info.json'):
                    model_id = filename.replace('_info.json', '')
                    model_info = self.get_model_info(model_id)
                    if model_info:
                        models.append(model_info)
            
            # Sort by creation date (newest first)
            models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            
            return models
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            return []
    
    def _prepare_training_data(self, time_series_data):
        """
        Prepare training data for LSTM model
        
        Args:
            time_series_data (list): Raw time series data
            
        Returns:
            tuple: (training_data_dict, scaler)
        """
        # Convert to numpy array
        data = np.array(time_series_data).reshape(-1, 1)
        
        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - 1):
            X.append(scaled_data[i])
            y.append(scaled_data[i + 1])
        
        X = np.array(X)
        y = np.array(y)
        
        # Reshape for LSTM input (samples, time steps, features)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        
        return {
            'X_train': X,
            'y_train': y
        }, scaler
    
    def _create_lstm_model(self, lstm_units):
        """
        Create LSTM model architecture
        
        Args:
            lstm_units (int): Number of LSTM units
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        model = Sequential()
        model.add(LSTM(units=lstm_units, activation='sigmoid', input_shape=(None, 1)))
        model.add(Dense(units=1))
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['accuracy']
        )
        
        return model
    
    def _save_model_artifacts(self, model_id, scaler, model_info):
        """
        Save model artifacts (scaler and info)
        
        Args:
            model_id (str): Model identifier
            scaler (MinMaxScaler): Fitted scaler
            model_info (dict): Model information
        """
        import pickle
        import json
        
        # Save scaler
        scaler_path = os.path.join(self.models_dir, f"{model_id}_scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save model info
        info_path = os.path.join(self.models_dir, f"{model_id}_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2) 
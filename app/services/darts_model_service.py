"""
Darts Model Service for training, evaluating, and managing time series forecasting models.
"""

import os
import logging
import time
import json
import pickle
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import (
    RNNModel, TCNModel, TransformerModel, NBEATSModel, TFTModel,
    ARIMA, ExponentialSmoothing, Prophet, RandomForest,
    AutoARIMA, StatsForecastAutoETS, StatsForecastAutoTheta, StatsForecastAutoCES
)
from darts.metrics import mae, rmse, mape
from darts.utils.statistics import check_seasonality
from sklearn.preprocessing import MinMaxScaler

from app.models.darts_models import (
    ModelType, DartsTimeSeriesData, ModelTrainingRequest,
    ModelEvaluationMetrics, ForecastResult, DEFAULT_MODEL_PARAMETERS,
    generate_model_id
)
from app.models.prediction_model import ModelMetadata
from app.utils.error_handlers import ModelError, ValidationError


class DartsModelService:
    """Service for managing Darts time series forecasting models."""
    
    def __init__(self, models_dir: str = "models", mlflow_tracking_uri: str = "sqlite:///mlflow.db"):
        """Initialize the Darts model service.
        
        Args:
            models_dir: Directory to store trained models
            mlflow_tracking_uri: MLflow tracking URI for experiment tracking
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Setup MLflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        self.experiment_name = "google_trends_forecaster"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model type to Darts class mapping
        self.model_mapping = {
            "lstm": RNNModel,  # RNNModel supports LSTM
            "gru": RNNModel,   # RNNModel supports GRU
            "tcn": TCNModel,
            "transformer": TransformerModel,
            "n_beats": NBEATSModel,
            "tft": TFTModel,
            "arima": ARIMA,
            "exponential_smoothing": ExponentialSmoothing,
            "prophet": Prophet,
            "random_forest": RandomForest,
            "auto_arima": AutoARIMA,
            "auto_ets": StatsForecastAutoETS,
            "auto_theta": StatsForecastAutoTheta,
            "auto_ces": StatsForecastAutoCES
        }
    
    def train_model(self, request: ModelTrainingRequest) -> Tuple[str, ModelEvaluationMetrics]:
        """Train a Darts model with the provided data and parameters.
        
        Args:
            request: Model training request containing data and parameters
            
        Returns:
            Tuple of (model_id, evaluation_metrics)
            
        Raises:
            ValidationError: If request validation fails
            ModelError: If model training fails
        """
        try:
            # Validate request
            self._validate_training_request(request)
            
            # Generate model ID
            model_id = generate_model_id()
            
            # Create TimeSeries object
            time_series = self._create_time_series(request)
            
            # Split data for training and testing
            train_series, test_series = self._split_time_series(
                time_series, request.train_test_split
            )
            
            # Create and train model
            model = self._create_model(request.model_type, request.model_parameters)
            
            # Train the model
            start_time = time.time()
            model.fit(train_series)
            training_time = time.time() - start_time
            
            # Evaluate the model
            evaluation_metrics = self._evaluate_model(
                model, train_series, test_series, training_time, model_id, request.keyword, request.model_type
            )
            
            # Save the model
            self._save_model(model, model_id, request, evaluation_metrics)
            
            # Log to MLflow
            self._log_to_mlflow(model_id, request, evaluation_metrics, training_time)
            
            self.logger.info(f"Successfully trained {request.model_type} model: {model_id}")
            
            return model_id, evaluation_metrics
            
        except Exception as e:
            self.logger.error(f"Model training failed: {str(e)}")
            raise ModelError(f"Model training failed: {str(e)}")
    
    def _validate_training_request(self, request: ModelTrainingRequest) -> None:
        """Validate the training request.
        
        Args:
            request: Model training request to validate
            
        Raises:
            ValidationError: If validation fails
        """
        if not request.keyword or not request.keyword.strip():
            raise ValidationError("Keyword cannot be empty")
        
        if len(request.time_series_data) < 52:
            raise ValidationError("At least 52 data points required for Darts models")
        
        if not request.dates:
            raise ValidationError("Dates are required for Darts models")
        
        if len(request.dates) != len(request.time_series_data):
            raise ValidationError("Dates and time_series_data must have same length")
        
        if not 0.1 <= request.train_test_split <= 0.9:
            raise ValidationError("train_test_split must be between 0.1 and 0.9")
        
        if request.forecast_horizon <= 0:
            raise ValidationError("forecast_horizon must be positive")
        
        valid_strategies = ["holdout", "expanding_window", "rolling_window"]
        if request.validation_strategy not in valid_strategies:
            raise ValidationError(f"validation_strategy must be one of: {valid_strategies}")
    
    def _create_time_series(self, request: ModelTrainingRequest) -> TimeSeries:
        """Create a Darts TimeSeries object from the request data.
        
        Args:
            request: Model training request
            
        Returns:
            Darts TimeSeries object
        """
        # Convert dates to datetime objects
        dates = [datetime.fromisoformat(d) for d in request.dates]
        
        # Create pandas Series with datetime index
        series_data = pd.Series(
            request.time_series_data,
            index=pd.DatetimeIndex(dates)
        )
        
        # Create TimeSeries object
        time_series = TimeSeries.from_series(series_data)
        
        return time_series
    
    def _split_time_series(self, time_series: TimeSeries, split_ratio: float) -> Tuple[TimeSeries, TimeSeries]:
        """Split time series into training and testing sets.
        
        Args:
            time_series: Full time series data
            split_ratio: Ratio for train/test split
            
        Returns:
            Tuple of (train_series, test_series)
        """
        split_point = int(len(time_series) * split_ratio)
        train_series = time_series[:split_point]
        test_series = time_series[split_point:]
        
        return train_series, test_series
    
    def _create_model(self, model_type: str, parameters: Dict[str, Any]):
        """Create a Darts model instance.
        
        Args:
            model_type: Type of model to create
            parameters: Model parameters
            
        Returns:
            Darts model instance
            
        Raises:
            ValidationError: If model type is not supported
        """
        try:
            model_class = self.model_mapping.get(model_type)
            if not model_class:
                raise ValidationError(f"Unsupported model type: {model_type}")
            
            # Get default parameters for this model type
            default_params = DEFAULT_MODEL_PARAMETERS.get(model_type, {})
            
            # Merge default parameters with provided parameters
            model_params = {**default_params, **parameters}
            
            # Handle special cases for RNNModel (LSTM/GRU)
            if model_type in ["lstm", "gru"] and model_class == RNNModel:
                # Add model type parameter for RNNModel
                model_params["model"] = model_type.upper()
            
            # Create model instance
            model = model_class(**model_params)
            
            return model
            
        except Exception as e:
            raise ModelError(f"Failed to create model {model_type}: {str(e)}")

    def _evaluate_model(self, model, train_series: TimeSeries, test_series: TimeSeries,
                       training_time: float, model_id: str, keyword: str, model_type: str) -> ModelEvaluationMetrics:
        """Evaluate a trained model.
        
        Args:
            model: Trained Darts model
            train_series: Training data
            test_series: Test data
            training_time: Time taken for training
            
        Returns:
            Model evaluation metrics
        """
        try:
            # Generate predictions on test set
            predictions = model.predict(len(test_series))
            
            # Calculate metrics
            mae_score = mae(test_series, predictions)
            rmse_score = rmse(test_series, predictions)
            mape_score = mape(test_series, predictions)
            
            # Calculate directional accuracy
            directional_accuracy = self._calculate_directional_accuracy(test_series, predictions)
            
            # Calculate confidence interval coverage (if available)
            coverage_95 = self._calculate_confidence_coverage(test_series, predictions)
            
            # Create evaluation metrics
            metrics = ModelEvaluationMetrics(
                model_id=model_id,
                keyword=keyword,
                model_type=ModelType(model_type),
                train_mae=mae_score,  # Using test metrics for both for now
                train_rmse=rmse_score,
                train_mape=mape_score,
                test_mae=mae_score,
                test_rmse=rmse_score,
                test_mape=mape_score,
                directional_accuracy=directional_accuracy / 100.0,  # Convert to 0-1 range
                coverage_95=coverage_95 / 100.0,  # Convert to 0-1 range
                train_samples=len(train_series),
                test_samples=len(test_series),
                total_samples=len(train_series) + len(test_series),
                training_time_seconds=training_time
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Model evaluation failed: {str(e)}")
            raise ModelError(f"Model evaluation failed: {str(e)}")
    
    def _calculate_directional_accuracy(self, actual: TimeSeries, predicted: TimeSeries) -> float:
        """Calculate directional accuracy of predictions.
        
        Args:
            actual: Actual time series values
            predicted: Predicted time series values
            
        Returns:
            Directional accuracy as a percentage
        """
        try:
            # Get values as numpy arrays
            actual_values = actual.values()
            predicted_values = predicted.values()
            
            # Calculate direction changes
            actual_direction = np.diff(actual_values.flatten())
            predicted_direction = np.diff(predicted_values.flatten())
            
            # Count correct directions
            correct_directions = np.sum(
                (actual_direction > 0) == (predicted_direction > 0)
            )
            
            # Calculate accuracy
            total_changes = len(actual_direction)
            if total_changes == 0:
                return 0.0
            
            accuracy = (correct_directions / total_changes) * 100
            return min(100.0, max(0.0, accuracy))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate directional accuracy: {str(e)}")
            return 0.0
    
    def _calculate_confidence_coverage(self, actual: TimeSeries, predicted: TimeSeries) -> float:
        """Calculate confidence interval coverage.
        
        Args:
            actual: Actual time series values
            predicted: Predicted time series values
            
        Returns:
            Coverage percentage (0.0 if not available)
        """
        try:
            # For now, return 0.0 as most models don't provide confidence intervals by default
            # This will be enhanced when we implement probabilistic forecasting
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate confidence coverage: {str(e)}")
            return 0.0
    
    def _save_model(self, model, model_id: str, request: ModelTrainingRequest, 
                   metrics: ModelEvaluationMetrics) -> None:
        """Save the trained model and metadata.
        
        Args:
            model: Trained Darts model
            model_id: Unique model identifier
            request: Original training request
            metrics: Model evaluation metrics
        """
        try:
            # Create model directory
            model_dir = self.models_dir / model_id
            model_dir.mkdir(exist_ok=True)
            
            # Save the model
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Create and save metadata
            metadata = ModelMetadata(
                keyword=request.keyword,
                training_date=datetime.now(),
                parameters=request.model_parameters,
                metrics={
                    "test_mae": metrics.test_mae,
                    "test_rmse": metrics.test_rmse,
                    "test_mape": metrics.test_mape,
                    "directional_accuracy": metrics.directional_accuracy,
                    "coverage_95": metrics.coverage_95,
                    "training_time_seconds": metrics.training_time_seconds
                },
                model_id=model_id,
                model_path=str(model_path),
                model_type=request.model_type.value,
                darts_model_path=str(model_path),
                status="completed",
                data_points=len(request.time_series_data)
            )
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2, default=str)
            
            # Save evaluation metrics
            metrics_path = model_dir / "evaluation.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2, default=str)
            
            # Save training request for reference
            request_path = model_dir / "training_request.json"
            with open(request_path, 'w') as f:
                json.dump(request.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Model {model_id} saved to {model_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save model {model_id}: {str(e)}")
            raise ModelError(f"Failed to save model: {str(e)}")
    
    def _log_to_mlflow(self, model_id: str, request: ModelTrainingRequest, 
                      metrics: ModelEvaluationMetrics, training_time: float) -> None:
        """Log model training to MLflow.
        
        Args:
            model_id: Unique model identifier
            request: Training request
            metrics: Evaluation metrics
            training_time: Training time
        """
        try:
            # Set experiment
            mlflow.set_experiment(self.experiment_name)
            
            with mlflow.start_run(run_name=f"{request.model_type}_{model_id}"):
                # Log parameters
                mlflow.log_param("model_type", request.model_type)
                mlflow.log_param("keyword", request.keyword)
                mlflow.log_param("train_test_split", request.train_test_split)
                mlflow.log_param("forecast_horizon", request.forecast_horizon)
                mlflow.log_param("validation_strategy", request.validation_strategy)
                mlflow.log_param("data_points", len(request.time_series_data))
                
                # Log model parameters
                for key, value in request.model_parameters.items():
                    mlflow.log_param(f"model_{key}", value)
                
                # Log metrics
                mlflow.log_metric("mae", metrics.mae)
                mlflow.log_metric("rmse", metrics.rmse)
                mlflow.log_metric("mape", metrics.mape)
                mlflow.log_metric("directional_accuracy", metrics.directional_accuracy)
                mlflow.log_metric("coverage_95", metrics.coverage_95)
                mlflow.log_metric("training_time", training_time)
                mlflow.log_metric("train_samples", metrics.train_samples)
                mlflow.log_metric("test_samples", metrics.test_samples)
                
                # Log model
                model_path = self.models_dir / model_id / "model.pkl"
                if model_path.exists():
                    mlflow.log_artifact(str(model_path))
                
                # Log metadata
                metadata_path = self.models_dir / model_id / "metadata.json"
                if metadata_path.exists():
                    mlflow.log_artifact(str(metadata_path))
                
                self.logger.info(f"Logged model {model_id} to MLflow")
                
        except Exception as e:
            self.logger.warning(f"Failed to log to MLflow: {str(e)}")
            # Don't raise error as MLflow logging is not critical
    
    def load_model(self, model_id: str):
        """Load a trained model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Trained Darts model
            
        Raises:
            ModelError: If model loading fails
        """
        try:
            model_path = self.models_dir / model_id / "model.pkl"
            
            if not model_path.exists():
                raise ModelError(f"Model {model_id} not found")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {str(e)}")
            raise ModelError(f"Failed to load model: {str(e)}")
    
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Get model metadata.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Model metadata
            
        Raises:
            ModelError: If metadata loading fails
        """
        try:
            metadata_path = self.models_dir / model_id / "metadata.json"
            
            if not metadata_path.exists():
                raise ModelError(f"Model metadata {model_id} not found")
            
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            
            return ModelMetadata.from_dict(metadata_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to load model metadata {model_id}: {str(e)}")
            raise ModelError(f"Failed to load model metadata: {str(e)}")
    
    def get_evaluation_metrics(self, model_id: str) -> ModelEvaluationMetrics:
        """Get model evaluation metrics.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Model evaluation metrics
            
        Raises:
            ModelError: If metrics loading fails
        """
        try:
            metrics_path = self.models_dir / model_id / "evaluation.json"
            
            if not metrics_path.exists():
                raise ModelError(f"Model evaluation metrics {model_id} not found")
            
            with open(metrics_path, 'r') as f:
                metrics_dict = json.load(f)

            # Convert from nested dict structure to flat parameters
            return ModelEvaluationMetrics(
                model_id=metrics_dict["model_id"],
                keyword=metrics_dict["keyword"],
                model_type=ModelType(metrics_dict["model_type"]),
                train_mae=metrics_dict["train_metrics"]["mae"],
                train_rmse=metrics_dict["train_metrics"]["rmse"],
                train_mape=metrics_dict["train_metrics"]["mape"],
                test_mae=metrics_dict["test_metrics"]["mae"],
                test_rmse=metrics_dict["test_metrics"]["rmse"],
                test_mape=metrics_dict["test_metrics"]["mape"],
                directional_accuracy=metrics_dict["test_metrics"]["directional_accuracy"],
                coverage_95=metrics_dict["test_metrics"]["coverage_95"],
                train_samples=metrics_dict["data_info"]["train_samples"],
                test_samples=metrics_dict["data_info"]["test_samples"],
                total_samples=metrics_dict["data_info"]["total_samples"],
                training_time_seconds=metrics_dict["training_info"]["training_time_seconds"],
                mlflow_run_id=metrics_dict["training_info"].get("mlflow_run_id"),
                created_at=datetime.fromisoformat(metrics_dict["training_info"]["created_at"])
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation metrics {model_id}: {str(e)}")
            raise ModelError(f"Failed to load evaluation metrics: {str(e)}")
    
    def list_models(self) -> List[ModelMetadata]:
        """List all available models.
        
        Returns:
            List of model metadata
        """
        models = []
        
        try:
            for model_dir in self.models_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r') as f:
                                metadata_dict = json.load(f)
                            metadata = ModelMetadata.from_dict(metadata_dict)
                            models.append(metadata)
                        except Exception as e:
                            self.logger.warning(f"Failed to load metadata from {model_dir}: {str(e)}")
                            continue
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {str(e)}")
            return []
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model and its associated files.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            model_dir = self.models_dir / model_id
            
            if not model_dir.exists():
                return False
            
            # Remove all files in the model directory
            for file_path in model_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
            
            # Remove the directory
            model_dir.rmdir()
            
            self.logger.info(f"Successfully deleted model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_id}: {str(e)}")
            return False 
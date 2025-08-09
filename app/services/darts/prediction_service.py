"""
Darts Prediction Service for generating forecasts and confidence intervals.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from darts import TimeSeries
# Note: Darts doesn't have a direct ProbabilisticForecastingModel base class
# We'll use a different approach to check for probabilistic models

from app.models.darts.darts_models import (
    ForecastResult, ModelEvaluationMetrics, ModelType
)
from app.services.darts.training_service import TrainingService
from app.utils.error_handlers import ModelError, ValidationError


class PredictionService:
    """Service for generating predictions using trained Darts models."""
    
    def __init__(self, model_service: TrainingService):
        """Initialize the Darts prediction service.
        
        Args:
            model_service: Darts model service for loading models
        """
        self.model_service = model_service
        self.logger = logging.getLogger(__name__)
    
    def generate_forecast(self, model_id: str, forecast_horizon: int = 25,
                         include_confidence_intervals: bool = True) -> ForecastResult:
        """Generate a forecast using a trained model.
        
        Args:
            model_id: Unique model identifier
            forecast_horizon: Number of periods to forecast
            include_confidence_intervals: Whether to include confidence intervals
            
        Returns:
            Forecast result with predictions and metrics
            
        Raises:
            ModelError: If forecast generation fails
            ValidationError: If parameters are invalid
        """
        try:
            # Validate parameters
            if forecast_horizon <= 0:
                raise ValidationError("Forecast horizon must be positive")
            
            if forecast_horizon > 100:
                raise ValidationError("Forecast horizon cannot exceed 100 periods")
            
            # Load model and metadata
            model = self.model_service.load_model(model_id)
            metadata = self.model_service.get_model_metadata(model_id)
            evaluation_metrics = self.model_service.get_evaluation_metrics(model_id)
            
            # Generate forecast
            start_time = time.time()
            
            # Check if model supports probabilistic forecasting
            has_probabilistic_methods = hasattr(model, 'predict') and hasattr(model, 'quantile')
            if include_confidence_intervals and has_probabilistic_methods:
                # Generate probabilistic forecast with confidence intervals
                forecast = model.predict(
                    n=forecast_horizon,
                    num_samples=1000  # Number of samples for confidence intervals
                )
                
                # Extract point forecasts and confidence intervals
                point_forecast = forecast.mean()
                confidence_intervals = self._extract_confidence_intervals(forecast)
                
            else:
                # Generate point forecast only
                forecast = model.predict(n=forecast_horizon)
                point_forecast = forecast
                confidence_intervals = None
            
            prediction_time = time.time() - start_time
            
            # Create forecast dates
            training_date = datetime.fromisoformat(metadata["training_date"])
            forecast_dates = self._generate_forecast_dates(
                training_date, forecast_horizon
            )
            
            # Create forecast result
            result = ForecastResult(
                model_id=model_id,
                keyword=metadata["keyword"],
                forecast_horizon=forecast_horizon,
                forecast_values=point_forecast.values().flatten().tolist(),
                forecast_dates=forecast_dates,  # Already datetime objects
                confidence_intervals=confidence_intervals or {},
                model_metrics=evaluation_metrics,
                generated_at=datetime.now()
            )
            
            self.logger.info(f"Generated forecast for model {model_id}: {forecast_horizon} periods")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Forecast generation failed for model {model_id}: {str(e)}")
            raise ModelError(f"Forecast generation failed: {str(e)}")
    
    def _extract_confidence_intervals(self, forecast: TimeSeries) -> List[Dict[str, float]]:
        """Extract confidence intervals from probabilistic forecast.
        
        Args:
            forecast: Probabilistic forecast with multiple samples
            
        Returns:
            List of confidence intervals for each forecast period
        """
        try:
            # Get quantiles for confidence intervals
            quantiles = forecast.quantile_df([0.025, 0.975])  # 95% confidence interval
            
            confidence_intervals = []
            
            for i in range(len(quantiles)):
                interval = {
                    "lower": float(quantiles.iloc[i, 0]),  # 2.5th percentile
                    "upper": float(quantiles.iloc[i, 1])   # 97.5th percentile
                }
                confidence_intervals.append(interval)
            
            return confidence_intervals
            
        except Exception as e:
            self.logger.warning(f"Failed to extract confidence intervals: {str(e)}")
            return []
    
    def _generate_forecast_dates(self, training_date: datetime, 
                                forecast_horizon: int) -> List[datetime]:
        """Generate forecast dates based on training date.
        
        Args:
            training_date: Date when model was trained
            forecast_horizon: Number of periods to forecast
            
        Returns:
            List of forecast dates
        """
        # Assume weekly data (7 days per period)
        # This can be made configurable based on the data frequency
        dates = []
        current_date = training_date + timedelta(days=7)  # Start from next week
        
        for i in range(forecast_horizon):
            dates.append(current_date)
            current_date += timedelta(days=7)
        
        return dates
    
    def compare_models(self, model_ids: List[str], forecast_horizon: int = 25,
                      include_confidence_intervals: bool = True) -> Dict[str, Any]:
        """Compare multiple models by generating forecasts.
        
        Args:
            model_ids: List of model identifiers to compare
            forecast_horizon: Number of periods to forecast
            include_confidence_intervals: Whether to include confidence intervals
            
        Returns:
            Comparison results with forecasts and metrics
            
        Raises:
            ValidationError: If parameters are invalid
            ModelError: If comparison fails
        """
        try:
            # Validate parameters
            if not model_ids:
                raise ValidationError("At least one model ID is required")
            
            if len(model_ids) > 10:
                raise ValidationError("Cannot compare more than 10 models at once")
            
            if forecast_horizon <= 0:
                raise ValidationError("Forecast horizon must be positive")
            
            comparison_results = {
                "comparison_date": datetime.now().isoformat(),
                "forecast_horizon": forecast_horizon,
                "models": {},
                "summary": {}
            }
            
            # Generate forecasts for each model
            for model_id in model_ids:
                try:
                    # Get model metadata and evaluation metrics
                    metadata = self.model_service.get_model_metadata(model_id)
                    evaluation_metrics = self.model_service.get_evaluation_metrics(model_id)
                    
                    # Generate forecast
                    forecast_result = self.generate_forecast(
                        model_id, forecast_horizon, include_confidence_intervals
                    )
                    
                    # Store results
                    comparison_results["models"][model_id] = {
                        "metadata": metadata,  # Already a dictionary
                        "evaluation_metrics": evaluation_metrics.to_dict(),
                        "forecast": forecast_result.to_dict()
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to generate forecast for model {model_id}: {str(e)}")
                    comparison_results["models"][model_id] = {
                        "error": str(e)
                    }
            
            # Generate comparison summary
            comparison_results["summary"] = self._generate_comparison_summary(
                comparison_results["models"]
            )
            
            self.logger.info(f"Completed model comparison for {len(model_ids)} models")
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {str(e)}")
            raise ModelError(f"Model comparison failed: {str(e)}")
    
    def _generate_comparison_summary(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for model comparison.
        
        Args:
            models_data: Dictionary containing model forecasts and metrics
            
        Returns:
            Summary statistics
        """
        try:
            summary = {
                "total_models": len(models_data),
                "successful_models": 0,
                "failed_models": 0,
                "best_model_by_mae": None,
                "best_model_by_rmse": None,
                "best_model_by_mape": None,
                "best_model_by_directional_accuracy": None,
                "average_metrics": {},
                "model_rankings": {}
            }
            
            successful_models = []
            
            # Process each model
            for model_id, data in models_data.items():
                if "error" not in data:
                    summary["successful_models"] += 1
                    successful_models.append((model_id, data))
                else:
                    summary["failed_models"] += 1
            
            if not successful_models:
                return summary
            
            # Calculate rankings and best models
            metrics_data = []
            
            for model_id, data in successful_models:
                metrics = data["evaluation_metrics"]
                metrics_data.append({
                    "model_id": model_id,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "mape": metrics["mape"],
                    "directional_accuracy": metrics["directional_accuracy"]
                })
            
            # Find best models by each metric
            if metrics_data:
                best_mae = min(metrics_data, key=lambda x: x["mae"])
                best_rmse = min(metrics_data, key=lambda x: x["rmse"])
                best_mape = min(metrics_data, key=lambda x: x["mape"])
                best_directional = max(metrics_data, key=lambda x: x["directional_accuracy"])
                
                summary["best_model_by_mae"] = best_mae["model_id"]
                summary["best_model_by_rmse"] = best_rmse["model_id"]
                summary["best_model_by_mape"] = best_mape["model_id"]
                summary["best_model_by_directional_accuracy"] = best_directional["model_id"]
                
                # Calculate average metrics
                avg_mae = np.mean([m["mae"] for m in metrics_data])
                avg_rmse = np.mean([m["rmse"] for m in metrics_data])
                avg_mape = np.mean([m["mape"] for m in metrics_data])
                avg_directional = np.mean([m["directional_accuracy"] for m in metrics_data])
                
                summary["average_metrics"] = {
                    "mae": avg_mae,
                    "rmse": avg_rmse,
                    "mape": avg_mape,
                    "directional_accuracy": avg_directional
                }
                
                # Generate rankings
                summary["model_rankings"] = {
                    "by_mae": [m["model_id"] for m in sorted(metrics_data, key=lambda x: x["mae"])],
                    "by_rmse": [m["model_id"] for m in sorted(metrics_data, key=lambda x: x["rmse"])],
                    "by_mape": [m["model_id"] for m in sorted(metrics_data, key=lambda x: x["mape"])],
                    "by_directional_accuracy": [m["model_id"] for m in sorted(metrics_data, key=lambda x: x["directional_accuracy"], reverse=True)]
                }
            
            return summary
            
        except Exception as e:
            self.logger.warning(f"Failed to generate comparison summary: {str(e)}")
            return {"error": str(e)}
    
    def get_forecast_accuracy_report(self, model_id: str, 
                                   actual_values: Optional[List[float]] = None,
                                   actual_dates: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate accuracy report for a forecast.
        
        Args:
            model_id: Unique model identifier
            actual_values: Actual values for comparison (optional)
            actual_dates: Actual dates for comparison (optional)
            
        Returns:
            Accuracy report with metrics and analysis
            
        Raises:
            ModelError: If accuracy calculation fails
        """
        try:
            # Get model evaluation metrics
            evaluation_metrics = self.model_service.get_evaluation_metrics(model_id)
            metadata = self.model_service.get_model_metadata(model_id)
            
            report = {
                "model_id": model_id,
                "keyword": metadata["keyword"],
                "model_type": metadata["model_type"],
                "training_date": metadata["training_date"],
                "evaluation_metrics": evaluation_metrics.to_dict(),
                "forecast_accuracy": {},
                "recommendations": []
            }
            
            # Calculate forecast accuracy if actual values are provided
            if actual_values and actual_dates:
                forecast_result = self.generate_forecast(model_id, len(actual_values))
                
                # Calculate accuracy metrics
                accuracy_metrics = self._calculate_forecast_accuracy(
                    actual_values, forecast_result.forecast_values
                )
                
                report["forecast_accuracy"] = accuracy_metrics
                
                # Generate recommendations
                report["recommendations"] = self._generate_accuracy_recommendations(
                    accuracy_metrics, evaluation_metrics
                )
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate accuracy report for model {model_id}: {str(e)}")
            raise ModelError(f"Failed to generate accuracy report: {str(e)}")
    
    def _calculate_forecast_accuracy(self, actual: List[float], 
                                   predicted: List[float]) -> Dict[str, float]:
        """Calculate accuracy metrics between actual and predicted values.
        
        Args:
            actual: Actual values
            predicted: Predicted values
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            if len(actual) != len(predicted):
                raise ValueError("Actual and predicted values must have same length")
            
            if not actual or not predicted:
                raise ValueError("Arrays cannot be empty")
            
            actual_array = np.array(actual)
            predicted_array = np.array(predicted)
            
            # Calculate metrics
            mae = np.mean(np.abs(actual_array - predicted_array))
            rmse = np.sqrt(np.mean((actual_array - predicted_array) ** 2))
            
            # Calculate MAPE (handle zero values)
            mape = np.mean(np.abs((actual_array - predicted_array) / np.where(actual_array != 0, actual_array, 1))) * 100
            
            # Calculate directional accuracy
            actual_direction = np.diff(actual_array)
            predicted_direction = np.diff(predicted_array)
            
            correct_directions = np.sum(
                (actual_direction > 0) == (predicted_direction > 0)
            )
            directional_accuracy = (correct_directions / len(actual_direction)) * 100 if len(actual_direction) > 0 else 0
            
            return {
                "mae": float(mae),
                "rmse": float(rmse),
                "mape": float(mape),
                "directional_accuracy": float(directional_accuracy)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate forecast accuracy: {str(e)}")
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "mape": 0.0,
                "directional_accuracy": 0.0,
                "error": str(e)
            }
    
    def _generate_accuracy_recommendations(self, forecast_accuracy: Dict[str, float],
                                         evaluation_metrics: ModelEvaluationMetrics) -> List[str]:
        """Generate recommendations based on accuracy metrics.
        
        Args:
            forecast_accuracy: Forecast accuracy metrics
            evaluation_metrics: Model evaluation metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Compare forecast accuracy with training accuracy
            if forecast_accuracy.get("mae", 0) > evaluation_metrics.mae * 1.5:
                recommendations.append("Forecast MAE is significantly higher than training MAE. Consider retraining the model.")
            
            if forecast_accuracy.get("directional_accuracy", 0) < 50:
                recommendations.append("Directional accuracy is below 50%. The model may not be capturing trend changes well.")
            
            if evaluation_metrics.mape > 20:
                recommendations.append("Training MAPE is high (>20%). Consider using a different model type or more data.")
            
            if evaluation_metrics.directional_accuracy < 60:
                recommendations.append("Training directional accuracy is low. Consider using models that better capture trends.")
            
            # Add general recommendations
            if not recommendations:
                recommendations.append("Model performance looks good. Continue monitoring for any degradation.")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {str(e)}")
            return ["Unable to generate recommendations due to calculation error."] 
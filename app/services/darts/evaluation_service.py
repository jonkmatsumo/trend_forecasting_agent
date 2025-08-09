"""
Darts Evaluation Service for comprehensive model evaluation and benchmarking.
"""

import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.metrics import mae, rmse, mape, smape, mase
from darts.utils.statistics import check_seasonality, plot_acf, plot_pacf

from app.models.darts.darts_models import (
    ModelEvaluationMetrics, ModelType, ModelTrainingRequest
)
from app.services.darts.training_service import TrainingService
from app.utils.error_handlers import ModelError, ValidationError


class EvaluationService:
    """Service for comprehensive model evaluation and benchmarking."""
    
    def __init__(self, model_service: TrainingService):
        """Initialize the Darts evaluation service.
        
        Args:
            model_service: Darts model service for accessing models
        """
        self.model_service = model_service
        self.logger = logging.getLogger(__name__)
    
    def evaluate_model_comprehensive(self, model_id: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of a trained model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Comprehensive evaluation results
            
        Raises:
            ModelError: If evaluation fails
        """
        try:
            # Get model metadata and evaluation metrics
            metadata = self.model_service.get_model_metadata(model_id)
            evaluation_metrics = self.model_service.get_evaluation_metrics(model_id)
            
            # Load the model
            model = self.model_service.load_model(model_id)
            
            # Get training request for additional analysis
            training_request = self._load_training_request(model_id)
            
            evaluation_results = {
                "model_id": model_id,
                "evaluation_date": datetime.now().isoformat(),
                "metadata": metadata,  # Already a dictionary
                "basic_metrics": evaluation_metrics.to_dict(),
                "detailed_analysis": {},
                "performance_benchmarks": {},
                "recommendations": []
            }
            
            # Perform detailed analysis
            if training_request:
                evaluation_results["detailed_analysis"] = self._perform_detailed_analysis(
                    model, training_request, evaluation_metrics
                )
            
            # Generate performance benchmarks
            evaluation_results["performance_benchmarks"] = self._generate_performance_benchmarks(
                evaluation_metrics, metadata["model_type"]
            )
            
            # Generate recommendations
            evaluation_results["recommendations"] = self._generate_evaluation_recommendations(
                evaluation_metrics, evaluation_results["detailed_analysis"]
            )
            
            self.logger.info(f"Completed comprehensive evaluation for model {model_id}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive evaluation failed for model {model_id}: {str(e)}")
            raise ModelError(f"Comprehensive evaluation failed: {str(e)}")
    
    def _load_training_request(self, model_id: str) -> Optional[ModelTrainingRequest]:
        """Load the original training request for a model.
        
        Args:
            model_id: Unique model identifier
            
        Returns:
            Training request if available, None otherwise
        """
        try:
            import json
            from pathlib import Path
            
            request_path = Path(self.model_service.models_dir) / model_id / "training_request.json"
            
            if request_path.exists():
                with open(request_path, 'r') as f:
                    request_dict = json.load(f)
                
                return ModelTrainingRequest(**request_dict)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Failed to load training request for model {model_id}: {str(e)}")
            return None
    
    def _perform_detailed_analysis(self, model, training_request: ModelTrainingRequest,
                                 evaluation_metrics: ModelEvaluationMetrics) -> Dict[str, Any]:
        """Perform detailed analysis of model performance.
        
        Args:
            model: Trained Darts model
            training_request: Original training request
            evaluation_metrics: Basic evaluation metrics
            
        Returns:
            Detailed analysis results
        """
        analysis = {
            "data_characteristics": {},
            "model_characteristics": {},
            "performance_analysis": {},
            "seasonality_analysis": {},
            "residual_analysis": {}
        }
        
        try:
            # Analyze data characteristics
            analysis["data_characteristics"] = self._analyze_data_characteristics(training_request)
            
            # Analyze model characteristics
            analysis["model_characteristics"] = self._analyze_model_characteristics(model)
            
            # Analyze performance patterns
            analysis["performance_analysis"] = self._analyze_performance_patterns(evaluation_metrics)
            
            # Analyze seasonality
            analysis["seasonality_analysis"] = self._analyze_seasonality(training_request)
            
            # Analyze residuals (if possible)
            analysis["residual_analysis"] = self._analyze_residuals(model, training_request)
            
        except Exception as e:
            self.logger.warning(f"Failed to perform detailed analysis: {str(e)}")
            analysis["error"] = str(e)
        
        return analysis
    
    def _analyze_data_characteristics(self, training_request: ModelTrainingRequest) -> Dict[str, Any]:
        """Analyze characteristics of the training data.
        
        Args:
            training_request: Training request containing data
            
        Returns:
            Data characteristics analysis
        """
        try:
            data = np.array(training_request.time_series_data)
            
            characteristics = {
                "data_points": len(data),
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "range": float(np.max(data) - np.min(data)),
                "coefficient_of_variation": float(np.std(data) / np.mean(data)) if np.mean(data) != 0 else 0,
                "skewness": float(self._calculate_skewness(data)),
                "kurtosis": float(self._calculate_kurtosis(data)),
                "trend_analysis": self._analyze_trend(data),
                "volatility_analysis": self._analyze_volatility(data)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze data characteristics: {str(e)}")
            return {"error": str(e)}
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of the data.
        
        Args:
            data: Input data array
            
        Returns:
            Skewness value
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            
            skewness = np.mean(((data - mean) / std) ** 3)
            return skewness
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of the data.
        
        Args:
            data: Input data array
            
        Returns:
            Kurtosis value
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3
            return kurtosis
            
        except Exception:
            return 0.0
    
    def _analyze_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Trend analysis results
        """
        try:
            # Simple linear trend analysis
            x = np.arange(len(data))
            slope, intercept = np.polyfit(x, data, 1)
            
            # Calculate trend strength
            trend_line = slope * x + intercept
            ss_tot = np.sum((data - np.mean(data)) ** 2)
            ss_res = np.sum((data - trend_line) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
                "trend_strength": float(r_squared),
                "trend_strength_category": self._categorize_trend_strength(r_squared)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _categorize_trend_strength(self, r_squared: float) -> str:
        """Categorize trend strength based on R-squared value.
        
        Args:
            r_squared: R-squared value
            
        Returns:
            Trend strength category
        """
        if r_squared >= 0.7:
            return "strong"
        elif r_squared >= 0.4:
            return "moderate"
        elif r_squared >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def _analyze_volatility(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze volatility in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Volatility analysis results
        """
        try:
            # Calculate rolling volatility
            window_size = min(12, len(data) // 4)  # Adaptive window size
            if window_size < 2:
                window_size = 2
            
            rolling_std = []
            for i in range(window_size, len(data)):
                rolling_std.append(np.std(data[i-window_size:i]))
            
            if not rolling_std:
                rolling_std = [np.std(data)]
            
            return {
                "overall_volatility": float(np.std(data)),
                "rolling_volatility_mean": float(np.mean(rolling_std)),
                "rolling_volatility_std": float(np.std(rolling_std)),
                "volatility_trend": self._analyze_volatility_trend(rolling_std),
                "volatility_clustering": self._detect_volatility_clustering(rolling_std)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_volatility_trend(self, rolling_std: List[float]) -> str:
        """Analyze trend in volatility.
        
        Args:
            rolling_std: Rolling standard deviation values
            
        Returns:
            Volatility trend description
        """
        try:
            if len(rolling_std) < 2:
                return "insufficient_data"
            
            x = np.arange(len(rolling_std))
            slope, _ = np.polyfit(x, rolling_std, 1)
            
            if slope > 0.01:
                return "increasing"
            elif slope < -0.01:
                return "decreasing"
            else:
                return "stable"
                
        except Exception:
            return "unknown"
    
    def _detect_volatility_clustering(self, rolling_std: List[float]) -> bool:
        """Detect volatility clustering (high volatility periods followed by high volatility).
        
        Args:
            rolling_std: Rolling standard deviation values
            
        Returns:
            True if volatility clustering is detected
        """
        try:
            if len(rolling_std) < 4:
                return False
            
            # Simple autocorrelation test
            autocorr = np.corrcoef(rolling_std[:-1], rolling_std[1:])[0, 1]
            return autocorr > 0.3
            
        except Exception:
            return False
    
    def _analyze_model_characteristics(self, model) -> Dict[str, Any]:
        """Analyze characteristics of the trained model.
        
        Args:
            model: Trained Darts model
            
        Returns:
            Model characteristics analysis
        """
        try:
            characteristics = {
                "model_type": type(model).__name__,
                "is_probabilistic": hasattr(model, 'likelihood'),
                "supports_confidence_intervals": hasattr(model, 'predict'),
                "model_parameters": self._extract_model_parameters(model),
                "model_complexity": self._estimate_model_complexity(model)
            }
            
            return characteristics
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze model characteristics: {str(e)}")
            return {"error": str(e)}
    
    def _extract_model_parameters(self, model) -> Dict[str, Any]:
        """Extract model parameters for analysis.
        
        Args:
            model: Trained Darts model
            
        Returns:
            Model parameters
        """
        try:
            # Get model parameters if available
            if hasattr(model, 'model_params'):
                return model.model_params
            elif hasattr(model, '__dict__'):
                # Extract relevant parameters from model attributes
                params = {}
                for key, value in model.__dict__.items():
                    if not key.startswith('_') and isinstance(value, (int, float, str, bool)):
                        params[key] = value
                return params
            else:
                return {}
                
        except Exception:
            return {}
    
    def _estimate_model_complexity(self, model) -> str:
        """Estimate model complexity.
        
        Args:
            model: Trained Darts model
            
        Returns:
            Complexity level
        """
        try:
            model_type = type(model).__name__.lower()
            
            # Categorize models by complexity
            simple_models = ['arima', 'exponentialsmoothing', 'prophet', 'autotheta']
            medium_models = ['lstm', 'gru', 'tcn', 'randomforest']
            complex_models = ['transformer', 'nbeats', 'tft']
            
            if any(simple in model_type for simple in simple_models):
                return "low"
            elif any(medium in model_type for medium in medium_models):
                return "medium"
            elif any(complex in model_type for complex in complex_models):
                return "high"
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def _analyze_performance_patterns(self, evaluation_metrics: ModelEvaluationMetrics) -> Dict[str, Any]:
        """Analyze patterns in model performance.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            
        Returns:
            Performance pattern analysis
        """
        try:
            patterns = {
                "accuracy_level": self._categorize_accuracy_level(evaluation_metrics.mape),
                "error_distribution": self._analyze_error_distribution(evaluation_metrics),
                "performance_balance": self._analyze_performance_balance(evaluation_metrics),
                "training_efficiency": self._analyze_training_efficiency(evaluation_metrics)
            }
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze performance patterns: {str(e)}")
            return {"error": str(e)}
    
    def _categorize_accuracy_level(self, mape: float) -> str:
        """Categorize accuracy level based on MAPE.
        
        Args:
            mape: Mean Absolute Percentage Error
            
        Returns:
            Accuracy level category
        """
        if mape < 5:
            return "excellent"
        elif mape < 10:
            return "very_good"
        elif mape < 20:
            return "good"
        elif mape < 30:
            return "fair"
        else:
            return "poor"
    
    def _analyze_error_distribution(self, evaluation_metrics: ModelEvaluationMetrics) -> Dict[str, Any]:
        """Analyze distribution of errors.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            
        Returns:
            Error distribution analysis
        """
        try:
            # This would require access to actual vs predicted values
            # For now, provide basic analysis based on available metrics
            return {
                "mae_rmse_ratio": evaluation_metrics.mae / evaluation_metrics.rmse if evaluation_metrics.rmse > 0 else 0,
                "error_consistency": "consistent" if evaluation_metrics.mae / evaluation_metrics.rmse < 0.8 else "inconsistent",
                "outlier_sensitivity": "high" if evaluation_metrics.rmse > evaluation_metrics.mae * 1.5 else "low"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_performance_balance(self, evaluation_metrics: ModelEvaluationMetrics) -> Dict[str, Any]:
        """Analyze balance between different performance metrics.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            
        Returns:
            Performance balance analysis
        """
        try:
            # Check if model performs well across different metrics
            mape_good = evaluation_metrics.mape < 20
            directional_good = evaluation_metrics.directional_accuracy > 60
            
            if mape_good and directional_good:
                balance = "excellent"
            elif mape_good or directional_good:
                balance = "good"
            else:
                balance = "poor"
            
            return {
                "overall_balance": balance,
                "accuracy_vs_direction": "balanced" if abs(evaluation_metrics.mape - (100 - evaluation_metrics.directional_accuracy)) < 20 else "unbalanced",
                "metric_consistency": "consistent" if mape_good == directional_good else "inconsistent"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _analyze_training_efficiency(self, evaluation_metrics: ModelEvaluationMetrics) -> Dict[str, Any]:
        """Analyze training efficiency.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            
        Returns:
            Training efficiency analysis
        """
        try:
            # Calculate efficiency metrics
            samples_per_second = evaluation_metrics.total_samples / evaluation_metrics.training_time if evaluation_metrics.training_time > 0 else 0
            
            efficiency = {
                "training_time_per_sample": evaluation_metrics.training_time / evaluation_metrics.total_samples if evaluation_metrics.total_samples > 0 else 0,
                "samples_per_second": samples_per_second,
                "efficiency_category": self._categorize_training_efficiency(samples_per_second),
                "scalability_indicator": "good" if samples_per_second > 10 else "poor"
            }
            
            return efficiency
            
        except Exception as e:
            return {"error": str(e)}
    
    def _categorize_training_efficiency(self, samples_per_second: float) -> str:
        """Categorize training efficiency.
        
        Args:
            samples_per_second: Training speed metric
            
        Returns:
            Efficiency category
        """
        if samples_per_second > 50:
            return "very_fast"
        elif samples_per_second > 20:
            return "fast"
        elif samples_per_second > 5:
            return "moderate"
        else:
            return "slow"
    
    def _analyze_seasonality(self, training_request: ModelTrainingRequest) -> Dict[str, Any]:
        """Analyze seasonality in the data.
        
        Args:
            training_request: Training request containing data
            
        Returns:
            Seasonality analysis results
        """
        try:
            # This would require more sophisticated seasonality analysis
            # For now, provide basic analysis
            data = np.array(training_request.time_series_data)
            
            # Simple seasonality detection
            if len(data) >= 52:  # At least one year of weekly data
                # Check for weekly patterns
                weekly_patterns = self._detect_weekly_patterns(data)
                
                return {
                    "has_seasonality": weekly_patterns["has_pattern"],
                    "seasonality_strength": weekly_patterns["strength"],
                    "seasonality_period": "weekly" if weekly_patterns["has_pattern"] else "none",
                    "seasonality_confidence": weekly_patterns["confidence"]
                }
            else:
                return {
                    "has_seasonality": False,
                    "seasonality_strength": 0.0,
                    "seasonality_period": "insufficient_data",
                    "seasonality_confidence": 0.0
                }
                
        except Exception as e:
            return {"error": str(e)}
    
    def _detect_weekly_patterns(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect weekly patterns in the data.
        
        Args:
            data: Input data array
            
        Returns:
            Weekly pattern detection results
        """
        try:
            # Simple autocorrelation analysis for weekly patterns
            if len(data) < 14:  # Need at least 2 weeks
                return {"has_pattern": False, "strength": 0.0, "confidence": 0.0}
            
            # Calculate autocorrelation at lag 7 (weekly)
            lag = 7
            if len(data) <= lag:
                return {"has_pattern": False, "strength": 0.0, "confidence": 0.0}
            
            # Calculate autocorrelation
            mean_data = np.mean(data)
            var_data = np.var(data)
            
            if var_data == 0:
                return {"has_pattern": False, "strength": 0.0, "confidence": 0.0}
            
            autocorr = np.corrcoef(data[:-lag], data[lag:])[0, 1]
            
            has_pattern = abs(autocorr) > 0.3
            strength = abs(autocorr)
            confidence = min(1.0, abs(autocorr) * 2)  # Simple confidence estimate
            
            return {
                "has_pattern": has_pattern,
                "strength": float(strength),
                "confidence": float(confidence)
            }
            
        except Exception:
            return {"has_pattern": False, "strength": 0.0, "confidence": 0.0}
    
    def _analyze_residuals(self, model, training_request: ModelTrainingRequest) -> Dict[str, Any]:
        """Analyze model residuals.
        
        Args:
            model: Trained Darts model
            training_request: Training request
            
        Returns:
            Residual analysis results
        """
        try:
            # This would require access to actual vs predicted values
            # For now, provide placeholder analysis
            return {
                "residual_analysis_available": False,
                "reason": "Requires access to actual vs predicted values",
                "recommendation": "Use forecast accuracy report for detailed residual analysis"
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _generate_performance_benchmarks(self, evaluation_metrics: ModelEvaluationMetrics,
                                       model_type: str) -> Dict[str, Any]:
        """Generate performance benchmarks for the model.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            model_type: Type of model
            
        Returns:
            Performance benchmarks
        """
        try:
            # Define benchmark thresholds based on model type
            benchmarks = self._get_model_type_benchmarks(model_type)
            
            # Compare against benchmarks
            performance = {
                "mae_performance": self._categorize_performance(evaluation_metrics.mae, benchmarks["mae"]),
                "rmse_performance": self._categorize_performance(evaluation_metrics.rmse, benchmarks["rmse"]),
                "mape_performance": self._categorize_performance(evaluation_metrics.mape, benchmarks["mape"]),
                "directional_performance": self._categorize_performance(evaluation_metrics.directional_accuracy, benchmarks["directional"], reverse=True),
                "overall_performance": self._calculate_overall_performance(evaluation_metrics, benchmarks)
            }
            
            return {
                "benchmarks": benchmarks,
                "performance": performance,
                "performance_score": self._calculate_performance_score(performance)
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to generate performance benchmarks: {str(e)}")
            return {"error": str(e)}
    
    def _get_model_type_benchmarks(self, model_type: str) -> Dict[str, Dict[str, float]]:
        """Get benchmark thresholds for different model types.
        
        Args:
            model_type: Type of model
            
        Returns:
            Benchmark thresholds
        """
        # Define benchmarks based on model type
        if model_type in ["lstm", "gru", "tcn", "transformer", "n_beats", "tft"]:
            # Neural network models
            return {
                "mae": {"excellent": 2.0, "good": 5.0, "fair": 10.0},
                "rmse": {"excellent": 3.0, "good": 7.0, "fair": 15.0},
                "mape": {"excellent": 5.0, "good": 15.0, "fair": 25.0},
                "directional": {"excellent": 80.0, "good": 65.0, "fair": 50.0}
            }
        elif model_type in ["arima", "exponential_smoothing", "prophet"]:
            # Statistical models
            return {
                "mae": {"excellent": 3.0, "good": 7.0, "fair": 15.0},
                "rmse": {"excellent": 4.0, "good": 10.0, "fair": 20.0},
                "mape": {"excellent": 8.0, "good": 20.0, "fair": 35.0},
                "directional": {"excellent": 75.0, "good": 60.0, "fair": 45.0}
            }
        else:
            # Default benchmarks
            return {
                "mae": {"excellent": 5.0, "good": 10.0, "fair": 20.0},
                "rmse": {"excellent": 7.0, "good": 15.0, "fair": 30.0},
                "mape": {"excellent": 10.0, "good": 25.0, "fair": 40.0},
                "directional": {"excellent": 70.0, "good": 55.0, "fair": 40.0}
            }
    
    def _categorize_performance(self, value: float, benchmarks: Dict[str, float], 
                               reverse: bool = False) -> str:
        """Categorize performance based on benchmarks.
        
        Args:
            value: Performance value
            benchmarks: Benchmark thresholds
            reverse: Whether higher values are better
            
        Returns:
            Performance category
        """
        if reverse:
            if value >= benchmarks["excellent"]:
                return "excellent"
            elif value >= benchmarks["good"]:
                return "good"
            elif value >= benchmarks["fair"]:
                return "fair"
            else:
                return "poor"
        else:
            if value <= benchmarks["excellent"]:
                return "excellent"
            elif value <= benchmarks["good"]:
                return "good"
            elif value <= benchmarks["fair"]:
                return "fair"
            else:
                return "poor"
    
    def _calculate_overall_performance(self, evaluation_metrics: ModelEvaluationMetrics,
                                     benchmarks: Dict[str, Dict[str, float]]) -> str:
        """Calculate overall performance category.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            benchmarks: Benchmark thresholds
            
        Returns:
            Overall performance category
        """
        try:
            # Calculate performance scores
            mae_score = self._get_performance_score(evaluation_metrics.mae, benchmarks["mae"])
            rmse_score = self._get_performance_score(evaluation_metrics.rmse, benchmarks["rmse"])
            mape_score = self._get_performance_score(evaluation_metrics.mape, benchmarks["mape"])
            directional_score = self._get_performance_score(evaluation_metrics.directional_accuracy, benchmarks["directional"], reverse=True)
            
            # Calculate average score
            avg_score = (mae_score + rmse_score + mape_score + directional_score) / 4
            
            # Categorize overall performance
            if avg_score >= 3.5:
                return "excellent"
            elif avg_score >= 2.5:
                return "good"
            elif avg_score >= 1.5:
                return "fair"
            else:
                return "poor"
                
        except Exception:
            return "unknown"
    
    def _get_performance_score(self, value: float, benchmarks: Dict[str, float], 
                              reverse: bool = False) -> float:
        """Get numerical performance score.
        
        Args:
            value: Performance value
            benchmarks: Benchmark thresholds
            reverse: Whether higher values are better
            
        Returns:
            Performance score (1-4)
        """
        if reverse:
            if value >= benchmarks["excellent"]:
                return 4.0
            elif value >= benchmarks["good"]:
                return 3.0
            elif value >= benchmarks["fair"]:
                return 2.0
            else:
                return 1.0
        else:
            if value <= benchmarks["excellent"]:
                return 4.0
            elif value <= benchmarks["good"]:
                return 3.0
            elif value <= benchmarks["fair"]:
                return 2.0
            else:
                return 1.0
    
    def _calculate_performance_score(self, performance: Dict[str, str]) -> float:
        """Calculate overall performance score.
        
        Args:
            performance: Performance categories
            
        Returns:
            Overall performance score (0-100)
        """
        try:
            score_mapping = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
            
            scores = []
            for metric, category in performance.items():
                if metric != "overall_performance" and category in score_mapping:
                    scores.append(score_mapping[category])
            
            if not scores:
                return 0.0
            
            avg_score = np.mean(scores)
            return (avg_score / 4) * 100  # Convert to percentage
            
        except Exception:
            return 0.0
    
    def _generate_evaluation_recommendations(self, evaluation_metrics: ModelEvaluationMetrics,
                                           detailed_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on evaluation results.
        
        Args:
            evaluation_metrics: Model evaluation metrics
            detailed_analysis: Detailed analysis results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            # Performance-based recommendations
            if evaluation_metrics.mape > 25:
                recommendations.append("High MAPE indicates poor accuracy. Consider using a different model type or more training data.")
            
            if evaluation_metrics.directional_accuracy < 55:
                recommendations.append("Low directional accuracy suggests the model struggles with trend prediction. Consider trend-aware models like Prophet or N-BEATS.")
            
            if evaluation_metrics.training_time > 300:  # 5 minutes
                recommendations.append("Training time is high. Consider using faster models or reducing model complexity.")
            
            # Data-based recommendations
            if "data_characteristics" in detailed_analysis:
                data_char = detailed_analysis["data_characteristics"]
                
                if data_char.get("coefficient_of_variation", 0) > 1.0:
                    recommendations.append("High data variability detected. Consider using robust models like Random Forest or ensemble methods.")
                
                if data_char.get("trend_analysis", {}).get("trend_strength", 0) > 0.7:
                    recommendations.append("Strong trend detected. Consider using trend-aware models like Prophet or ARIMA.")
            
            # Model-specific recommendations
            if "model_characteristics" in detailed_analysis:
                model_char = detailed_analysis["model_characteristics"]
                
                if model_char.get("model_complexity") == "high" and evaluation_metrics.mape > 20:
                    recommendations.append("Complex model with poor performance. Consider simpler models or more training data.")
                
                if not model_char.get("is_probabilistic", False):
                    recommendations.append("Model doesn't provide confidence intervals. Consider probabilistic models for uncertainty quantification.")
            
            # Add general recommendations if none specific
            if not recommendations:
                recommendations.append("Model performance looks good. Continue monitoring and consider retraining periodically.")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Failed to generate recommendations: {str(e)}")
            return ["Unable to generate recommendations due to analysis error."]
    
    def benchmark_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Benchmark multiple models against each other.
        
        Args:
            model_ids: List of model identifiers to benchmark
            
        Returns:
            Benchmarking results
            
        Raises:
            ValidationError: If parameters are invalid
            ModelError: If benchmarking fails
        """
        try:
            # Validate parameters
            if not model_ids:
                raise ValidationError("At least one model ID is required")
            
            if len(model_ids) > 20:
                raise ValidationError("Cannot benchmark more than 20 models at once")
            
            benchmark_results = {
                "benchmark_date": datetime.now().isoformat(),
                "models": {},
                "comparison": {},
                "rankings": {},
                "recommendations": []
            }
            
            # Evaluate each model
            for model_id in model_ids:
                try:
                    evaluation = self.evaluate_model_comprehensive(model_id)
                    benchmark_results["models"][model_id] = evaluation
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate model {model_id}: {str(e)}")
                    benchmark_results["models"][model_id] = {"error": str(e)}
            
            # Generate comparisons and rankings
            benchmark_results["comparison"] = self._compare_models_benchmark(benchmark_results["models"])
            benchmark_results["rankings"] = self._generate_benchmark_rankings(benchmark_results["models"])
            benchmark_results["recommendations"] = self._generate_benchmark_recommendations(benchmark_results)
            
            self.logger.info(f"Completed benchmarking for {len(model_ids)} models")
            
            return benchmark_results
            
        except Exception as e:
            self.logger.error(f"Model benchmarking failed: {str(e)}")
            raise ModelError(f"Model benchmarking failed: {str(e)}")
    
    def _compare_models_benchmark(self, models_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare models in benchmarking.
        
        Args:
            models_data: Dictionary containing model evaluation data
            
        Returns:
            Comparison results
        """
        try:
            comparison = {
                "best_performing": {},
                "most_efficient": {},
                "most_accurate": {},
                "best_directional": {}
            }
            
            successful_models = []
            
            for model_id, data in models_data.items():
                if "error" not in data:
                    successful_models.append((model_id, data))
            
            if not successful_models:
                return comparison
            
            # Find best models by different criteria
            if successful_models:
                # Best overall performance
                best_overall = min(successful_models, 
                                 key=lambda x: x[1]["performance_benchmarks"]["performance"]["overall_performance"])
                comparison["best_performing"] = {
                    "model_id": best_overall[0],
                    "performance": best_overall[1]["performance_benchmarks"]["performance"]["overall_performance"]
                }
                
                # Most efficient (fastest training)
                most_efficient = min(successful_models,
                                   key=lambda x: x[1]["basic_metrics"]["training_time"])
                comparison["most_efficient"] = {
                    "model_id": most_efficient[0],
                    "training_time": most_efficient[1]["basic_metrics"]["training_time"]
                }
                
                # Most accurate (lowest MAPE)
                most_accurate = min(successful_models,
                                  key=lambda x: x[1]["basic_metrics"]["mape"])
                comparison["most_accurate"] = {
                    "model_id": most_accurate[0],
                    "mape": most_accurate[1]["basic_metrics"]["mape"]
                }
                
                # Best directional accuracy
                best_directional = max(successful_models,
                                     key=lambda x: x[1]["basic_metrics"]["directional_accuracy"])
                comparison["best_directional"] = {
                    "model_id": best_directional[0],
                    "directional_accuracy": best_directional[1]["basic_metrics"]["directional_accuracy"]
                }
            
            return comparison
            
        except Exception as e:
            self.logger.warning(f"Failed to compare models: {str(e)}")
            return {"error": str(e)}
    
    def _generate_benchmark_rankings(self, models_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate rankings for benchmarked models.
        
        Args:
            models_data: Dictionary containing model evaluation data
            
        Returns:
            Model rankings by different criteria
        """
        try:
            rankings = {
                "by_overall_performance": [],
                "by_accuracy": [],
                "by_efficiency": [],
                "by_directional_accuracy": []
            }
            
            successful_models = []
            
            for model_id, data in models_data.items():
                if "error" not in data:
                    successful_models.append((model_id, data))
            
            if not successful_models:
                return rankings
            
            # Sort by different criteria
            rankings["by_overall_performance"] = [
                m[0] for m in sorted(successful_models,
                                   key=lambda x: x[1]["performance_benchmarks"]["performance_score"],
                                   reverse=True)
            ]
            
            rankings["by_accuracy"] = [
                m[0] for m in sorted(successful_models,
                                   key=lambda x: x[1]["basic_metrics"]["mape"])
            ]
            
            rankings["by_efficiency"] = [
                m[0] for m in sorted(successful_models,
                                   key=lambda x: x[1]["basic_metrics"]["training_time"])
            ]
            
            rankings["by_directional_accuracy"] = [
                m[0] for m in sorted(successful_models,
                                   key=lambda x: x[1]["basic_metrics"]["directional_accuracy"],
                                   reverse=True)
            ]
            
            return rankings
            
        except Exception as e:
            self.logger.warning(f"Failed to generate rankings: {str(e)}")
            return {"error": str(e)}
    
    def _generate_benchmark_recommendations(self, benchmark_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on benchmarking results.
        
        Args:
            benchmark_results: Complete benchmarking results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        try:
            models_data = benchmark_results["models"]
            successful_models = [m for m in models_data.values() if "error" not in m]
            
            if not successful_models:
                recommendations.append("No successful model evaluations. Check model training and data quality.")
                return recommendations
            
            # Performance-based recommendations
            best_model = benchmark_results["comparison"].get("best_performing", {})
            if best_model:
                recommendations.append(f"Model {best_model['model_id']} shows the best overall performance.")
            
            # Efficiency recommendations
            efficient_model = benchmark_results["comparison"].get("most_efficient", {})
            if efficient_model and efficient_model.get("training_time", 0) < 60:
                recommendations.append(f"Model {efficient_model['model_id']} is the most efficient (training time: {efficient_model['training_time']:.2f}s).")
            
            # Accuracy recommendations
            accurate_model = benchmark_results["comparison"].get("most_accurate", {})
            if accurate_model and accurate_model.get("mape", 100) < 15:
                recommendations.append(f"Model {accurate_model['model_id']} has the best accuracy (MAPE: {accurate_model['mape']:.2f}%).")
            
            # General recommendations
            if len(successful_models) > 1:
                recommendations.append("Consider ensemble methods combining the best performing models.")
            
            if not recommendations:
                recommendations.append("All models show similar performance. Consider business requirements for model selection.")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Failed to generate benchmark recommendations: {str(e)}")
            return ["Unable to generate recommendations due to analysis error."] 
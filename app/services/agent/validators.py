"""
Agent Validators
Validation for extracted slots and parameters.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

from app.models.agent_models import AgentIntent
from app.services.agent.slot_extractor import ExtractedSlots


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str] = None
    warnings: List[str] = None
    sanitized_slots: Optional[ExtractedSlots] = None
    
    def __post_init__(self):
        """Initialize default values."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class AgentValidator:
    """Validator for agent slots and parameters."""
    
    def __init__(self):
        """Initialize the validator with constraints."""
        self.max_horizon_days = 90
        self.max_keywords = 5
        self.max_query_length = 1000
        self.min_confidence = 0.3
        
    def validate_slots(
        self, 
        slots: ExtractedSlots, 
        intent: AgentIntent,
        confidence: float
    ) -> ValidationResult:
        """Validate extracted slots for an intent.
        
        Args:
            slots: The extracted slots to validate
            intent: The recognized intent
            confidence: The intent recognition confidence
            
        Returns:
            ValidationResult with validation status and errors
        """
        result = ValidationResult(is_valid=True)
        
        # Validate confidence
        if confidence < self.min_confidence:
            result.is_valid = False
            result.errors.append(
                f"Intent confidence too low ({confidence:.2f}). "
                f"Minimum required: {self.min_confidence}"
            )
        
        # Validate keywords
        keyword_result = self._validate_keywords(slots.keywords, intent)
        if not keyword_result.is_valid:
            result.is_valid = False
            result.errors.extend(keyword_result.errors)
        if keyword_result.warnings:
            result.warnings.extend(keyword_result.warnings)
        
        # Validate intent-specific parameters
        if intent in [AgentIntent.FORECAST, AgentIntent.TRAIN]:
            horizon_result = self._validate_horizon(slots.horizon)
            if not horizon_result.is_valid:
                result.is_valid = False
                result.errors.extend(horizon_result.errors)
            if horizon_result.warnings:
                result.warnings.extend(horizon_result.warnings)
            
            quantile_result = self._validate_quantiles(slots.quantiles)
            if not quantile_result.is_valid:
                result.is_valid = False
                result.errors.extend(quantile_result.errors)
            if quantile_result.warnings:
                result.warnings.extend(quantile_result.warnings)
        
        elif intent == AgentIntent.SUMMARY:
            date_result = self._validate_date_range(slots.date_range)
            if not date_result.is_valid:
                result.is_valid = False
                result.errors.extend(date_result.errors)
            if date_result.warnings:
                result.warnings.extend(date_result.warnings)
        
        elif intent == AgentIntent.EVALUATE:
            model_result = self._validate_model_id(slots.model_id)
            if not model_result.is_valid:
                result.is_valid = False
                result.errors.extend(model_result.errors)
            if model_result.warnings:
                result.warnings.extend(model_result.warnings)
        
        elif intent == AgentIntent.COMPARE:
            # COMPARE requires at least 2 keywords
            if not slots.keywords or len(slots.keywords) < 2:
                result.is_valid = False
                result.errors.append(
                    "Compare intent requires at least 2 keywords to compare."
                )
            
            model_result = self._validate_model_id(slots.model_id)
            if not model_result.is_valid:
                result.is_valid = False
                result.errors.extend(model_result.errors)
            if model_result.warnings:
                result.warnings.extend(model_result.warnings)
        
        # Create sanitized slots if validation passes
        if result.is_valid:
            result.sanitized_slots = self._sanitize_slots(slots, intent)
        
        return result
    
    def _validate_keywords(
        self, 
        keywords: Optional[List[str]], 
        intent: AgentIntent
    ) -> ValidationResult:
        """Validate keywords for an intent.
        
        Args:
            keywords: List of keywords to validate
            intent: The intent being validated
            
        Returns:
            ValidationResult for keywords
        """
        result = ValidationResult(is_valid=True)
        
        # Keywords are required for most intents
        if intent not in [AgentIntent.HEALTH, AgentIntent.CACHE_STATS, AgentIntent.CACHE_CLEAR, AgentIntent.EVALUATE]:
            if not keywords:
                result.is_valid = False
                result.errors.append(
                    f"Keywords are required for {intent.value} intent. "
                    "Please specify what you want to analyze."
                )
                return result
        
        if keywords:
            # Check keyword count
            if len(keywords) > self.max_keywords:
                result.warnings.append(
                    f"Too many keywords ({len(keywords)}). "
                    f"Using first {self.max_keywords} keywords."
                )
            
            # Check keyword length and content
            for i, keyword in enumerate(keywords):
                if len(keyword.strip()) < 2:
                    result.errors.append(
                        f"Keyword {i+1} is too short: '{keyword}'"
                    )
                    result.is_valid = False
                
                if len(keyword.strip()) > 50:
                    result.warnings.append(
                        f"Keyword {i+1} is very long: '{keyword}'"
                    )
                
                # Check for potentially unsafe content
                if self._contains_unsafe_content(keyword):
                    result.errors.append(
                        f"Keyword {i+1} contains potentially unsafe content: '{keyword}'"
                    )
                    result.is_valid = False
        
        return result
    
    def _validate_horizon(self, horizon: Optional[int]) -> ValidationResult:
        """Validate forecast horizon.
        
        Args:
            horizon: Horizon in days
            
        Returns:
            ValidationResult for horizon
        """
        result = ValidationResult(is_valid=True)
        
        if horizon is not None:
            if horizon < 1:
                result.errors.append("Horizon must be at least 1 day")
                result.is_valid = False
            
            if horizon > self.max_horizon_days:
                result.warnings.append(
                    f"Horizon ({horizon} days) exceeds recommended maximum "
                    f"({self.max_horizon_days} days). This may affect accuracy."
                )
        
        return result
    
    def _validate_quantiles(self, quantiles: Optional[List[float]]) -> ValidationResult:
        """Validate quantiles.
        
        Args:
            quantiles: List of quantiles to validate
            
        Returns:
            ValidationResult for quantiles
        """
        result = ValidationResult(is_valid=True)
        
        if quantiles:
            # Check quantile range
            for i, quantile in enumerate(quantiles):
                if not 0 < quantile < 1:
                    result.errors.append(
                        f"Quantile {i+1} ({quantile}) must be between 0 and 1"
                    )
                    result.is_valid = False
            
            # Check if quantiles are sorted
            if result.is_valid and len(quantiles) > 1:
                if quantiles != sorted(quantiles):
                    result.warnings.append(
                        "Quantiles should be in ascending order. "
                        "Auto-sorting quantiles."
                    )
            
            # Check for reasonable quantile combinations
            if result.is_valid:
                if 0.5 in quantiles and len(quantiles) == 1:
                    result.warnings.append(
                        "Only median (0.5) specified. Consider adding other quantiles "
                        "for better uncertainty representation."
                    )
        
        return result
    
    def _validate_date_range(
        self, 
        date_range: Optional[Dict[str, str]]
    ) -> ValidationResult:
        """Validate date range.
        
        Args:
            date_range: Dictionary with start_date and end_date
            
        Returns:
            ValidationResult for date range
        """
        result = ValidationResult(is_valid=True)
        
        if date_range:
            try:
                from datetime import datetime
                
                start_date = date_range.get('start_date')
                end_date = date_range.get('end_date')
                
                if not start_date or not end_date:
                    result.errors.append("Both start_date and end_date are required")
                    result.is_valid = False
                else:
                    # Validate date format
                    try:
                        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                        
                        if start_dt > end_dt:
                            result.errors.append(
                                "start_date must be before end_date"
                            )
                            result.is_valid = False
                        
                        # Check if date range is reasonable
                        days_diff = (end_dt - start_dt).days
                        if days_diff > 365:
                            result.warnings.append(
                                f"Date range is very long ({days_diff} days). "
                                "Consider using a shorter range for better performance."
                            )
                        
                        if days_diff < 1:
                            result.warnings.append(
                                "Date range is very short. Consider using a longer range."
                            )
                    
                    except ValueError:
                        result.errors.append(
                            "Invalid date format. Use YYYY-MM-DD format."
                        )
                        result.is_valid = False
            
            except ImportError:
                result.errors.append("Date validation not available")
                result.is_valid = False
        
        return result
    
    def _validate_model_id(self, model_id: Optional[str]) -> ValidationResult:
        """Validate model ID.
        
        Args:
            model_id: Model ID to validate
            
        Returns:
            ValidationResult for model ID
        """
        result = ValidationResult(is_valid=True)
        
        if model_id:
            # Check model ID format
            if len(model_id) < 3:
                result.errors.append("Model ID is too short")
                result.is_valid = False
            
            if len(model_id) > 100:
                result.errors.append("Model ID is too long")
                result.is_valid = False
            
            # Check for valid characters
            import re
            if not re.match(r'^[a-zA-Z0-9_-]+$', model_id):
                result.errors.append(
                    "Model ID contains invalid characters. "
                    "Use only letters, numbers, hyphens, and underscores."
                )
                result.is_valid = False
        
        return result
    
    def _sanitize_slots(self, slots: ExtractedSlots, intent: AgentIntent) -> ExtractedSlots:
        """Sanitize slots by applying reasonable defaults and constraints.
        
        Args:
            slots: Original slots
            intent: The intent
            
        Returns:
            Sanitized slots
        """
        sanitized = ExtractedSlots()
        
        # Copy keywords with limit
        if slots.keywords:
            sanitized.keywords = slots.keywords[:self.max_keywords]
        
        # Copy and constrain horizon
        if slots.horizon:
            sanitized.horizon = min(max(slots.horizon, 1), self.max_horizon_days)
        
        # Copy and sort quantiles
        if slots.quantiles:
            sanitized.quantiles = sorted([
                q for q in slots.quantiles if 0 < q < 1
            ])
        
        # Copy other fields
        sanitized.date_range = slots.date_range
        sanitized.model_id = slots.model_id
        sanitized.geo = slots.geo
        sanitized.category = slots.category
        
        return sanitized
    
    def _contains_unsafe_content(self, text: str) -> bool:
        """Check if text contains potentially unsafe content.
        
        Args:
            text: Text to check
            
        Returns:
            True if potentially unsafe content is found
        """
        unsafe_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'<iframe',
            r'<object',
            r'<embed'
        ]
        
        text_lower = text.lower()
        for pattern in unsafe_patterns:
            if pattern in text_lower:
                return True
        
        return False
    
    def validate_query_length(self, query: str) -> ValidationResult:
        """Validate query length.
        
        Args:
            query: The query to validate
            
        Returns:
            ValidationResult for query length
        """
        result = ValidationResult(is_valid=True)
        
        if len(query) > self.max_query_length:
            result.is_valid = False
            result.errors.append(
                f"Query too long ({len(query)} characters). "
                f"Maximum allowed: {self.max_query_length}"
            )
        
        if len(query.strip()) < 3:
            result.is_valid = False
            result.errors.append("Query too short. Please provide more details.")
        
        return result
    
    def get_validation_help(self, intent: AgentIntent) -> str:
        """Get help text for validating a specific intent.
        
        Args:
            intent: The intent to get help for
            
        Returns:
            Help text for the intent
        """
        help_texts = {
            AgentIntent.FORECAST: (
                "For forecasting, please provide:\n"
                "- Keywords to forecast (e.g., 'machine learning')\n"
                "- Optional: Time horizon (e.g., 'next week', '30 days')\n"
                "- Optional: Quantiles (e.g., 'p10/p50/p90')\n"
                "Example: 'Forecast machine learning trends for the next month with p10/p50/p90'"
            ),
            AgentIntent.COMPARE: (
                "For comparison, please provide:\n"
                "- Two or more keywords to compare\n"
                "- Optional: Time period (e.g., 'last month')\n"
                "Example: 'Compare machine learning vs artificial intelligence'"
            ),
            AgentIntent.SUMMARY: (
                "For summary, please provide:\n"
                "- Keywords to summarize (e.g., 'python programming')\n"
                "- Optional: Time period (e.g., 'last week', 'this month')\n"
                "Example: 'Give me a summary of python programming trends this month'"
            ),
            AgentIntent.TRAIN: (
                "For training, please provide:\n"
                "- Keywords to train on (e.g., 'data science')\n"
                "- Optional: Time horizon for forecasting\n"
                "Example: 'Train a model for data science with 30-day horizon'"
            ),
            AgentIntent.EVALUATE: (
                "For evaluation, please provide:\n"
                "- Optional: Specific model ID to evaluate\n"
                "- If no model specified, will evaluate all models\n"
                "Example: 'Evaluate model performance' or 'Evaluate model abc-123'"
            ),
            AgentIntent.HEALTH: (
                "Health check requires no additional parameters.\n"
                "Example: 'Is the service working?'"
            ),
            AgentIntent.CACHE_STATS: (
                "Cache statistics requires no additional parameters.\n"
                "Example: 'Show me cache statistics'"
            ),
            AgentIntent.CACHE_CLEAR: (
                "Cache clear requires no additional parameters.\n"
                "Example: 'Clear the cache'"
            ),
            AgentIntent.LIST_MODELS: (
                "List models requires no additional parameters.\n"
                "Example: 'Show me available models'"
            )
        }
        
        return help_texts.get(intent, "No specific validation requirements for this intent.") 
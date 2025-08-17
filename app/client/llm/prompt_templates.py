"""
Prompt Templates for LLM Intent Classification
Few-shot prompts with strict JSON schema for intent classification.
"""

from typing import List, Dict, Any
from dataclasses import dataclass


@dataclass
class FewShotExample:
    """Few-shot example for intent classification."""
    query: str
    intent: str
    confidence: float
    rationale: str


class IntentClassificationPrompt:
    """Prompt template for intent classification."""
    
    # Allowed intents
    ALLOWED_INTENTS = [
        "forecast", "compare", "summary", "train", 
        "evaluate", "health", "list_models", "unknown"
    ]
    
    def __init__(self):
        """Initialize prompt template with few-shot examples."""
        self.few_shot_examples = self._build_examples()
        self.system_message = self._build_system_message()
    
    def build_prompt(self, query: str) -> List[Dict[str, str]]:
        """Build the complete prompt for intent classification.
        
        Args:
            query: User query to classify
            
        Returns:
            List of message dictionaries for the LLM
        """
        messages = [
            {"role": "system", "content": self.system_message}
        ]
        
        # Add few-shot examples
        for example in self.few_shot_examples:
            messages.append({
                "role": "user", 
                "content": example.query
            })
            messages.append({
                "role": "assistant",
                "content": self._format_response(example.intent, example.confidence, example.rationale)
            })
        
        # Add the actual query
        messages.append({
            "role": "user",
            "content": query
        })
        
        return messages
    
    def _build_system_message(self) -> str:
        """Build the system message with instructions and schema."""
        return """You are an intent classification system for a trend forecasting agent. Your task is to classify user queries into one of the following intents:

- forecast: User wants to predict future trends
- compare: User wants to compare multiple keywords/trends
- summary: User wants current trend data or summary statistics
- train: User wants to train a new forecasting model
- evaluate: User wants to evaluate model performance
- health: User wants system health/status information
- list_models: User wants to see available models
- unknown: Query doesn't match any clear intent

Return ONLY valid JSON in this exact format:
{
  "intent": "<intent_name>",
  "confidence": <float_between_0_and_1>,
  "rationale": "<brief_explanation>"
}

Rules:
- Use "forecast" for future predictions (e.g., "how will X trend next week?")
- Use "compare" for direct comparisons (e.g., "compare A vs B")
- Use "summary" for current data requests (e.g., "show current data for X")
- Use "train" for model training requests
- Use "evaluate" for model evaluation requests
- Use "health" for system status requests
- Use "list_models" for model listing requests
- Use "unknown" for unclear or ambiguous queries
- Confidence should reflect your certainty (0.0 = uncertain, 1.0 = very certain)
- Keep rationale brief and factual"""
    
    def _build_examples(self) -> List[FewShotExample]:
        """Build few-shot examples for each intent."""
        return [
            # Forecast examples
            FewShotExample(
                query="How will 'iphone 17' trend next week?",
                intent="forecast",
                confidence=0.95,
                rationale="Direct request for future trend prediction"
            ),
            FewShotExample(
                query="Predict the popularity of 'bitcoin' in the next month",
                intent="forecast",
                confidence=0.90,
                rationale="Future prediction request with specific timeframe"
            ),
            
            # Compare examples
            FewShotExample(
                query="Compare 'python' vs 'javascript' trends",
                intent="compare",
                confidence=0.95,
                rationale="Direct comparison between two keywords"
            ),
            FewShotExample(
                query="Which is more popular: 'tiktok' or 'instagram'?",
                intent="compare",
                confidence=0.90,
                rationale="Comparison question between two platforms"
            ),
            
            # Summary examples
            FewShotExample(
                query="Show current data for 'bitcoin'",
                intent="summary",
                confidence=0.85,
                rationale="Request for current trend data"
            ),
            FewShotExample(
                query="What's the current trend for 'artificial intelligence'?",
                intent="summary",
                confidence=0.80,
                rationale="Current trend information request"
            ),
            
            # Train examples
            FewShotExample(
                query="Train a new model for 'olympics 2028'",
                intent="train",
                confidence=0.95,
                rationale="Direct request to train a new model"
            ),
            FewShotExample(
                query="Create a forecasting model for 'electric vehicles'",
                intent="train",
                confidence=0.90,
                rationale="Model creation request"
            ),
            
            # Evaluate examples
            FewShotExample(
                query="How accurate are our models?",
                intent="evaluate",
                confidence=0.85,
                rationale="Request for model evaluation"
            ),
            FewShotExample(
                query="Evaluate the performance of model 'bitcoin_forecast'",
                intent="evaluate",
                confidence=0.90,
                rationale="Specific model evaluation request"
            ),
            
            # Health examples
            FewShotExample(
                query="Are you up?",
                intent="health",
                confidence=0.80,
                rationale="System health check request"
            ),
            FewShotExample(
                query="What's the system status?",
                intent="health",
                confidence=0.85,
                rationale="System status inquiry"
            ),
            
            # List models examples
            FewShotExample(
                query="List available models",
                intent="list_models",
                confidence=0.90,
                rationale="Direct request to list models"
            ),
            FewShotExample(
                query="What models do you have?",
                intent="list_models",
                confidence=0.85,
                rationale="Model listing request"
            ),
            
            # Unknown examples
            FewShotExample(
                query="not sure, tell me more",
                intent="unknown",
                confidence=0.70,
                rationale="Ambiguous request without clear intent"
            ),
            FewShotExample(
                query="hello there",
                intent="unknown",
                confidence=0.60,
                rationale="Greeting without specific request"
            ),
            
            # Edge cases
            FewShotExample(
                query="don't compare anything",
                intent="unknown",
                confidence=0.75,
                rationale="Negative instruction without clear positive intent"
            ),
            FewShotExample(
                query="maybe show me something",
                intent="unknown",
                confidence=0.65,
                rationale="Vague request without specific intent"
            )
        ]
    
    def _format_response(self, intent: str, confidence: float, rationale: str) -> str:
        """Format a response as JSON string."""
        return f'{{"intent": "{intent}", "confidence": {confidence}, "rationale": "{rationale}"}}'
    
    def validate_intent(self, intent: str) -> bool:
        """Validate that an intent is in the allowed list.
        
        Args:
            intent: Intent to validate
            
        Returns:
            True if valid, False otherwise
        """
        return intent in self.ALLOWED_INTENTS
    
    def get_allowed_intents(self) -> List[str]:
        """Get list of allowed intents.
        
        Returns:
            List of allowed intent names
        """
        return self.ALLOWED_INTENTS.copy() 
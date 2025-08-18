"""
Hybrid Intent Recognizer
Combines semantic similarity, regex patterns, and optional LLM classification.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from app.models.agent_models import AgentIntent
from app.utils.text_normalizer import TextNormalizer
from app.config.config import Config


class ScorerType(Enum):
    """Types of scorers used in the hybrid system."""
    SEMANTIC = "semantic"
    REGEX = "regex"
    LLM = "llm"


@dataclass
class ScorerResult:
    """Result from a single scorer."""
    scorer_type: ScorerType
    scores: Dict[AgentIntent, float]
    confidence: float
    valid: bool = True  # Indicates if this scorer is valid and should contribute to ensemble


@dataclass
class IntentExample:
    """Example utterance for an intent."""
    text: str
    intent: AgentIntent


@dataclass
class IntentPattern:
    """Regex pattern for an intent."""
    pattern: str
    intent: AgentIntent
    weight: float = 1.0


class SimpleSemanticScorer:
    """Simple TF-IDF based semantic scorer as fallback."""
    
    def __init__(self):
        """Initialize the semantic scorer."""
        self.intent_examples = self._build_examples()
        self.tfidf_vectors = self._build_tfidf_vectors()
    
    def score_query(self, query: str) -> Dict[AgentIntent, float]:
        """Score a query against all intents using TF-IDF similarity."""
        query_vector = self._get_query_vector(query)
        scores = {}
        
        for intent, examples in self.intent_examples.items():
            if not examples:
                scores[intent] = 0.0
                continue
                
            # Calculate similarity to all examples for this intent
            intent_similarities = []
            for example in examples:
                example_vector = self._get_query_vector(example.text)
                similarity = self._cosine_similarity(query_vector, example_vector)
                intent_similarities.append(similarity)
            
            # Take the mean of top similarities and boost for better examples
            if intent_similarities:
                intent_similarities.sort(reverse=True)
                top_k = min(3, len(intent_similarities))
                top_similarities = intent_similarities[:top_k]
                
                # Boost score if we have high similarity matches
                base_score = sum(top_similarities) / top_k
                if max(top_similarities) > 0.7:
                    base_score *= 1.3  # Boost for very good matches
                elif max(top_similarities) > 0.5:
                    base_score *= 1.1  # Small boost for good matches
                
                scores[intent] = min(1.0, base_score)  # Cap at 1.0
            else:
                scores[intent] = 0.0
        
        return scores
    
    def _get_query_vector(self, query: str) -> Dict[str, float]:
        """Get TF-IDF vector for a query."""
        words = query.lower().split()
        word_counts = {}
        for word in words:
            if len(word) > 2:  # Skip very short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Simple TF-IDF (without IDF for now)
        total_words = len(words)
        if total_words == 0:
            return {}
        
        vector = {}
        for word, count in word_counts.items():
            vector[word] = count / total_words
        
        return vector
    
    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Get all unique words
        all_words = set(vec1.keys()) | set(vec2.keys())
        
        # Calculate dot product and magnitudes
        dot_product = 0.0
        mag1 = 0.0
        mag2 = 0.0
        
        for word in all_words:
            val1 = vec1.get(word, 0.0)
            val2 = vec2.get(word, 0.0)
            dot_product += val1 * val2
            mag1 += val1 * val1
            mag2 += val2 * val2
        
        # Calculate magnitudes
        mag1 = mag1 ** 0.5
        mag2 = mag2 ** 0.5
        
        # Avoid division by zero
        if mag1 == 0.0 or mag2 == 0.0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def _build_examples(self) -> Dict[AgentIntent, List[IntentExample]]:
        """Build example utterances for each intent."""
        examples = {
            AgentIntent.FORECAST: [
                IntentExample("How will machine learning trend next week?", AgentIntent.FORECAST),
                IntentExample("Forecast the trend for data science", AgentIntent.FORECAST),
                IntentExample("What's the future of AI?", AgentIntent.FORECAST),
                IntentExample("Predict trends for blockchain", AgentIntent.FORECAST),
                IntentExample("Show me forecast for python", AgentIntent.FORECAST),
                IntentExample("Next week's trends for cybersecurity", AgentIntent.FORECAST),
                IntentExample("Predict what will happen with python programming", AgentIntent.FORECAST),
                IntentExample("Forecast machine learning popularity", AgentIntent.FORECAST),
                IntentExample("What will be the trend for AI next month?", AgentIntent.FORECAST),
                IntentExample("Show me future trends for data science", AgentIntent.FORECAST)
            ],
            AgentIntent.COMPARE: [
                IntentExample("Compare python vs javascript", AgentIntent.COMPARE),
                IntentExample("Which is more popular: AI or ML?", AgentIntent.COMPARE),
                IntentExample("Compare trends between react and vue", AgentIntent.COMPARE),
                IntentExample("Show me comparison of data science vs machine learning", AgentIntent.COMPARE),
                IntentExample("Compare machine learning vs artificial intelligence", AgentIntent.COMPARE),
                IntentExample("Which is more popular: python or javascript?", AgentIntent.COMPARE),
                IntentExample("Compare blockchain vs cryptocurrency trends", AgentIntent.COMPARE),
                IntentExample("Show me python vs javascript comparison", AgentIntent.COMPARE),
                IntentExample("Compare AI and machine learning popularity", AgentIntent.COMPARE),
                IntentExample("Which technology is more popular: react or vue?", AgentIntent.COMPARE)
            ],
            AgentIntent.SUMMARY: [
                IntentExample("Give me a summary of trends", AgentIntent.SUMMARY),
                IntentExample("What are the recent trends?", AgentIntent.SUMMARY),
                IntentExample("Show me overview of current trends", AgentIntent.SUMMARY),
                IntentExample("Summarize the trends for me", AgentIntent.SUMMARY),
                IntentExample("Give me a summary of machine learning trends", AgentIntent.SUMMARY),
                IntentExample("What are the recent trends for artificial intelligence?", AgentIntent.SUMMARY),
                IntentExample("Summarize the current state of data science", AgentIntent.SUMMARY),
                IntentExample("Overview of blockchain technology", AgentIntent.SUMMARY),
                IntentExample("Tell me about data science insights", AgentIntent.SUMMARY),
                IntentExample("Show me a summary of current AI trends", AgentIntent.SUMMARY),
                IntentExample("What's the current state of machine learning?", AgentIntent.SUMMARY),
                IntentExample("Give me insights about data science trends", AgentIntent.SUMMARY)
            ],
            AgentIntent.TRAIN: [
                IntentExample("Train a model for forecasting", AgentIntent.TRAIN),
                IntentExample("Build a forecasting model", AgentIntent.TRAIN),
                IntentExample("Create model for trend prediction", AgentIntent.TRAIN),
                IntentExample("Train forecasting model", AgentIntent.TRAIN),
                IntentExample("Train a model for machine learning trends", AgentIntent.TRAIN),
                IntentExample("Build a forecasting model for python programming", AgentIntent.TRAIN),
                IntentExample("Create a model to predict data science trends", AgentIntent.TRAIN),
                IntentExample("Develop an algorithm for AI trend prediction", AgentIntent.TRAIN),
                IntentExample("Model training for AI applications", AgentIntent.TRAIN),
                IntentExample("Train a forecasting model for blockchain", AgentIntent.TRAIN),
                IntentExample("Build a model to predict trends", AgentIntent.TRAIN),
                IntentExample("Create a machine learning model for forecasting", AgentIntent.TRAIN)
            ],
            AgentIntent.EVALUATE: [
                IntentExample("Evaluate model performance", AgentIntent.EVALUATE),
                IntentExample("How accurate are the models?", AgentIntent.EVALUATE),
                IntentExample("Assess model quality", AgentIntent.EVALUATE),
                IntentExample("Check model performance", AgentIntent.EVALUATE),
                IntentExample("Evaluate the performance of my models", AgentIntent.EVALUATE),
                IntentExample("How accurate are the forecasting models?", AgentIntent.EVALUATE),
                IntentExample("Assess the quality of predictions", AgentIntent.EVALUATE),
                IntentExample("Test the accuracy of predictions", AgentIntent.EVALUATE),
                IntentExample("How good is the model?", AgentIntent.EVALUATE),
                IntentExample("Evaluate forecasting model accuracy", AgentIntent.EVALUATE),
                IntentExample("Check how well the model performs", AgentIntent.EVALUATE),
                IntentExample("Assess prediction quality", AgentIntent.EVALUATE)
            ],
            AgentIntent.HEALTH: [
                IntentExample("Is the service working?", AgentIntent.HEALTH),
                IntentExample("Check system health", AgentIntent.HEALTH),
                IntentExample("Service status", AgentIntent.HEALTH),
                IntentExample("Are you alive?", AgentIntent.HEALTH),
                IntentExample("What's the system status?", AgentIntent.HEALTH),
                IntentExample("Are you up and running?", AgentIntent.HEALTH),
                IntentExample("Is everything okay?", AgentIntent.HEALTH),
                IntentExample("System health check", AgentIntent.HEALTH),
                IntentExample("Check if the service is working", AgentIntent.HEALTH),
                IntentExample("Is the system operational?", AgentIntent.HEALTH),
                IntentExample("Service health status", AgentIntent.HEALTH),
                IntentExample("Are you functioning properly?", AgentIntent.HEALTH)
            ],
            AgentIntent.LIST_MODELS: [
                IntentExample("Show me available models", AgentIntent.LIST_MODELS),
                IntentExample("List all models", AgentIntent.LIST_MODELS),
                IntentExample("What models do you have?", AgentIntent.LIST_MODELS),
                IntentExample("Show models", AgentIntent.LIST_MODELS),
                IntentExample("Which models are available?", AgentIntent.LIST_MODELS),
                IntentExample("Show existing models", AgentIntent.LIST_MODELS),
                IntentExample("List available forecasting models", AgentIntent.LIST_MODELS),
                IntentExample("What forecasting models do you have?", AgentIntent.LIST_MODELS),
                IntentExample("Show me all available models", AgentIntent.LIST_MODELS),
                IntentExample("List the models you have", AgentIntent.LIST_MODELS),
                IntentExample("What models are available for use?", AgentIntent.LIST_MODELS),
                IntentExample("Show me the list of models", AgentIntent.LIST_MODELS)
            ]
        }
        
        # Initialize empty lists for other intents
        for intent in AgentIntent:
            if intent not in examples:
                examples[intent] = []
        
        return examples
    
    def _build_tfidf_vectors(self) -> Dict[str, Dict[str, float]]:
        """Build TF-IDF vectors for all examples."""
        # This is a simplified implementation
        # In a real system, you'd use a proper TF-IDF vectorizer
        return {}


class IntentRecognizer:
    """Hybrid intent recognizer combining multiple scoring methods."""
    
    def __init__(self):
        """Initialize the hybrid intent recognizer."""
        self.semantic_scorer = SimpleSemanticScorer()
        self.text_normalizer = TextNormalizer()
        self.logger = logging.getLogger(__name__)
        self.examples = {intent: examples[:] for intent, examples in self.semantic_scorer.intent_examples.items()}
        self.patterns = self._build_regex_patterns()
        self.weights = {
            ScorerType.SEMANTIC: 0.6,
            ScorerType.REGEX: 0.3,
            ScorerType.LLM: 0.1
        }
        self.confidence_thresholds = {
            'high': 0.2,
            'low': 0.05
        }
        self.ensemble_weights = {
            ScorerType.SEMANTIC: Config.INTENT_LLM_ENSEMBLE_WEIGHTS["semantic"],
            ScorerType.REGEX: Config.INTENT_LLM_ENSEMBLE_WEIGHTS["regex"],
            ScorerType.LLM: Config.INTENT_LLM_ENSEMBLE_WEIGHTS["llm"]
        }
    
    def recognize_intent(self, query: str, raw_text: Optional[str] = None, is_normalized: bool = False) -> 'IntentRecognition':
        """Recognize intent from a natural language query.
        
        Args:
            query: The user query (can be pre-normalized)
            raw_text: Original raw text (for logging/statistics)
            is_normalized: Whether the query is already normalized (default: False)
            
        Returns:
            IntentRecognition with intent and confidence
        """
        # Store original text for reference
        original_text = raw_text if raw_text is not None else query
        
        # C1.2: Implement normalization skip logic
        if is_normalized:
            # Skip normalization if already normalized
            normalized_text = query
            norm_stats = {"skipped": True, "reason": "already_normalized"}
        else:
            # Normalize text for processing
            normalized_text, norm_stats = self.text_normalizer.normalize(query)
        
        # Get scores from different scorers
        semantic_result = self._semantic_scorer(normalized_text)
        regex_result = self._regex_scorer(normalized_text)
        llm_result = self._llm_scorer(normalized_text)
        
        # Ensemble the scores
        intent, confidence = self._ensemble_scores([semantic_result, regex_result, llm_result])
        
        # Apply confidence thresholds
        if confidence >= self.confidence_thresholds['high']:
            final_intent = intent
        elif confidence >= self.confidence_thresholds['low']:
            final_intent = intent  # Best guess with low confidence
        else:
            final_intent = AgentIntent.UNKNOWN
        
        # Create result with additional metadata
        from app.models.agent_models import IntentRecognition
        result = IntentRecognition(
            intent=final_intent,
            confidence=confidence,
            raw_text=original_text,
            normalized_text=normalized_text,
            normalization_stats=norm_stats
        )
        
        return result
    
    def _semantic_scorer(self, query: str) -> ScorerResult:
        """Compute semantic similarity scores using simple TF-IDF.
        
        Args:
            query: The user query
            
        Returns:
            ScorerResult with semantic scores
        """
        try:
            scores = self.semantic_scorer.score_query(query)
            
            # Ensure all intents are included in scores and are floats
            for intent in AgentIntent:
                if intent != AgentIntent.UNKNOWN:
                    if intent not in scores:
                        scores[intent] = 0.0
                    else:
                        scores[intent] = float(scores[intent])
            
            # Calculate overall confidence
            if scores:
                confidence = max(scores.values())
            else:
                confidence = 0.0
            
            return ScorerResult(
                scorer_type=ScorerType.SEMANTIC,
                scores=scores,
                confidence=confidence
            )
        except Exception as e:
            # Fallback to zero scores
            scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
            return ScorerResult(
                scorer_type=ScorerType.SEMANTIC,
                scores=scores,
                confidence=0.0,
                valid=False  # Mark as invalid for weight redistribution
            )
    
    def _regex_scorer(self, query: str) -> ScorerResult:
        """Compute regex pattern matching scores.
        
        Args:
            query: The user query
            
        Returns:
            ScorerResult with regex scores
        """
        patterns = self._build_regex_patterns()
        scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
        
        for pattern in patterns:
            if re.search(pattern.pattern, query, re.IGNORECASE):
                scores[pattern.intent] += pattern.weight
        
        # Normalize scores and boost for multiple matches
        max_score = max(scores.values()) if scores else 0.0
        if max_score > 0:
            for intent in scores:
                normalized_score = scores[intent] / max_score
                # Boost score if we have multiple pattern matches
                if scores[intent] > 1.0:
                    normalized_score *= 1.2  # Boost for multiple matches
                scores[intent] = min(1.0, normalized_score)
        
        confidence = max(scores.values()) if scores else 0.0
        
        return ScorerResult(
            scorer_type=ScorerType.REGEX,
            scores=scores,
            confidence=confidence
        )
    
    def _llm_scorer(self, query: str) -> ScorerResult:
        """Compute LLM-based classification scores.
        
        Args:
            query: The user query
            
        Returns:
            ScorerResult with LLM scores
        """
        # Early return if LLM is disabled
        if not Config.INTENT_LLM_ENABLED:
            scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
            return ScorerResult(
                scorer_type=ScorerType.LLM,
                scores=scores,
                confidence=0.0,
                valid=False  # Not valid when disabled
            )
        
        try:
            # Import here to avoid circular imports
            from app.client.llm import LLMClient, OpenAIClient, LocalClient, IntentCache
            from app.client.llm.intent_cache import hash_query
            
            # Initialize LLM client if not already done
            if not hasattr(self, '_llm_client'):
                self._llm_client = self._create_llm_client()
                self._llm_cache = IntentCache(
                    max_size=Config.INTENT_LLM_CACHE_SIZE,
                    ttl_hours=Config.INTENT_LLM_CACHE_TTL_HOURS
                )
            
            # Check cache first
            query_hash = hash_query(query)
            cached_result = self._llm_cache.get(query_hash)
            if cached_result:
                return self._format_cached_result(cached_result)
            
            # Make LLM classification
            result = self._llm_client.classify_intent(query)
            
            # Convert string intent to AgentIntent
            intent_str = result.intent
            if intent_str == "forecast":
                intent = AgentIntent.FORECAST
            elif intent_str == "compare":
                intent = AgentIntent.COMPARE
            elif intent_str == "summary":
                intent = AgentIntent.SUMMARY
            elif intent_str == "train":
                intent = AgentIntent.TRAIN
            elif intent_str == "evaluate":
                intent = AgentIntent.EVALUATE
            elif intent_str == "health":
                intent = AgentIntent.HEALTH
            elif intent_str == "list_models":
                intent = AgentIntent.LIST_MODELS
            else:
                intent = AgentIntent.UNKNOWN
            
            # Create scores dict
            scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
            scores[intent] = result.confidence
            
            # Cache the result
            self._llm_cache.set(query_hash, {
                "intent": intent,
                "confidence": result.confidence,
                "scores": scores
            }, result.model_version)
            
            return ScorerResult(
                scorer_type=ScorerType.LLM,
                scores=scores,
                confidence=result.confidence
            )
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
            # Return zero scores on failure - the ensemble method will handle weight redistribution
            scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
            return ScorerResult(
                scorer_type=ScorerType.LLM,
                scores=scores,
                confidence=0.0,
                valid=False  # Mark this scorer as invalid
            )
    
    def _create_llm_client(self):
        """Create LLM client based on configuration.
        
        Returns:
            Configured LLM client
        """
        # Import here to avoid circular imports
        from app.client.llm import OpenAIClient, LocalClient
        
        if Config.INTENT_LLM_PROVIDER == "openai":
            if not Config.INTENT_LLM_API_KEY:
                raise ValueError("OpenAI API key not configured")
            return OpenAIClient(
                api_key=Config.INTENT_LLM_API_KEY,
                model=Config.INTENT_LLM_MODEL,
                timeout_ms=Config.INTENT_LLM_TIMEOUT_MS,
                max_tokens=Config.INTENT_LLM_MAX_TOKENS,
                temperature=Config.INTENT_LLM_TEMPERATURE
            )
        elif Config.INTENT_LLM_PROVIDER == "local":
            return LocalClient(
                base_url=Config.INTENT_LLM_BASE_URL,
                model=Config.INTENT_LLM_MODEL,
                timeout_ms=Config.INTENT_LLM_TIMEOUT_MS,
                max_tokens=Config.INTENT_LLM_MAX_TOKENS,
                temperature=Config.INTENT_LLM_TEMPERATURE
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {Config.INTENT_LLM_PROVIDER}")
    
    def _format_cached_result(self, cached_result: Dict) -> ScorerResult:
        """Format cached result as ScorerResult.
        
        Args:
            cached_result: Cached classification result
            
        Returns:
            ScorerResult from cached data
        """
        return ScorerResult(
            scorer_type=ScorerType.LLM,
            scores=cached_result["scores"],
            confidence=cached_result["confidence"]
        )
    
    def _ensemble_scores(self, scorer_results: List[ScorerResult]) -> Tuple[AgentIntent, float]:
        """Combine scores from multiple scorers using weighted ensemble.
        
        Args:
            scorer_results: List of ScorerResult objects
            
        Returns:
            Tuple of (best_intent, confidence)
        """
        ensemble_scores = {intent: 0.0 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
        
        # Calculate dynamic weights based on failed scorers
        dynamic_weights = self._calculate_dynamic_weights(scorer_results)
        
        for result in scorer_results:
            weight = dynamic_weights.get(result.scorer_type, 0.0)
            for intent, score in result.scores.items():
                if intent != AgentIntent.UNKNOWN:
                    ensemble_scores[intent] += weight * score
        
        # Find the best intent
        if ensemble_scores:
            best_intent = max(ensemble_scores.items(), key=lambda x: x[1])
            return best_intent[0], best_intent[1]
        else:
            return AgentIntent.UNKNOWN, 0.0
    
    def _calculate_dynamic_weights(self, scorer_results: List[ScorerResult]) -> Dict[ScorerType, float]:
        """Calculate dynamic weights, redistributing invalid scorer weights proportionally.
        
        Args:
            scorer_results: List of ScorerResult objects
            
        Returns:
            Dictionary mapping ScorerType to adjusted weight
        """
        # Start with original weights
        dynamic_weights = self.ensemble_weights.copy()
        
        # Identify invalid scorers
        invalid_scorers = [result.scorer_type for result in scorer_results if not result.valid]
        
        if not invalid_scorers:
            # No invalid scorers, use original weights
            return dynamic_weights
        
        # Calculate total weight of invalid scorers
        invalid_weight = sum(dynamic_weights.get(scorer_type, 0.0) for scorer_type in invalid_scorers)
        
        if invalid_weight == 0.0:
            # No weight to redistribute
            return dynamic_weights
        
        # Calculate total weight of valid scorers
        valid_scorers = [result.scorer_type for result in scorer_results if result.valid]
        valid_weight = sum(dynamic_weights.get(scorer_type, 0.0) for scorer_type in valid_scorers)
        
        if valid_weight == 0.0:
            # No valid scorers, return original weights
            return dynamic_weights
        
        # Redistribute invalid weight proportionally to valid scorers
        redistribution_factor = invalid_weight / valid_weight
        
        for scorer_type in valid_scorers:
            original_weight = dynamic_weights.get(scorer_type, 0.0)
            dynamic_weights[scorer_type] = original_weight * (1.0 + redistribution_factor)
        
        # Zero out weights for invalid scorers
        for scorer_type in invalid_scorers:
            dynamic_weights[scorer_type] = 0.0
        
        # Log the weight redistribution for debugging
        if invalid_scorers:
            self.logger.info(f"Weight redistribution due to invalid scorers: {invalid_scorers}. "
                           f"Original weights: {self.ensemble_weights}, "
                           f"Adjusted weights: {dynamic_weights}")
        
        return dynamic_weights
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and extra whitespace
        import re
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _build_regex_patterns(self) -> List[IntentPattern]:
        """Build regex patterns for intent recognition."""
        return [
            # Forecast patterns - more specific
            IntentPattern(r'\b(forecast|predict)\b.*\b(trend|future|outlook)\b', AgentIntent.FORECAST, 2.0),
            IntentPattern(r'\b(how will|what will|will)\b.*\b(trend|popular|popularity)\b', AgentIntent.FORECAST, 2.0),
            IntentPattern(r'\b(next week|next month|future)\b.*\b(trends?)\b', AgentIntent.FORECAST, 1.5),
            IntentPattern(r'\b(show me|give me)\b.*\b(forecast|prediction)\b', AgentIntent.FORECAST, 1.5),
            IntentPattern(r'\b(forecast|predict)\b.*\b(for|of)\b', AgentIntent.FORECAST, 1.0),
            
            # Compare patterns - more specific
            IntentPattern(r'\b(compare|comparison)\b.*\b(vs|versus|and)\b', AgentIntent.COMPARE, 2.0),
            IntentPattern(r'\b(which is|what is).*\b(more|better|popular)\b', AgentIntent.COMPARE, 2.0),
            IntentPattern(r'\b(compare|comparison)\b', AgentIntent.COMPARE, 1.5),
            IntentPattern(r'\b(vs|versus)\b', AgentIntent.COMPARE, 1.0),
            
            # Summary patterns - more specific
            IntentPattern(r'\b(give me|show me|tell me)\b.*\b(summary|overview|insights)\b', AgentIntent.SUMMARY, 2.0),
            IntentPattern(r'\b(summary|overview|insights)\b.*\b(of|for)\b', AgentIntent.SUMMARY, 1.5),
            IntentPattern(r'\b(what are|what\'s)\b.*\b(recent|current)\b.*\b(trends?)\b', AgentIntent.SUMMARY, 2.0),
            IntentPattern(r'\b(summarize|summarise)\b', AgentIntent.SUMMARY, 1.5),
            IntentPattern(r'\b(current state|recent trends)\b', AgentIntent.SUMMARY, 1.0),
            
            # Train patterns - more specific
            IntentPattern(r'\b(train|build|create|develop)\b.*\b(model|algorithm)\b', AgentIntent.TRAIN, 2.0),
            IntentPattern(r'\b(model training|training)\b', AgentIntent.TRAIN, 2.0),
            IntentPattern(r'\b(train|build|create)\b.*\b(for|to)\b.*\b(predict|forecast)\b', AgentIntent.TRAIN, 2.0),
            IntentPattern(r'\b(create|build)\b.*\b(machine learning|ml|ai)\b.*\b(model)\b', AgentIntent.TRAIN, 1.5),
            
            # Evaluate patterns - more specific
            IntentPattern(r'\b(evaluate|assess|check|test)\b.*\b(performance|accuracy|quality)\b', AgentIntent.EVALUATE, 2.0),
            IntentPattern(r'\b(how accurate|how good)\b.*\b(model|prediction)\b', AgentIntent.EVALUATE, 2.0),
            IntentPattern(r'\b(model quality|prediction quality|forecast accuracy)\b', AgentIntent.EVALUATE, 1.5),
            IntentPattern(r'\b(evaluate|assess)\b.*\b(of|the)\b', AgentIntent.EVALUATE, 1.0),
            
            # Health patterns - more specific
            IntentPattern(r'\b(is|are)\b.*\b(working|alive|okay|operational)\b', AgentIntent.HEALTH, 2.0),
            IntentPattern(r'\b(service|system)\b.*\b(status|health)\b', AgentIntent.HEALTH, 2.0),
            IntentPattern(r'\b(health check|system check)\b', AgentIntent.HEALTH, 2.0),
            IntentPattern(r'\b(are you|is the service)\b.*\b(up|running|working)\b', AgentIntent.HEALTH, 1.5),
            IntentPattern(r'\b(health|status)\b', AgentIntent.HEALTH, 1.0),
            
            # List models patterns - more specific
            IntentPattern(r'\b(list|show|display)\b.*\b(models?)\b', AgentIntent.LIST_MODELS, 2.0),
            IntentPattern(r'\b(what|which)\b.*\b(models?)\b.*\b(available|do you have)\b', AgentIntent.LIST_MODELS, 2.0),
            IntentPattern(r'\b(available|all)\b.*\b(models?)\b', AgentIntent.LIST_MODELS, 1.5),
            IntentPattern(r'\b(show me|list)\b.*\b(available|all)\b', AgentIntent.LIST_MODELS, 1.0),
        ]
    
    def add_example(self, text: str, intent: AgentIntent):
        """Add a new example for active learning.
        
        Args:
            text: Example text
            intent: Target intent
        """
        example = IntentExample(text, intent)
        if intent in self.semantic_scorer.intent_examples:
            self.semantic_scorer.intent_examples[intent].append(example)
            # Update the examples attribute for consistency
            if intent in self.examples:
                self.examples[intent].append(example)
    
    def get_intent_examples(self, intent: AgentIntent) -> List[str]:
        """Get examples for a specific intent.
        
        Args:
            intent: The intent to get examples for
            
        Returns:
            List of example texts for the intent
        """
        examples = self.semantic_scorer.intent_examples.get(intent, [])
        return [example.text for example in examples] 
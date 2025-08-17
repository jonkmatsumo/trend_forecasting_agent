"""
Hybrid Intent Recognizer
Advanced intent recognition using semantic similarity, regex patterns, and ensemble scoring.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.spatial.distance import cosine

from app.models.agent_models import AgentIntent, IntentRecognition, create_intent_recognition


class ScorerType(str, Enum):
    """Types of scorers in the ensemble."""
    SEMANTIC = "semantic"
    REGEX = "regex"
    LLM = "llm"


@dataclass
class ScorerResult:
    """Result from a single scorer."""
    scorer_type: ScorerType
    scores: Dict[AgentIntent, float]
    confidence: float


@dataclass
class IntentExample:
    """Example utterance for an intent."""
    text: str
    intent: AgentIntent
    embedding: Optional[np.ndarray] = None


@dataclass
class IntentPattern:
    """Pattern for intent recognition (now used as guardrails)."""
    intent: AgentIntent
    patterns: List[str]
    confidence: float
    required_keywords: Optional[List[str]] = None
    excluded_keywords: Optional[List[str]] = None


class SimpleSemanticScorer:
    """Simple semantic scorer using TF-IDF and cosine similarity."""
    
    def __init__(self):
        self.vocabulary = set()
        self.intent_vectors = {}
        self._build_vocabulary()
        self._build_intent_vectors()
    
    def _build_vocabulary(self):
        """Build vocabulary from all examples."""
        examples = self._get_all_examples()
        for example in examples:
            words = self._tokenize(example.text)
            self.vocabulary.update(words)
        self.vocabulary = sorted(list(self.vocabulary))
    
    def _get_all_examples(self) -> List[IntentExample]:
        """Get all example utterances."""
        examples = []
        
        # Forecast examples
        forecast_examples = [
            "How will machine learning trend next week?",
            "What will happen with artificial intelligence in the future?",
            "Forecast the trend for data science",
            "Predict what will happen with python programming",
            "Show me the forecast for blockchain technology",
            "What's the future outlook for cloud computing?",
            "Can you predict the trend for cybersecurity?",
            "How will AI evolve in the coming months?"
        ]
        
        # Compare examples
        compare_examples = [
            "Compare machine learning vs artificial intelligence",
            "Which is more popular: python or javascript?",
            "Show me the difference between data science and analytics",
            "Compare blockchain vs cryptocurrency trends",
            "Which technology is trending more: AI or ML?",
            "What's the difference between cloud and edge computing?",
            "Compare the popularity of React vs Angular",
            "Which programming language is more in demand?"
        ]
        
        # Summary examples
        summary_examples = [
            "Give me a summary of machine learning trends",
            "Show me the current data for python programming",
            "What are the recent trends for artificial intelligence?",
            "Summarize the current state of data science",
            "Tell me about blockchain technology trends",
            "What's happening with cloud computing lately?",
            "Give me an overview of cybersecurity trends",
            "What's the current status of AI development?"
        ]
        
        # Train examples
        train_examples = [
            "Train a model for machine learning trends",
            "Build a forecasting model for python programming",
            "Create a model to predict data science trends",
            "Develop an algorithm for AI trend prediction",
            "Train a new forecasting model",
            "Build a model for blockchain analysis",
            "Create a prediction model for cybersecurity",
            "Develop a forecasting algorithm"
        ]
        
        # Evaluate examples
        evaluate_examples = [
            "Evaluate the performance of my models",
            "How accurate are the forecasting models?",
            "Show me the model evaluation metrics",
            "Assess the quality of predictions",
            "Test the model performance",
            "How well are the models performing?",
            "Evaluate the accuracy of forecasts",
            "What's the model performance like?"
        ]
        
        # Health examples
        health_examples = [
            "Is the service working?",
            "What's the system status?",
            "Are you up and running?",
            "Is everything okay?",
            "Check if the system is healthy",
            "Are the services operational?",
            "What's the health status?",
            "Is the system functioning properly?"
        ]
        
        # List models examples
        list_models_examples = [
            "List all models",
            "Show me available models",
            "What models do you have?",
            "Display the trained models",
            "Show available forecasting models",
            "List the models I can use",
            "What forecasting models are available?",
            "Show me all trained algorithms"
        ]
        
        # Add examples to the list
        for text in forecast_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.FORECAST))
        for text in compare_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.COMPARE))
        for text in summary_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.SUMMARY))
        for text in train_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.TRAIN))
        for text in evaluate_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.EVALUATE))
        for text in health_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.HEALTH))
        for text in list_models_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.LIST_MODELS))
            
        return examples
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        # Convert to lowercase and split
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to TF-IDF vector."""
        words = self._tokenize(text)
        vector = np.zeros(len(self.vocabulary))
        
        # Simple TF (term frequency)
        for word in words:
            if word in self.vocabulary:
                idx = self.vocabulary.index(word)
                vector[idx] += 1
        
        # Normalize
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def _build_intent_vectors(self):
        """Build representative vectors for each intent."""
        examples = self._get_all_examples()
        intent_examples = {}
        
        # Group examples by intent
        for example in examples:
            if example.intent not in intent_examples:
                intent_examples[example.intent] = []
            intent_examples[example.intent].append(example)
        
        # Build average vector for each intent
        for intent, examples_list in intent_examples.items():
            if intent == AgentIntent.UNKNOWN:
                continue
            
            vectors = []
            for example in examples_list:
                vector = self._text_to_vector(example.text)
                vectors.append(vector)
            
            if vectors:
                # Average vector for the intent
                avg_vector = np.mean(vectors, axis=0)
                self.intent_vectors[intent] = avg_vector
    
    def score_query(self, query: str) -> Dict[AgentIntent, float]:
        """Score a query against all intents."""
        query_vector = self._text_to_vector(query)
        scores = {}
        
        for intent, intent_vector in self.intent_vectors.items():
            if intent == AgentIntent.UNKNOWN:
                continue
            
            # Cosine similarity with safety checks
            try:
                # Check for zero vectors
                if np.sum(query_vector) == 0 or np.sum(intent_vector) == 0:
                    similarity = 0.0
                else:
                    # Normalize vectors to unit length
                    query_norm = query_vector / np.linalg.norm(query_vector)
                    intent_norm = intent_vector / np.linalg.norm(intent_vector)
                    similarity = np.dot(query_norm, intent_norm)
                    similarity = max(0, similarity)  # Ensure non-negative
            except Exception:
                similarity = 0.0
            
            scores[intent] = similarity
        
        return scores


class HybridIntentRecognizer:
    """Hybrid intent recognition with semantic similarity, regex patterns, and ensemble scoring."""
    
    def __init__(self):
        """Initialize the hybrid intent recognizer."""
        self.logger = logging.getLogger(__name__)
        self.examples = self._build_examples()
        self.patterns = self._build_patterns()
        self.semantic_scorer = SimpleSemanticScorer()
        self.weights = {
            ScorerType.SEMANTIC: 0.6,
            ScorerType.REGEX: 0.3,
            ScorerType.LLM: 0.1
        }
        self.confidence_thresholds = {
            'high': 0.45,  # Lowered from 0.55
            'low': 0.25    # Lowered from 0.4
        }
        
    def _build_examples(self) -> List[IntentExample]:
        """Build example utterances for each intent."""
        examples = []
        
        # Forecast examples
        forecast_examples = [
            "How will machine learning trend next week?",
            "What will happen with artificial intelligence in the future?",
            "Forecast the trend for data science",
            "Predict what will happen with python programming",
            "Show me the forecast for blockchain technology",
            "What's the future outlook for cloud computing?",
            "Can you predict the trend for cybersecurity?",
            "How will AI evolve in the coming months?"
        ]
        
        # Compare examples
        compare_examples = [
            "Compare machine learning vs artificial intelligence",
            "Which is more popular: python or javascript?",
            "Show me the difference between data science and analytics",
            "Compare blockchain vs cryptocurrency trends",
            "Which technology is trending more: AI or ML?",
            "What's the difference between cloud and edge computing?",
            "Compare the popularity of React vs Angular",
            "Which programming language is more in demand?"
        ]
        
        # Summary examples
        summary_examples = [
            "Give me a summary of machine learning trends",
            "Show me the current data for python programming",
            "What are the recent trends for artificial intelligence?",
            "Summarize the current state of data science",
            "Tell me about blockchain technology trends",
            "What's happening with cloud computing lately?",
            "Give me an overview of cybersecurity trends",
            "What's the current status of AI development?"
        ]
        
        # Train examples
        train_examples = [
            "Train a model for machine learning trends",
            "Build a forecasting model for python programming",
            "Create a model to predict data science trends",
            "Develop an algorithm for AI trend prediction",
            "Train a new forecasting model",
            "Build a model for blockchain analysis",
            "Create a prediction model for cybersecurity",
            "Develop a forecasting algorithm"
        ]
        
        # Evaluate examples
        evaluate_examples = [
            "Evaluate the performance of my models",
            "How accurate are the forecasting models?",
            "Show me the model evaluation metrics",
            "Assess the quality of predictions",
            "Test the model performance",
            "How well are the models performing?",
            "Evaluate the accuracy of forecasts",
            "What's the model performance like?"
        ]
        
        # Health examples
        health_examples = [
            "Is the service working?",
            "What's the system status?",
            "Are you up and running?",
            "Is everything okay?",
            "Check if the system is healthy",
            "Are the services operational?",
            "What's the health status?",
            "Is the system functioning properly?"
        ]
        
        # List models examples
        list_models_examples = [
            "List all models",
            "Show me available models",
            "What models do you have?",
            "Display the trained models",
            "Show available forecasting models",
            "List the models I can use",
            "What forecasting models are available?",
            "Show me all trained algorithms"
        ]
        
        # Add examples to the list
        for text in forecast_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.FORECAST))
        for text in compare_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.COMPARE))
        for text in summary_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.SUMMARY))
        for text in train_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.TRAIN))
        for text in evaluate_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.EVALUATE))
        for text in health_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.HEALTH))
        for text in list_models_examples:
            examples.append(IntentExample(text=text, intent=AgentIntent.LIST_MODELS))
            
        return examples
    
    def _build_patterns(self) -> List[IntentPattern]:
        """Build intent recognition patterns (now used as guardrails)."""
        return [
            # Forecast intent patterns
            IntentPattern(
                intent=AgentIntent.FORECAST,
                patterns=[
                    r'\b(forecast|predict|future|next|upcoming|project)\b',
                    r'\b(what will|how will|when will)\b',
                    r'\b(trend|direction|outlook)\b'
                ],
                confidence=0.9,
                required_keywords=None,
                excluded_keywords=['compare', 'versus', 'vs', 'difference', 'train', 'build', 'create', 'develop', 'evaluate', 'assess', 'test', 'performance', 'accuracy', 'health', 'working', 'alive', 'okay', 'list', 'show', 'display', 'models']
            ),
            
            # Compare intent patterns
            IntentPattern(
                intent=AgentIntent.COMPARE,
                patterns=[
                    r'\b(compare|versus|vs|difference|contrast)\b',
                    r'\b(which is|which has|better|worse)\b',
                    r'\b(versus|against|between)\b'
                ],
                confidence=0.9,
                required_keywords=None,
                excluded_keywords=['train', 'build', 'create', 'develop', 'evaluate', 'assess', 'test', 'performance', 'accuracy', 'health', 'working', 'alive', 'okay', 'list', 'show', 'display', 'models']
            ),
            
            # Summary intent patterns
            IntentPattern(
                intent=AgentIntent.SUMMARY,
                patterns=[
                    r'\b(summary|overview|insights)\b',
                    r'\b(show me|tell me about|what is)\b',
                    r'\b(current|recent|latest)\b'
                ],
                confidence=0.8,
                required_keywords=None,
                excluded_keywords=['forecast', 'predict', 'future', 'compare', 'versus', 'vs', 'train', 'build', 'create', 'develop', 'evaluate', 'assess', 'test', 'performance', 'accuracy', 'health', 'working', 'alive', 'okay', 'list', 'show', 'display', 'models']
            ),
            
            # Train intent patterns
            IntentPattern(
                intent=AgentIntent.TRAIN,
                patterns=[
                    r'\b(train|build|create|develop)\b',
                    r'\b(algorithm)\b',
                    r'\b(training|modeling)\b'
                ],
                confidence=0.9,
                required_keywords=None,
                excluded_keywords=['summary', 'overview', 'insights', 'evaluate', 'assess', 'test', 'performance', 'accuracy', 'health', 'working', 'alive', 'okay', 'list', 'show', 'display', 'system']
            ),
            
            # Evaluate intent patterns
            IntentPattern(
                intent=AgentIntent.EVALUATE,
                patterns=[
                    r'\b(evaluate|assess|test|performance|accuracy)\b',
                    r'\b(how good|how accurate|how well)\b',
                    r'\b(metrics|scores|results)\b',
                    r'\b(evaluation|assessment|testing)\b'
                ],
                confidence=0.9,
                required_keywords=None,
                excluded_keywords=['summary', 'overview', 'insights', 'list', 'show', 'display', 'train', 'build', 'create', 'develop', 'health', 'working', 'alive', 'okay']
            ),
            
            # Health intent patterns
            IntentPattern(
                intent=AgentIntent.HEALTH,
                patterns=[
                    r'\b(health|working|alive|okay)\b',
                    r'\b(is it|are you|system)\b',
                    r'\b(up|down|running)\b'
                ],
                confidence=0.8,
                required_keywords=None,
                excluded_keywords=['summary', 'overview', 'insights', 'trends', 'data', 'train', 'build', 'create', 'develop', 'evaluate', 'assess', 'test', 'performance', 'accuracy', 'list', 'show', 'display', 'models']
            ),
            
            # List models intent patterns
            IntentPattern(
                intent=AgentIntent.LIST_MODELS,
                patterns=[
                    r'\b(list|show|display|what|models?)\b',
                    r'\b(available|trained|existing)\b',
                    r'\b(which models?|what models?)\b'
                ],
                confidence=0.8,
                required_keywords=['model', 'models'],
                excluded_keywords=['evaluate', 'assess', 'test', 'performance', 'accuracy', 'train', 'build', 'create', 'develop', 'health', 'working', 'alive', 'okay']
            ),
        ]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Convert to lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove punctuation (but keep apostrophes for contractions)
        normalized = re.sub(r'[^\w\s\']', '', normalized)
        
        # Remove apostrophes (convert contractions to separate words)
        normalized = re.sub(r'\'', '', normalized)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
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
                
            return ScorerResult(ScorerType.SEMANTIC, scores, confidence)
            
        except Exception as e:
            self.logger.error(f"Error in semantic scoring: {e}")
            scores = {intent: 0.5 for intent in AgentIntent if intent != AgentIntent.UNKNOWN}
            return ScorerResult(ScorerType.SEMANTIC, scores, 0.5)
    
    def _regex_scorer(self, query: str) -> ScorerResult:
        """Compute regex pattern scores (now used as guardrails).
        
        Args:
            query: The user query
            
        Returns:
            ScorerResult with regex scores
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        scores = {}
        max_confidence = 0.0
        
        # Initialize scores for all intents
        for intent in AgentIntent:
            if intent != AgentIntent.UNKNOWN:
                scores[intent] = 0.0
        
        # Calculate pattern confidence for each pattern
        for pattern in self.patterns:
            confidence = self._calculate_pattern_confidence(pattern, query_lower, query_words)
            scores[pattern.intent] = confidence
            max_confidence = max(max_confidence, confidence)
        
        return ScorerResult(ScorerType.REGEX, scores, max_confidence)
    
    def _calculate_pattern_confidence(
        self, 
        pattern: IntentPattern, 
        query_lower: str, 
        query_words: set
    ) -> float:
        """Calculate confidence score for a pattern match.
        
        Args:
            pattern: The intent pattern to match against
            query_lower: Lowercase query text
            query_words: Set of words in the query
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.0
        
        # Check for excluded keywords
        if pattern.excluded_keywords:
            for excluded in pattern.excluded_keywords:
                if excluded in query_lower:
                    return 0.0  # Immediate disqualification
        
        # Check for required keywords
        if pattern.required_keywords:
            required_found = any(
                required in query_lower for required in pattern.required_keywords
            )
            if not required_found:
                return 0.0  # Missing required keywords
        
        # Check pattern matches
        pattern_matches = 0
        for regex_pattern in pattern.patterns:
            if re.search(regex_pattern, query_lower):
                pattern_matches += 1
        
        if pattern_matches > 0:
            # Base confidence from pattern
            confidence = pattern.confidence
            
            # Increase confidence based on number of pattern matches
            if pattern_matches > 1:
                confidence = min(confidence + 0.1, 1.0)
        
        return confidence
    
    def _llm_scorer(self, query: str) -> ScorerResult:
        """Compute LLM-based scores (placeholder for future implementation).
        
        Args:
            query: The user query
            
        Returns:
            ScorerResult with LLM scores
        """
        # Placeholder for LLM-based scoring
        # This could be implemented with a small LLM classifier
        # For now, return uniform scores for all intents
        scores = {}
        for intent in AgentIntent:
            if intent != AgentIntent.UNKNOWN:
                scores[intent] = 0.5
        return ScorerResult(ScorerType.LLM, scores, 0.5)
    
    def _ensemble_scores(self, scorers: List[ScorerResult]) -> Tuple[AgentIntent, float]:
        """Combine scores from multiple scorers using weighted ensemble.
        
        Args:
            scorers: List of scorer results
            
        Returns:
            Tuple of (best_intent, confidence)
        """
        # Initialize combined scores for all intents
        combined_scores = {}
        for intent in AgentIntent:
            if intent != AgentIntent.UNKNOWN:
                combined_scores[intent] = 0.0
        
        # Combine scores from all scorers
        for scorer in scorers:
            weight = self.weights.get(scorer.scorer_type, 0.0)
            
            for intent, score in scorer.scores.items():
                if intent in combined_scores:
                    combined_scores[intent] += weight * score
        
        # Find the best intent
        if combined_scores:
            best_intent = max(combined_scores, key=combined_scores.get)
            best_score = combined_scores[best_intent]
        else:
            best_intent = AgentIntent.UNKNOWN
            best_score = 0.0
        
        return best_intent, best_score
    
    def recognize_intent(self, query: str) -> IntentRecognition:
        """Recognize intent from natural language query using hybrid approach.
        
        Args:
            query: The natural language query
            
        Returns:
            IntentRecognition with intent and confidence
        """
        # Normalize the query
        normalized_query = self._normalize_text(query)
        
        # Run all scorers
        scorers = [
            self._semantic_scorer(normalized_query),
            self._regex_scorer(normalized_query),
            self._llm_scorer(normalized_query)
        ]
        
        # Ensemble the scores
        best_intent, final_confidence = self._ensemble_scores(scorers)
        
        # Apply confidence thresholds
        if final_confidence >= self.confidence_thresholds['high']:
            confidence_level = 'high'
        elif final_confidence >= self.confidence_thresholds['low']:
            confidence_level = 'low'
            # For low confidence, we might want to add a flag
            best_intent = AgentIntent.UNKNOWN
        else:
            confidence_level = 'unknown'
            best_intent = AgentIntent.UNKNOWN
        
        # Log for active learning
        self._log_recognition_result(query, best_intent, final_confidence, confidence_level, scorers)
        
        return create_intent_recognition(
            best_intent,
            final_confidence,
            raw_text=query
        )
    
    def _log_recognition_result(
        self, 
        query: str, 
        intent: AgentIntent, 
        confidence: float, 
        confidence_level: str,
        scorers: List[ScorerResult]
    ):
        """Log recognition results for active learning and debugging.
        
        Args:
            query: Original query
            intent: Recognized intent
            confidence: Final confidence score
            confidence_level: Confidence level (high/low/unknown)
            scorers: List of scorer results
        """
        # Log low confidence and unknown results for active learning
        if confidence_level in ['low', 'unknown']:
            self.logger.info(
                f"Low confidence recognition - Query: '{query}', "
                f"Intent: {intent}, Confidence: {confidence:.3f}, "
                f"Level: {confidence_level}"
            )
        
        # Log scorer details for debugging
        scorer_details = {}
        for scorer in scorers:
            top_scores = sorted(scorer.scores.items(), key=lambda x: x[1], reverse=True)[:3]
            scorer_details[scorer.scorer_type.value] = {
                'confidence': scorer.confidence,
                'top_scores': [(intent.value, score) for intent, score in top_scores]
            }
        
        self.logger.debug(
            f"Intent recognition details - Query: '{query}', "
            f"Final: {intent.value} ({confidence:.3f}), "
            f"Scorers: {scorer_details}"
        )
    
    def get_intent_examples(self, intent: AgentIntent) -> List[str]:
        """Get example queries for an intent.
        
        Args:
            intent: The intent to get examples for
            
        Returns:
            List of example queries
        """
        return [example.text for example in self.examples if example.intent == intent]
    
    def add_example(self, text: str, intent: AgentIntent):
        """Add a new example for active learning.
        
        Args:
            text: Example text
            intent: Associated intent
        """
        example = IntentExample(text=text, intent=intent)
        self.examples.append(example)
        self.logger.info(f"Added new example for intent {intent.value}: '{text}'")


# Backward compatibility - alias the new class
IntentRecognizer = HybridIntentRecognizer 
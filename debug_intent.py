#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.agent.intent_recognizer import HybridIntentRecognizer
from app.models.agent_models import AgentIntent

def debug_intent_recognition():
    recognizer = HybridIntentRecognizer()
    
    test_cases = [
        "List all models",
        "Show me available models", 
        "What models do you have?",
        "Which models are available?",
        "Show existing models"
    ]
    
    for query in test_cases:
        print(f"\n=== Testing: '{query}' ===")
        
        # Test semantic scorer
        semantic_result = recognizer._semantic_scorer(query)
        print(f"Semantic scores: {semantic_result.scores}")
        print(f"Semantic confidence: {semantic_result.confidence}")
        
        # Test regex scorer
        regex_result = recognizer._regex_scorer(query)
        print(f"Regex scores: {regex_result.scores}")
        print(f"Regex confidence: {regex_result.confidence}")
        
        # Test full recognition
        result = recognizer.recognize_intent(query)
        print(f"Final intent: {result.intent}")
        print(f"Final confidence: {result.confidence}")
        print(f"Normalized text: '{result.normalized_text}'")

if __name__ == "__main__":
    debug_intent_recognition() 
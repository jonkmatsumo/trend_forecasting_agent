"""
Agent Configuration
Configuration management for the agent interface.
"""

import os
from typing import Optional
from dataclasses import dataclass


@dataclass
class AgentConfig:
    """Configuration for the agent interface."""
    
    # Timeout settings
    query_timeout: int = 30  # seconds
    intent_recognition_timeout: int = 5  # seconds
    slot_extraction_timeout: int = 5  # seconds
    
    # Rate limiting
    max_queries_per_minute: int = 60
    max_queries_per_hour: int = 1000
    
    # Safety limits
    max_query_length: int = 1000  # characters
    max_keywords_per_query: int = 10
    max_context_size: int = 1024  # characters
    
    # Feature flags
    enable_intent_recognition: bool = True
    enable_slot_extraction: bool = True
    enable_advanced_nlp: bool = False
    
    # Logging
    log_queries: bool = True
    log_responses: bool = True
    log_errors: bool = True
    
    # Development settings
    debug_mode: bool = False
    mock_responses: bool = False
    
    def __post_init__(self):
        """Load configuration from environment variables."""
        self.query_timeout = int(os.getenv('AGENT_QUERY_TIMEOUT', self.query_timeout))
        self.intent_recognition_timeout = int(os.getenv('AGENT_INTENT_TIMEOUT', self.intent_recognition_timeout))
        self.slot_extraction_timeout = int(os.getenv('AGENT_SLOT_TIMEOUT', self.slot_extraction_timeout))
        
        self.max_queries_per_minute = int(os.getenv('AGENT_RATE_LIMIT_MINUTE', self.max_queries_per_minute))
        self.max_queries_per_hour = int(os.getenv('AGENT_RATE_LIMIT_HOUR', self.max_queries_per_hour))
        
        self.max_query_length = int(os.getenv('AGENT_MAX_QUERY_LENGTH', self.max_query_length))
        self.max_keywords_per_query = int(os.getenv('AGENT_MAX_KEYWORDS', self.max_keywords_per_query))
        self.max_context_size = int(os.getenv('AGENT_MAX_CONTEXT_SIZE', self.max_context_size))
        
        self.enable_intent_recognition = os.getenv('AGENT_ENABLE_INTENT', 'true').lower() == 'true'
        self.enable_slot_extraction = os.getenv('AGENT_ENABLE_SLOTS', 'true').lower() == 'true'
        self.enable_advanced_nlp = os.getenv('AGENT_ENABLE_ADVANCED_NLP', 'false').lower() == 'true'
        
        self.log_queries = os.getenv('AGENT_LOG_QUERIES', 'true').lower() == 'true'
        self.log_responses = os.getenv('AGENT_LOG_RESPONSES', 'true').lower() == 'true'
        self.log_errors = os.getenv('AGENT_LOG_ERRORS', 'true').lower() == 'true'
        
        self.debug_mode = os.getenv('AGENT_DEBUG_MODE', 'false').lower() == 'true'
        self.mock_responses = os.getenv('AGENT_MOCK_RESPONSES', 'false').lower() == 'true'


# Global agent configuration instance
agent_config = AgentConfig()


def get_agent_config() -> AgentConfig:
    """Get the global agent configuration instance."""
    return agent_config


def reload_agent_config():
    """Reload agent configuration from environment variables."""
    global agent_config
    agent_config = AgentConfig()
    return agent_config 
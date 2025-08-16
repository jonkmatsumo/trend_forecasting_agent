"""
Adapter Configuration Management
Handles feature flags and configuration for switching between adapters.
"""

import os
from typing import Optional
from enum import Enum


class AdapterType(str, Enum):
    """Supported adapter types."""
    IN_PROCESS = "in_process"
    HTTP = "http"


class AdapterConfig:
    """Configuration for forecaster service adapters."""
    
    def __init__(self):
        """Initialize adapter configuration from environment variables."""
        # Adapter type configuration
        self.adapter_type = self._get_adapter_type()
        
        # HTTP adapter configuration
        self.http_url = self._get_http_url()
        self.timeout = self._get_timeout()
        self.max_retries = self._get_max_retries()
        
        # Validate configuration
        self._validate_config()
    
    def _get_adapter_type(self) -> AdapterType:
        """Get adapter type from environment variable."""
        adapter_type_str = os.getenv("FORECASTER_ADAPTER_TYPE", "in_process").lower()
        
        try:
            return AdapterType(adapter_type_str)
        except ValueError:
            # Default to in_process if invalid value
            return AdapterType.IN_PROCESS
    
    def _get_http_url(self) -> Optional[str]:
        """Get HTTP URL from environment variable."""
        return os.getenv("FORECASTER_HTTP_URL")
    
    def _get_timeout(self) -> int:
        """Get timeout from environment variable."""
        try:
            return int(os.getenv("FORECASTER_TIMEOUT", "30"))
        except ValueError:
            return 30
    
    def _get_max_retries(self) -> int:
        """Get max retries from environment variable."""
        try:
            return int(os.getenv("FORECASTER_RETRY_ATTEMPTS", "3"))
        except ValueError:
            return 3
    
    def _validate_config(self):
        """Validate configuration settings."""
        if self.adapter_type == AdapterType.HTTP:
            if not self.http_url:
                raise ValueError(
                    "FORECASTER_HTTP_URL environment variable is required when using HTTP adapter"
                )
            
            if self.timeout <= 0:
                raise ValueError("FORECASTER_TIMEOUT must be positive")
            
            if self.max_retries < 0:
                raise ValueError("FORECASTER_RETRY_ATTEMPTS must be non-negative")
    
    def is_http_adapter(self) -> bool:
        """Check if HTTP adapter is configured."""
        return self.adapter_type == AdapterType.HTTP
    
    def is_in_process_adapter(self) -> bool:
        """Check if in-process adapter is configured."""
        return self.adapter_type == AdapterType.IN_PROCESS
    
    def get_http_config(self) -> dict:
        """Get HTTP adapter configuration."""
        if not self.is_http_adapter():
            raise ValueError("HTTP adapter not configured")
        
        return {
            "base_url": self.http_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "adapter_type": self.adapter_type.value,
            "http_url": self.http_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        return f"AdapterConfig({config_dict})"


# Global configuration instance
adapter_config = AdapterConfig()


def get_adapter_config() -> AdapterConfig:
    """Get the global adapter configuration instance."""
    return adapter_config


def create_adapter():
    """Create adapter instance based on configuration.
    
    Returns:
        Adapter instance (InProcessAdapter or HTTPAdapter)
        
    Raises:
        ValueError: If configuration is invalid
    """
    from app.services.adapters.in_process_adapter import InProcessAdapter
    from app.services.adapters.http_adapter import HTTPAdapter
    
    config = get_adapter_config()
    
    if config.is_in_process_adapter():
        return InProcessAdapter()
    elif config.is_http_adapter():
        http_config = config.get_http_config()
        return HTTPAdapter(**http_config)
    else:
        raise ValueError(f"Unsupported adapter type: {config.adapter_type}") 
"""
HTTP Client Configuration
Configuration settings for the HTTP forecaster client.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os


@dataclass
class HTTPClientConfig:
    """Configuration for HTTP client behavior."""
    
    # Connection settings
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    
    # Session settings
    pool_connections: int = 10
    pool_maxsize: int = 20
    keepalive_timeout: int = 30
    
    # Authentication
    api_key: Optional[str] = None
    auth_header: str = "Authorization"
    
    # Logging
    enable_request_logging: bool = True
    log_request_body: bool = False
    log_response_body: bool = False
    
    # Headers
    default_headers: Dict[str, str] = None
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.default_headers is None:
            self.default_headers = {
                "Content-Type": "application/json",
                "User-Agent": "TrendForecaster-HTTPClient/1.0"
            }
        
        # Add API key to headers if provided
        if self.api_key:
            self.default_headers[self.auth_header] = f"Bearer {self.api_key}"


def load_http_config() -> HTTPClientConfig:
    """Load HTTP client configuration from environment variables."""
    return HTTPClientConfig(
        base_url=os.environ.get("FORECASTER_API_URL", "http://localhost:5000"),
        timeout=int(os.environ.get("HTTP_TIMEOUT", "30")),
        max_retries=int(os.environ.get("HTTP_MAX_RETRIES", "3")),
        retry_delay=float(os.environ.get("HTTP_RETRY_DELAY", "1.0")),
        max_retry_delay=float(os.environ.get("HTTP_MAX_RETRY_DELAY", "60.0")),
        pool_connections=int(os.environ.get("HTTP_POOL_CONNECTIONS", "10")),
        pool_maxsize=int(os.environ.get("HTTP_POOL_MAXSIZE", "20")),
        api_key=os.environ.get("FORECASTER_API_KEY"),
        enable_request_logging=os.environ.get("HTTP_LOG_REQUESTS", "true").lower() == "true",
        log_request_body=os.environ.get("HTTP_LOG_REQUEST_BODY", "false").lower() == "true",
        log_response_body=os.environ.get("HTTP_LOG_RESPONSE_BODY", "false").lower() == "true"
    ) 
"""
Integration tests for Configuration System
Tests the integration of configuration with all system components.
"""

import pytest
from unittest.mock import Mock, patch

from app.config.config import Config


class TestConfigIntegration:
    """Test configuration integration with system components."""
    
    def test_configuration_integration(self):
        """Test that all configuration values are properly loaded."""
        config = Config()
        
        # Test core system config values are loaded
        assert hasattr(config, 'CIRCUIT_BREAKER_FAILURE_THRESHOLD')
        assert hasattr(config, 'RETRY_MAX_ATTEMPTS')
        assert hasattr(config, 'RATE_LIMITER_TOKENS_PER_SECOND')
        assert hasattr(config, 'MONITORING_ENABLED')
        assert hasattr(config, 'SECURITY_ENABLED')
        assert hasattr(config, 'COST_TRACKING_ENABLED')
    
    def test_circuit_breaker_configuration(self):
        """Test circuit breaker configuration values."""
        config = Config()
        
        # Test circuit breaker config values
        assert hasattr(config, 'CIRCUIT_BREAKER_FAILURE_THRESHOLD')
        assert hasattr(config, 'CIRCUIT_BREAKER_RECOVERY_TIMEOUT')
        assert hasattr(config, 'CIRCUIT_BREAKER_SUCCESS_THRESHOLD')
        
        # Test that values are reasonable
        assert config.CIRCUIT_BREAKER_FAILURE_THRESHOLD > 0
        assert config.CIRCUIT_BREAKER_RECOVERY_TIMEOUT > 0
        assert config.CIRCUIT_BREAKER_SUCCESS_THRESHOLD > 0
    
    def test_retry_configuration(self):
        """Test retry configuration values."""
        config = Config()
        
        # Test retry config values
        assert hasattr(config, 'RETRY_MAX_ATTEMPTS')
        assert hasattr(config, 'RETRY_BASE_DELAY')
        assert hasattr(config, 'RETRY_MAX_DELAY')
        assert hasattr(config, 'RETRY_EXPONENTIAL_BASE')
        assert hasattr(config, 'RETRY_JITTER')
        
        # Test that values are reasonable
        assert config.RETRY_MAX_ATTEMPTS > 0
        assert config.RETRY_BASE_DELAY > 0
        assert config.RETRY_MAX_DELAY > 0
        assert config.RETRY_EXPONENTIAL_BASE > 0
    
    def test_rate_limiter_configuration(self):
        """Test rate limiter configuration values."""
        config = Config()
        
        # Test rate limiter config values
        assert hasattr(config, 'RATE_LIMITER_TOKENS_PER_SECOND')
        assert hasattr(config, 'RATE_LIMITER_BUCKET_SIZE')
        assert hasattr(config, 'RATE_LIMITER_COST_PER_REQUEST')
        assert hasattr(config, 'RATE_LIMITER_BURST_SIZE')
        
        # Test that values are reasonable
        assert config.RATE_LIMITER_TOKENS_PER_SECOND > 0
        assert config.RATE_LIMITER_BUCKET_SIZE > 0
        assert config.RATE_LIMITER_COST_PER_REQUEST > 0
        assert config.RATE_LIMITER_BURST_SIZE > 0
    
    def test_monitoring_configuration(self):
        """Test monitoring configuration values."""
        config = Config()
        
        # Test monitoring config values
        assert hasattr(config, 'MONITORING_ENABLED')
        assert hasattr(config, 'MONITORING_METRICS_RETENTION_HOURS')
        assert hasattr(config, 'MONITORING_HEALTH_CHECK_INTERVAL')
        assert hasattr(config, 'MONITORING_EXPORT_FORMAT')
        
        # Test that values are reasonable
        assert isinstance(config.MONITORING_ENABLED, bool)
        assert config.MONITORING_METRICS_RETENTION_HOURS > 0
        assert config.MONITORING_HEALTH_CHECK_INTERVAL > 0
    
    def test_security_configuration(self):
        """Test security configuration values."""
        config = Config()
        
        # Test security config values
        assert hasattr(config, 'SECURITY_ENABLED')
        assert hasattr(config, 'SECURITY_AUDIT_LOG_FILE')
        assert hasattr(config, 'SECURITY_DATA_REDACTION_ENABLED')
        
        # Test that values are reasonable
        assert isinstance(config.SECURITY_ENABLED, bool)
        assert isinstance(config.SECURITY_DATA_REDACTION_ENABLED, bool)
    
    def test_cost_tracking_configuration(self):
        """Test cost tracking configuration values."""
        config = Config()
        
        # Test cost tracking config values
        assert hasattr(config, 'COST_TRACKING_ENABLED')
        assert hasattr(config, 'COST_DAILY_LIMIT')
        assert hasattr(config, 'COST_PER_TOKEN')
        assert hasattr(config, 'COST_ALERT_THRESHOLD')
        
        # Test that values are reasonable
        assert isinstance(config.COST_TRACKING_ENABLED, bool)
        assert config.COST_DAILY_LIMIT >= 0
        assert config.COST_PER_TOKEN >= 0
        assert config.COST_ALERT_THRESHOLD >= 0


if __name__ == "__main__":
    pytest.main([__file__]) 
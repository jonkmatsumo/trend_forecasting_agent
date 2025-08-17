"""
Security and Compliance Service
Provides enhanced logging policies, data redaction, and audit trail capabilities.
"""

import re
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log levels for security events."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: datetime
    event_type: str
    level: LogLevel
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    data_classification: DataClassification = DataClassification.INTERNAL


class DataRedactor:
    """Handles data redaction for sensitive information."""
    
    def __init__(self):
        """Initialize data redactor."""
        # Patterns for sensitive data
        self.patterns = {
            'api_key': r'[a-zA-Z0-9]{32,}',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'credit_card': r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'user_query': r'query["\']?\s*:\s*["\']([^"\']+)["\']',
            'model_response': r'response["\']?\s*:\s*["\']([^"\']+)["\']'
        }
        
        # Redaction replacement
        self.redaction_map = {
            'api_key': '[API_KEY_REDACTED]',
            'email': '[EMAIL_REDACTED]',
            'phone': '[PHONE_REDACTED]',
            'credit_card': '[CC_REDACTED]',
            'ssn': '[SSN_REDACTED]',
            'ip_address': '[IP_REDACTED]',
            'user_query': '[QUERY_REDACTED]',
            'model_response': '[RESPONSE_REDACTED]'
        }
    
    def redact_text(self, text: str, patterns_to_redact: Optional[List[str]] = None) -> str:
        """Redact sensitive information from text.
        
        Args:
            text: Text to redact
            patterns_to_redact: Specific patterns to redact, or None for all
            
        Returns:
            Redacted text
        """
        if patterns_to_redact is None:
            patterns_to_redact = list(self.patterns.keys())
        
        redacted_text = text
        
        for pattern_name in patterns_to_redact:
            if pattern_name in self.patterns:
                pattern = self.patterns[pattern_name]
                replacement = self.redaction_map[pattern_name]
                redacted_text = re.sub(pattern, replacement, redacted_text)
        
        return redacted_text
    
    def redact_dict(self, data: Dict[str, Any], 
                   patterns_to_redact: Optional[List[str]] = None) -> Dict[str, Any]:
        """Redact sensitive information from dictionary.
        
        Args:
            data: Dictionary to redact
            patterns_to_redact: Specific patterns to redact, or None for all
            
        Returns:
            Redacted dictionary
        """
        redacted_data = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                redacted_data[key] = self.redact_text(value, patterns_to_redact)
            elif isinstance(value, dict):
                redacted_data[key] = self.redact_dict(value, patterns_to_redact)
            elif isinstance(value, list):
                redacted_data[key] = [
                    self.redact_dict(item, patterns_to_redact) if isinstance(item, dict)
                    else self.redact_text(item, patterns_to_redact) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                redacted_data[key] = value
        
        return redacted_data
    
    def hash_sensitive_data(self, data: str, salt: Optional[str] = None) -> str:
        """Hash sensitive data for audit purposes.
        
        Args:
            data: Data to hash
            salt: Optional salt for hashing
            
        Returns:
            Hashed data
        """
        if salt:
            data_with_salt = f"{data}:{salt}"
        else:
            data_with_salt = data
        
        return hashlib.sha256(data_with_salt.encode()).hexdigest()


class AuditLogger:
    """Handles audit logging for compliance and security."""
    
    def __init__(self, log_file: Optional[str] = None):
        """Initialize audit logger.
        
        Args:
            log_file: Optional file path for audit logs
        """
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add file handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def log_access(self, user_id: str, resource: str, action: str, 
                   success: bool, details: Optional[Dict[str, Any]] = None):
        """Log access attempt.
        
        Args:
            user_id: User identifier
            resource: Resource being accessed
            action: Action performed
            success: Whether access was successful
            details: Additional details
        """
        event = {
            "event_type": "access_attempt",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "success": success,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        level = logging.INFO if success else logging.WARNING
        self.logger.log(level, f"Access attempt: {json.dumps(event)}")
    
    def log_data_access(self, user_id: str, data_type: str, 
                       classification: DataClassification,
                       access_method: str, details: Optional[Dict[str, Any]] = None):
        """Log data access.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            classification: Data classification level
            access_method: Method used to access data
            details: Additional details
        """
        event = {
            "event_type": "data_access",
            "user_id": user_id,
            "data_type": data_type,
            "classification": classification.value,
            "access_method": access_method,
            "timestamp": datetime.utcnow().isoformat(),
            "details": details or {}
        }
        
        self.logger.info(f"Data access: {json.dumps(event)}")
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event.
        
        Args:
            event: Security event to log
        """
        event_dict = {
            "event_type": event.event_type,
            "level": event.level.value,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "ip_address": event.ip_address,
            "timestamp": event.timestamp.isoformat(),
            "data_classification": event.data_classification.value,
            "details": event.details
        }
        
        log_level = getattr(logging, event.level.value.upper())
        self.logger.log(log_level, f"Security event: {json.dumps(event_dict)}")


class SecurityService:
    """Main security service that coordinates redaction and audit logging."""
    
    def __init__(self, audit_log_file: Optional[str] = None):
        """Initialize security service.
        
        Args:
            audit_log_file: Optional file path for audit logs
        """
        self.redactor = DataRedactor()
        self.audit_logger = AuditLogger(audit_log_file)
        self.logger = logging.getLogger("security_service")
        
        # Track sensitive operations
        self.sensitive_operations: Set[str] = {
            'llm_classification',
            'user_query_processing',
            'model_training',
            'data_export'
        }
    
    def process_log_data(self, data: Dict[str, Any], 
                        classification: DataClassification = DataClassification.INTERNAL,
                        redact_sensitive: bool = True) -> Dict[str, Any]:
        """Process log data with appropriate redaction and classification.
        
        Args:
            data: Data to process
            classification: Data classification level
            redact_sensitive: Whether to redact sensitive information
            
        Returns:
            Processed data
        """
        if redact_sensitive and classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
            data = self.redactor.redact_dict(data)
        
        # Add classification metadata
        data['_classification'] = classification.value
        data['_processed_at'] = datetime.utcnow().isoformat()
        
        return data
    
    def log_llm_request(self, user_id: Optional[str], query: str, response: str,
                       provider: str, model: str, success: bool, duration: float):
        """Log LLM request with appropriate security measures.
        
        Args:
            user_id: User identifier
            query: User query
            response: LLM response
            provider: LLM provider
            model: Model used
            success: Whether request was successful
            duration: Request duration
        """
        # Create security event
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="llm_request",
            level=LogLevel.INFO if success else LogLevel.ERROR,
            user_id=user_id,
            details={
                "provider": provider,
                "model": model,
                "duration": duration,
                "query_hash": self.redactor.hash_sensitive_data(query),
                "response_hash": self.redactor.hash_sensitive_data(response)
            },
            data_classification=DataClassification.CONFIDENTIAL
        )
        
        # Log the event
        self.audit_logger.log_security_event(event)
        
        # Log access attempt
        self.audit_logger.log_access(
            user_id=user_id or "anonymous",
            resource="llm_service",
            action="classification_request",
            success=success,
            details={
                "provider": provider,
                "model": model,
                "duration": duration
            }
        )
    
    def log_intent_classification(self, user_id: Optional[str], query: str,
                                intent: str, confidence: float, method: str):
        """Log intent classification with security measures.
        
        Args:
            user_id: User identifier
            query: User query
            intent: Classified intent
            confidence: Classification confidence
            method: Classification method used
        """
        # Create security event
        event = SecurityEvent(
            timestamp=datetime.utcnow(),
            event_type="intent_classification",
            level=LogLevel.INFO,
            user_id=user_id,
            details={
                "intent": intent,
                "confidence": confidence,
                "method": method,
                "query_hash": self.redactor.hash_sensitive_data(query)
            },
            data_classification=DataClassification.INTERNAL
        )
        
        # Log the event
        self.audit_logger.log_security_event(event)
    
    def log_data_access(self, user_id: str, data_type: str, 
                       access_method: str, details: Optional[Dict[str, Any]] = None):
        """Log data access for compliance.
        
        Args:
            user_id: User identifier
            data_type: Type of data accessed
            access_method: Method used to access data
            details: Additional details
        """
        classification = DataClassification.INTERNAL
        if data_type in ['user_queries', 'model_responses', 'training_data']:
            classification = DataClassification.CONFIDENTIAL
        
        self.audit_logger.log_data_access(
            user_id=user_id,
            data_type=data_type,
            classification=classification,
            access_method=access_method,
            details=details
        )
    
    def sanitize_for_logging(self, data: Any, 
                           classification: DataClassification = DataClassification.INTERNAL) -> Any:
        """Sanitize data for logging purposes.
        
        Args:
            data: Data to sanitize
            classification: Data classification level
            
        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            return self.process_log_data(data, classification)
        elif isinstance(data, str):
            if classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED]:
                return self.redactor.redact_text(data)
            return data
        else:
            return data
    
    def get_audit_summary(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get audit summary for specified time period.
        
        Args:
            start_time: Start time for summary
            end_time: End time for summary
            
        Returns:
            Audit summary
        """
        # In a real implementation, this would query the audit log database
        # For now, return a placeholder summary
        return {
            "period": {
                "start": start_time.isoformat() if start_time else None,
                "end": end_time.isoformat() if end_time else None
            },
            "total_events": 0,
            "events_by_type": {},
            "events_by_level": {},
            "data_access_summary": {}
        }


# Global security service instance
security_service = SecurityService() 
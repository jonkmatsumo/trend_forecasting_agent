"""
Agent State Model
Defines the state that flows through the LangGraph agent workflow.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from app.models.agent_models import AgentIntent
from app.services.agent.slot_extractor import ExtractedSlots


class AgentState(BaseModel):
    """State model for the LangGraph agent workflow."""
    
    # Input
    raw: str = Field(description="Raw user query")
    
    # Normalization
    norm_loose: Optional[str] = Field(default=None, description="Loose normalized text")
    norm_strict: Optional[str] = Field(default=None, description="Strict normalized text")
    
    # Intent Recognition
    intent: Optional[AgentIntent] = Field(default=None, description="Recognized intent")
    intent_conf: float = Field(default=0.0, description="Intent confidence score")
    
    # Slot Extraction
    slots: Union[Dict[str, Any], ExtractedSlots] = Field(default_factory=dict, description="Extracted slots")
    
    # Planning and Execution
    plan: List[Dict[str, Any]] = Field(default_factory=list, description="Execution plan")
    tool_outputs: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Tool execution outputs")
    
    # Output
    answer: Optional[Dict[str, Any]] = Field(default=None, description="Final formatted answer")
    
    # Metadata
    request_id: Optional[str] = Field(default=None, description="Request ID for tracking")
    user_id: Optional[str] = Field(default=None, description="User ID if available")
    session_id: Optional[str] = Field(default=None, description="Session ID if available")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        use_enum_values = True 
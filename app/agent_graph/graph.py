"""
LangGraph Graph
Main graph orchestration for the agent workflow.
"""

from typing import Callable
from langgraph.graph import StateGraph, END
from app.agent_graph.state import AgentState
from app.agent_graph.service_client import ForecasterClient
from app.agent_graph.nodes import normalize, recognize_intent, extract_slots, plan, step, format_answer


def should_continue(state: AgentState) -> str:
    """Determine if we should continue executing steps or format the answer.
    
    Args:
        state: Current agent state
        
    Returns:
        Next node name
    """
    if state.plan:
        return "step"
    else:
        return "format_answer"


def build_graph(service_client: ForecasterClient) -> StateGraph:
    """Build the LangGraph for agent orchestration.
    
    Args:
        service_client: Client for calling forecaster services
        
    Returns:
        Compiled LangGraph
    """
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("normalize", normalize)
    workflow.add_node("recognize_intent", recognize_intent)
    workflow.add_node("extract_slots", extract_slots)
    workflow.add_node("plan", plan)
    workflow.add_node("step", lambda state: step(state, service_client))
    workflow.add_node("format_answer", format_answer)
    
    # Set entry point
    workflow.set_entry_point("normalize")
    
    # Add edges
    workflow.add_edge("normalize", "recognize_intent")
    workflow.add_edge("recognize_intent", "extract_slots")
    workflow.add_edge("extract_slots", "plan")
    
    # Add conditional edge from plan to step or format_answer
    workflow.add_conditional_edges(
        "plan",
        should_continue,
        {
            "step": "step",
            "format_answer": "format_answer"
        }
    )
    
    # Add edge from step back to conditional check
    workflow.add_conditional_edges(
        "step",
        should_continue,
        {
            "step": "step",
            "format_answer": "format_answer"
        }
    )
    
    # Set end point
    workflow.add_edge("format_answer", END)
    
    # Compile the graph
    return workflow.compile()


def create_agent_graph(service_client: ForecasterClient) -> Callable:
    """Create a compiled agent graph.
    
    Args:
        service_client: Client for calling forecaster services
        
    Returns:
        Compiled graph function
    """
    graph = build_graph(service_client)
    return graph.invoke 
"""
Workflow Orchestration using LangGraph
──────────────────────────────────────
Creates and manages the agent workflow for document processing
"""

from langgraph.graph import StateGraph, END
from typing import Dict, Any

from agents.state import DocumentState, ProcessingStatus
from agents.preprocessing_agent import PreprocessingAgent
from agents.extraction_agent import ExtractionAgent
from agents.validation_agent import ValidationAgent
from agents.customer_lookup_agent import CustomerLookupAgent
from agents.action_matching_agent import ActionMatchingAgent
from agents.review_router_agent import ReviewRouterAgent
from agents.execution_agent import ExecutionAgent
from agents.logging_agent import LoggingAgent

def create_workflow():
    """Create the complete document processing workflow"""
    
    # Initialize agents
    preprocessing_agent = PreprocessingAgent()
    extraction_agent = ExtractionAgent()
    validation_agent = ValidationAgent()
    customer_lookup_agent = CustomerLookupAgent()
    action_matching_agent = ActionMatchingAgent()
    review_router_agent = ReviewRouterAgent()
    execution_agent = ExecutionAgent()
    logging_agent = LoggingAgent()
    
    # Create StateGraph
    workflow = StateGraph(DocumentState)
    
    # Add nodes (agents) to the workflow
    workflow.add_node("preprocessing", preprocessing_agent.process)
    workflow.add_node("extraction", extraction_agent.process)
    workflow.add_node("validation", validation_agent.process)
    workflow.add_node("customer_lookup", customer_lookup_agent.process)
    workflow.add_node("action_matching", action_matching_agent.process)
    workflow.add_node("review_router", review_router_agent.process)
    workflow.add_node("execution", execution_agent.process)
    workflow.add_node("logging", logging_agent.process)
    
    # Define the workflow edges (sequence)
    workflow.add_edge("preprocessing", "extraction")
    workflow.add_edge("extraction", "validation")
    workflow.add_edge("validation", "customer_lookup")
    workflow.add_edge("customer_lookup", "action_matching")
    workflow.add_edge("action_matching", "review_router")
    workflow.add_edge("review_router", "execution")
    """
    # Conditional edges from review_router
    workflow.add_conditional_edges(
        "review_router",
        lambda state: "execution" if state.status == ProcessingStatus.PROCESSING.value else "logging",
        {
            "execution": "execution",
            "logging": "logging"
        }
    )
    """
    workflow.add_edge("execution", "logging")
    workflow.add_edge("logging", END)
    
    # Set entry point
    workflow.set_entry_point("preprocessing")
    
    # Compile the workflow - REMOVE with_config here
    return workflow.compile()

# Simple workflow creation function
async def execute_workflow(initial_state: DocumentState) -> DocumentState:
    """Execute the workflow with the given initial state"""
    try:
        workflow = create_workflow()
        # Use ainvoke directly without with_config
        final_state = await workflow.ainvoke(initial_state)
        return final_state
    except Exception as exc:
        # Handle workflow execution errors
        initial_state.status = ProcessingStatus.FAILED.value
        initial_state.error_message = f"Workflow execution failed: {exc}"
        return initial_state

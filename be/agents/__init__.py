"""
Document Processing Agents Module
─────────────────────────────────

This module contains all agents used in the comprehensive LangGraph workflow
for document processing, extraction, validation, and execution.

Agent Pipeline:
1. PreprocessingAgent - PDF text extraction and OCR
2. ExtractionAgent - GPT-4o powered ID/Action extraction  
3. ValidationAgent - Data validation and normalization
4. CustomerLookupAgent - Database customer lookup
5. ActionMatchingAgent - Semantic action matching via ChromaDB
6. ReviewRouterAgent - Human-in-the-loop routing decisions
7. ExecutionAgent - Action execution with audit logging
8. LoggingAgent - Comprehensive audit and compliance logging

Each agent inherits from BaseAgent and operates on DocumentState,
providing a complete end-to-end processing pipeline with comprehensive
error handling, metrics collection, and audit capabilities.
"""

from .base_agent import BaseAgent
from .state import (
    DocumentState,
    ProcessingStatus,
    AgentStatus,
    PageText,
    IDActionPair,
    ValidatedPair,
    MatchedAction,
    ReviewItem,
    ExecutionResult,
    AgentExecutionInfo
)

# Import all concrete agent implementations
from .preprocessing_agent import PreprocessingAgent
from .extraction_agent import ExtractionAgent
from .validation_agent import ValidationAgent
from .customer_lookup_agent import CustomerLookupAgent
from .action_matching_agent import ActionMatchingAgent
from .review_router_agent import ReviewRouterAgent
from .execution_agent import ExecutionAgent
from .logging_agent import LoggingAgent

# Agent registry for easy access and workflow creation
AGENTS = {
    'preprocessing': PreprocessingAgent,
    'extraction': ExtractionAgent,
    'validation': ValidationAgent,
    'customer_lookup': CustomerLookupAgent,
    'action_matching': ActionMatchingAgent,
    'review_router': ReviewRouterAgent,
    'execution': ExecutionAgent,
    'logging': LoggingAgent
}

# Agent execution order for the standard workflow
AGENT_EXECUTION_ORDER = [
    'preprocessing',
    'extraction', 
    'validation',
    'customer_lookup',
    'action_matching',
    'review_router',
    'execution',
    'logging'
]

# Export all public classes and constants
__all__ = [
    # Base classes
    'BaseAgent',
    
    # State management
    'DocumentState',
    'ProcessingStatus',
    'AgentStatus',
    'PageText',
    'IDActionPair',
    'ValidatedPair',
    'MatchedAction',
    'ReviewItem',
    'ExecutionResult',
    'AgentExecutionInfo',
    
    # Agent implementations
    'PreprocessingAgent',
    'ExtractionAgent',
    'ValidationAgent',
    'CustomerLookupAgent',
    'ActionMatchingAgent',
    'ReviewRouterAgent',
    'ExecutionAgent',
    'LoggingAgent',
    
    # Utilities
    'AGENTS',
    'AGENT_EXECUTION_ORDER'
]

# Version information
__version__ = '1.0.0'
__author__ = 'Document Processing Team'
__description__ = 'Comprehensive agent-based document processing pipeline'

def get_agent_class(agent_name: str) -> BaseAgent:
    """
    Get agent class by name
    
    Args:
        agent_name: Name of the agent (from AGENTS registry)
        
    Returns:
        Agent class
        
    Raises:
        KeyError: If agent name not found
    """
    if agent_name not in AGENTS:
        raise KeyError(f"Agent '{agent_name}' not found. Available agents: {list(AGENTS.keys())}")
    
    return AGENTS[agent_name]

def create_agent_instance(agent_name: str) -> BaseAgent:
    """
    Create agent instance by name
    
    Args:
        agent_name: Name of the agent (from AGENTS registry)
        
    Returns:
        Initialized agent instance
        
    Raises:
        KeyError: If agent name not found
    """
    agent_class = get_agent_class(agent_name)
    return agent_class()

def get_workflow_agents() -> dict:
    """
    Get all agents needed for the standard workflow
    
    Returns:
        Dictionary of agent_name -> agent_instance
    """
    return {
        name: create_agent_instance(name) 
        for name in AGENT_EXECUTION_ORDER
    }

def validate_agent_workflow(agent_names: list) -> bool:
    """
    Validate that all required agents are present for a workflow
    
    Args:
        agent_names: List of agent names to validate
        
    Returns:
        True if all agents are valid, False otherwise
    """
    return all(name in AGENTS for name in agent_names)

# Module-level configuration
def configure_agents(**kwargs):
    """
    Configure module-level settings for all agents
    
    Args:
        **kwargs: Configuration parameters
    """
    # This could be used to set global agent configurations
    # For now, agents use individual configuration from settings
    pass

# Debugging and development utilities
def get_agent_info(agent_name: str = None) -> dict:
    """
    Get information about agents
    
    Args:
        agent_name: Specific agent name, or None for all agents
        
    Returns:
        Dictionary with agent information
    """
    if agent_name:
        if agent_name not in AGENTS:
            return {'error': f"Agent '{agent_name}' not found"}
        
        agent_class = AGENTS[agent_name]
        return {
            'name': agent_class.NAME,
            'description': agent_class.DESCRIPTION,
            'class': agent_class.__name__,
            'module': agent_class.__module__
        }
    else:
        return {
            name: {
                'name': cls.NAME,
                'description': cls.DESCRIPTION,
                'class': cls.__name__,
                'module': cls.__module__
            }
            for name, cls in AGENTS.items()
        }

def get_pipeline_info() -> dict:
    """
    Get information about the complete processing pipeline
    
    Returns:
        Dictionary with pipeline information
    """
    return {
        'total_agents': len(AGENTS),
        'execution_order': AGENT_EXECUTION_ORDER,
        'agents': get_agent_info(),
        'version': __version__,
        'description': __description__
    }

# Development and testing utilities
class AgentTestHarness:
    """
    Test harness for individual agent testing
    """
    
    def __init__(self, agent_name: str):
        """Initialize test harness for specific agent"""
        self.agent_name = agent_name
        self.agent = create_agent_instance(agent_name)
    
    async def test_agent(self, test_state: DocumentState) -> DocumentState:
        """
        Test agent with provided state
        
        Args:
            test_state: Test document state
            
        Returns:
            Updated state after agent processing
        """
        return await self.agent(test_state)
    
    def get_agent_info(self) -> dict:
        """Get information about the test agent"""
        return get_agent_info(self.agent_name)

# Workflow builder utility
class WorkflowBuilder:
    """
    Utility class for building custom agent workflows
    """
    
    def __init__(self):
        """Initialize workflow builder"""
        self.agents = []
        self.conditions = {}
    
    def add_agent(self, agent_name: str, condition: callable = None) -> 'WorkflowBuilder':
        """
        Add agent to workflow
        
        Args:
            agent_name: Name of agent to add
            condition: Optional condition function for conditional execution
            
        Returns:
            Self for method chaining
        """
        if agent_name not in AGENTS:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        self.agents.append(agent_name)
        if condition:
            self.conditions[agent_name] = condition
        
        return self
    
    def build(self) -> dict:
        """
        Build workflow configuration
        
        Returns:
            Workflow configuration dictionary
        """
        return {
            'agents': self.agents,
            'conditions': self.conditions,
            'agent_instances': {
                name: create_agent_instance(name) 
                for name in self.agents
            }
        }

# Monitoring utilities
def get_agent_metrics() -> dict:
    """
    Get metrics for all agents (placeholder for future implementation)
    
    Returns:
        Dictionary with agent metrics
    """
    return {
        'total_agents': len(AGENTS),
        'available_agents': list(AGENTS.keys()),
        'workflow_agents': AGENT_EXECUTION_ORDER,
        'status': 'initialized'
    }

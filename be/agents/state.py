"""
State management for the Document Processing LangGraph workflow
Defines all state classes and enums used throughout the agent pipeline
"""

from __future__ import annotations

import time
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Enums for State Management
# ─────────────────────────────────────────────────────────────────────────────

class ProcessingStatus(Enum):
    """Overall processing status for the document"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Status of individual agent execution"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ─────────────────────────────────────────────────────────────────────────────
# Data Classes for State Components
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageText:
    """Represents extracted text from a single PDF page"""
    page_num: int
    text: str
    source: str  # "text" or "ocr"
    confidence: float
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate page text data"""
        if self.confidence < 0.0 or self.confidence > 1.0:
            self.confidence = max(0.0, min(1.0, self.confidence))
        
        if not self.text:
            self.text = ""
        
        if self.source not in ["text", "ocr"]:
            self.source = "unknown"


@dataclass
class IDActionPair:
    """Represents an extracted National ID and Action pair"""
    national_id: str
    action: str
    confidence: float
    page_number: int
    context: str = ""
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    extraction_method: str = "llm"
    raw_text: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate ID/Action pair data"""
        self.national_id = self.national_id.strip()
        self.action = self.action.strip().lower()
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        if not self.context:
            self.context = self.raw_text


@dataclass
class ValidatedPair:
    """Represents a validated and normalized ID/Action pair"""
    national_id: str
    action: str
    confidence: float
    page_number: int
    context: str
    needs_review: bool = False
    validation_status: str = "validated"
    original_text: str = ""
    extraction_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Customer information (added by CustomerLookupAgent)
    customer_id: Optional[int] = None
    customer_found: bool = False
    customer_status: str = "unknown"
    customer_name: str = ""
    customer_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize validated pair data"""
        self.national_id = self.national_id.strip()
        self.action = self.action.strip().lower()
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class MatchedAction:
    """Represents an action matched to a standard action via semantic similarity"""
    national_id: str
    original_action: str
    matched_action: str
    similarity_score: float
    confidence: float
    page_number: int
    context: str
    needs_review: bool
    match_status: str  # "matched", "unknown_action", "low_confidence"
    match_method: str  # "semantic", "string_match", "exact", "none"
    
    # Customer information
    customer_id: Optional[int] = None
    customer_found: bool = False
    customer_status: str = "unknown"
    customer_name: str = ""
    
    # Matching metadata
    matching_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate matched action data"""
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class ReviewItem:
    """Represents an item that requires human review"""
    review_id: Optional[int]
    job_id: str
    national_id: str
    original_action: str
    matched_action: str
    confidence: float
    similarity_score: float
    page_number: int
    context: str
    customer_id: Optional[int]
    customer_name: str
    customer_status: str
    review_reasons: List[str]
    risk_level: str
    risk_factors: List[str]
    priority: str  # "low", "medium", "high", "critical"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Review tracking
    status: str = "pending"
    assigned_to: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    approved: Optional[bool] = None
    reviewer_comments: str = ""

    def __post_init__(self):
        """Validate review item data"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.similarity_score = max(0.0, min(1.0, self.similarity_score))
        
        if self.priority not in ["low", "medium", "high", "critical"]:
            self.priority = "medium"


@dataclass
class ExecutionResult:
    """Represents the result of executing an action"""
    national_id: str
    customer_id: Optional[int]
    action: str
    status: str  # From ExecutionStatus enum
    message: str = ""
    error_message: str = ""
    execution_time: float = 0.0
    timestamp: float = field(default_factory=time.time)
    result_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Rollback information
    rollback_data: Optional[Dict[str, Any]] = None
    can_rollback: bool = False

    def __post_init__(self):
        """Validate execution result data"""
        if self.execution_time < 0:
            self.execution_time = 0.0


@dataclass
class AgentExecutionInfo:
    """Tracks execution information for individual agents"""
    agent_name: str
    status: AgentStatus = AgentStatus.NOT_STARTED
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error_message: Optional[str] = None
    output_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def execution_time(self) -> Optional[float]:
        """Calculate execution time if both start and end times are available"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    @property
    def is_completed(self) -> bool:
        """Check if agent execution is completed (successfully or failed)"""
        return self.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]


# ─────────────────────────────────────────────────────────────────────────────
# Main Document State Class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentState:
    """
    Main state object that flows through the entire LangGraph workflow
    Contains all data and metadata for document processing
    """
    
    # Basic job information
    job_id: str
    file_path: str
    status: str = ProcessingStatus.PENDING.value
    progress: float = 0.0
    current_agent: Optional[str] = None
    error_message: Optional[str] = None
    
    # Processing timestamps
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    # Agent execution tracking
    agent_executions: Dict[str, AgentExecutionInfo] = field(default_factory=dict)
    
    # Stage-specific data (populated by each agent)
    pages_text: List[PageText] = field(default_factory=list)
    extracted_pairs: List[IDActionPair] = field(default_factory=list)
    validated_pairs: List[ValidatedPair] = field(default_factory=list)
    customer_mappings: Dict[str, Dict] = field(default_factory=dict)
    matched_actions: List[MatchedAction] = field(default_factory=list)
    review_required: List[ReviewItem] = field(default_factory=list)
    auto_approved: List[MatchedAction] = field(default_factory=list)
    execution_results: List[ExecutionResult] = field(default_factory=list)
    
    # Final results and statistics
    results: Dict[str, Any] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata and configuration
    metadata: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize state with default values and validation"""
        # Ensure progress is within valid range
        self.progress = max(0.0, min(1.0, self.progress))
        
        # Set started_at if status is processing and not already set
        if (self.status == ProcessingStatus.PROCESSING.value and 
            not self.started_at):
            self.started_at = time.time()
        
        # Set completed_at if status is completed/failed and not already set
        if (self.status in [ProcessingStatus.COMPLETED.value, ProcessingStatus.FAILED.value] and
            not self.completed_at):
            self.completed_at = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    # State Management Methods
    # ─────────────────────────────────────────────────────────────────────────

    def start_agent(self, agent_name: str) -> None:
        """Mark an agent as started"""
        if agent_name not in self.agent_executions:
            self.agent_executions[agent_name] = AgentExecutionInfo(agent_name=agent_name)
        
        self.agent_executions[agent_name].status = AgentStatus.RUNNING
        self.agent_executions[agent_name].start_time = time.time()
        self.current_agent = agent_name

    def complete_agent(self, agent_name: str, output_data: Optional[Dict[str, Any]] = None) -> None:
        """Mark an agent as completed successfully"""
        if agent_name in self.agent_executions:
            self.agent_executions[agent_name].status = AgentStatus.COMPLETED
            self.agent_executions[agent_name].end_time = time.time()
            if output_data:
                self.agent_executions[agent_name].output_data = output_data
        
        self.current_agent = None

    def fail_agent(self, agent_name: str, error_message: str) -> None:
        """Mark an agent as failed"""
        if agent_name in self.agent_executions:
            self.agent_executions[agent_name].status = AgentStatus.FAILED
            self.agent_executions[agent_name].end_time = time.time()
            self.agent_executions[agent_name].error_message = error_message
        
        self.current_agent = None
        self.status = ProcessingStatus.FAILED.value
        self.error_message = error_message

    def add_agent_metric(self, agent_name: str, metric_name: str, value: float) -> None:
        """Add a metric for an agent"""
        if agent_name not in self.agent_executions:
            self.agent_executions[agent_name] = AgentExecutionInfo(agent_name=agent_name)
        
        self.agent_executions[agent_name].metrics[metric_name] = value

    def update_progress(self, new_progress: float) -> None:
        """Update processing progress"""
        self.progress = max(self.progress, min(1.0, new_progress))

    def get_agent_status(self, agent_name: str) -> AgentStatus:
        """Get the status of a specific agent"""
        if agent_name in self.agent_executions:
            return self.agent_executions[agent_name].status
        return AgentStatus.NOT_STARTED

    def get_processing_duration(self) -> Optional[float]:
        """Get total processing duration if available"""
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None

    def get_agent_execution_times(self) -> Dict[str, Optional[float]]:
        """Get execution times for all agents"""
        return {
            name: info.execution_time 
            for name, info in self.agent_executions.items()
        }

    def calculate_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics"""
        stats = {
            'total_pages': len(self.pages_text),
            'extracted_pairs': len(self.extracted_pairs),
            'validated_pairs': len(self.validated_pairs),
            'matched_actions': len(self.matched_actions),
            'review_required': len(self.review_required),
            'auto_approved': len(self.auto_approved),
            'execution_results': len(self.execution_results),
            'processing_duration': self.get_processing_duration(),
            'agent_execution_times': self.get_agent_execution_times(),
            'successful_executions': len([r for r in self.execution_results if r.status == 'success']),
            'failed_executions': len([r for r in self.execution_results if r.status == 'failed']),
            'average_confidence': self._calculate_average_confidence(),
            'customers_found': len([p for p in self.validated_pairs if p.customer_found]),
            'customers_missing': len([p for p in self.validated_pairs if not p.customer_found]),
        }
        
        # Update internal processing stats
        self.processing_stats.update(stats)
        
        return stats

    def _calculate_average_confidence(self) -> float:
        """Calculate average confidence across all validated pairs"""
        if not self.validated_pairs:
            return 0.0
        
        total_confidence = sum(pair.confidence for pair in self.validated_pairs)
        return total_confidence / len(self.validated_pairs)

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            'job_id': self.job_id,
            'file_path': self.file_path,
            'status': self.status,
            'progress': self.progress,
            'current_agent': self.current_agent,
            'error_message': self.error_message,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'processing_duration': self.get_processing_duration(),
            'statistics': self.calculate_stats(),
            'results': self.results,
            'metadata': self.metadata
        }

    def is_completed(self) -> bool:
        """Check if processing is completed (successfully or failed)"""
        return self.status in [
            ProcessingStatus.COMPLETED.value,
            ProcessingStatus.FAILED.value,
            ProcessingStatus.CANCELLED.value
        ]

    def needs_review(self) -> bool:
        """Check if any items need human review"""
        return len(self.review_required) > 0

    def get_completion_summary(self) -> Dict[str, Any]:
        """Get a summary of completion status"""
        return {
            'job_id': self.job_id,
            'status': self.status,
            'progress': self.progress,
            'total_items_processed': len(self.matched_actions),
            'items_executed': len(self.execution_results),
            'items_requiring_review': len(self.review_required),
            'success_rate': (
                len([r for r in self.execution_results if r.status == 'success']) / 
                len(self.execution_results) if self.execution_results else 0
            ),
            'processing_time': self.get_processing_duration(),
            'error_message': self.error_message
        }

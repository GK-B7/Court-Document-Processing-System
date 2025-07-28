"""
ExecutionAgent
─────────────
Execute approved actions for only your 2 database actions:
- freeze_funds: Freeze all funds in the customer account
- release_funds: Release all held or restricted funds back to the customer account
- unknown_action: Handle unsupported actions

Process:
1. Takes matched_actions from ReviewRouterAgent (auto-approved items)
2. Executes only freeze_funds, release_funds, or handles unknown_action
3. Simulates action execution with proper logging and audit trail
4. Returns ExecutionResult objects with success/failure status
5. Handles only 3 cases based on your database schema
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
import json
from services.action_executor import action_executor, ActionResult
from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, MatchedAction, ExecutionResult
from database import ProcessingLogModel, db_manager
from psycopg2.extras import RealDictCursor
from exceptions import AgentError, DatabaseError
from metrics import (
    increment_database_query,
    record_database_query_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# Your supported actions only
SUPPORTED_ACTIONS = ['freeze_funds', 'release_funds']

# Execution configuration
EXECUTION_CONFIG = {
    'max_concurrent_executions': getattr(settings, 'MAX_CONCURRENT_EXECUTIONS', 5),
    'execution_timeout': getattr(settings, 'EXECUTION_TIMEOUT', 30),
    'enable_rollback': getattr(settings, 'ENABLE_ROLLBACK', True),
    'simulate_execution': getattr(settings, 'SIMULATE_EXECUTION', True),  # Set to False in production
    'audit_all_executions': getattr(settings, 'AUDIT_ALL_EXECUTIONS', True)
}

class ExecutionStatus(Enum):
    """Status of action execution"""
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMEOUT = "timeout"
    UNKNOWN_ACTION = "unknown_action"

class ExecutionAgent(BaseAgent):
    NAME = "execution_agent"
    DESCRIPTION = "Execute approved actions: freeze_funds, release_funds, or handle unknown_action"

    def __init__(self):
        """Initialize execution agent"""
        self.supported_actions = SUPPORTED_ACTIONS
        self.config = EXECUTION_CONFIG

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Execute all approved actions (auto_approved items)
        """
        start_time = time.perf_counter()
        
        # Get items to execute (auto_approved from ReviewRouterAgent)
        items_to_execute = getattr(state, 'auto_approved', [])
        
        if not items_to_execute:
            self.log_info("No approved actions to execute")
            state.execution_results = []
            state.progress = 0.95  # Almost complete
            return state

        self.log_info("Starting action execution", actions=len(items_to_execute))

        try:
            # Execute each approved action
            execution_results = []
            execution_stats = {
                'total_executed': 0,
                'successful': 0,
                'failed': 0,
                'freeze_funds_executed': 0,
                'release_funds_executed': 0,
                'unknown_actions': 0,
                'total_execution_time': 0.0
            }

            for action in items_to_execute:
                result = await self._execute_single_action(action, state.job_id)
                execution_results.append(result)
                
                # Update statistics
                execution_stats['total_executed'] += 1
                execution_stats['total_execution_time'] += result.execution_time
                
                if result.status == ExecutionStatus.SUCCESS.value:
                    execution_stats['successful'] += 1
                    
                    if result.action == 'freeze_funds':
                        execution_stats['freeze_funds_executed'] += 1
                    elif result.action == 'release_funds':
                        execution_stats['release_funds_executed'] += 1
                else:
                    execution_stats['failed'] += 1
                    
                    if result.action == 'unknown_action':
                        execution_stats['unknown_actions'] += 1

            # Log audit trail for all executions
            if self.config['audit_all_executions']:
                await self._create_execution_audit_log(state.job_id, execution_results, execution_stats)

        except Exception as exc:
            raise AgentError(
                f"Action execution failed: {exc}",
                agent_name=self.NAME,
                job_id=state.job_id,
                original_exception=exc
            ) from exc

        # Update state
        state.execution_results = execution_results
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.95  # 95% complete after execution

        # Log results
        duration = time.perf_counter() - start_time
        self._log_execution_results(execution_stats, duration)
        
        self.log_info(
            "Action execution completed",
            total_actions=execution_stats['total_executed'],
            successful=execution_stats['successful'],
            failed=execution_stats['failed'],
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    async def _execute_single_action(self, action: MatchedAction, job_id: str) -> ExecutionResult:
        """
        Execute a single action: freeze_funds, release_funds, or unknown_action
        """
        start_time = time.perf_counter()
        
        try:
            if action.matched_action == 'freeze_funds':
                result = await self._execute_freeze_funds(action)
            elif action.matched_action == 'release_funds':
                result = await self._execute_release_funds(action)
            elif action.matched_action == 'unknown_action':
                result = await self._execute_unknown_action(action)
            else:
                # Any other action is unsupported
                result = ExecutionResult(
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    action=action.matched_action,
                    status=ExecutionStatus.FAILED.value,
                    message="Unsupported action",
                    error_message=f"Action '{action.matched_action}' is not supported. Only freeze_funds and release_funds are supported.",
                    execution_time=0.0,
                    timestamp=time.time(),
                    metadata={'unsupported_action': True, 'supported_actions': self.supported_actions}
                )
            
            result.execution_time = time.perf_counter() - start_time
            return result
            
        except Exception as exc:
            execution_time = time.perf_counter() - start_time
            
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action=action.matched_action,
                status=ExecutionStatus.FAILED.value,
                message="Execution failed due to error",
                error_message=f"Execution error: {exc}",
                execution_time=execution_time,
                timestamp=time.time(),
                metadata={'exception_type': type(exc).__name__, 'original_error': str(exc)}
            )

    # ------------------------------------------------------------------
    async def _execute_freeze_funds(self, action: MatchedAction) -> ExecutionResult:
        """
        Execute freeze_funds action using dummy function
        """
        try:
            # Use the dummy function
            result = await action_executor.execute_action(action.national_id, 'freeze_funds')
            
            if result['status'] == ActionResult.SUCCESS.value:
                return ExecutionResult(
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    action='freeze_funds',
                    status=ExecutionStatus.SUCCESS.value,
                    message=result['message'],
                    execution_time=0.0,  # Will be set by caller
                    timestamp=time.time(),
                    result_data=result['details'],
                    rollback_data=result.get('rollback_data'),
                    can_rollback=result['details'].get('can_be_reversed', False)
                )
            else:
                return ExecutionResult(
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    action='freeze_funds',
                    status=ExecutionStatus.FAILED.value,
                    message=result['message'],
                    error_message=result.get('error_details', result['message']),
                    execution_time=0.0,
                    timestamp=time.time(),
                    metadata=result.get('metadata', {})
                )
                
        except Exception as exc:
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='freeze_funds',
                status=ExecutionStatus.FAILED.value,
                message="Freeze funds execution failed",
                error_message=f"Error freezing funds: {exc}",
                execution_time=0.0,
                timestamp=time.time(),
                metadata={'error_type': type(exc).__name__}
            )

    # ------------------------------------------------------------------
    async def _execute_release_funds(self, action: MatchedAction) -> ExecutionResult:
        """
        Execute release_funds action using dummy function
        """
        try:
            # Use the dummy function
            result = await action_executor.execute_action(action.national_id, 'release_funds')
            
            if result['status'] == ActionResult.SUCCESS.value:
                return ExecutionResult(
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    action='release_funds',
                    status=ExecutionStatus.SUCCESS.value,
                    message=result['message'],
                    execution_time=0.0,  # Will be set by caller
                    timestamp=time.time(),
                    result_data=result['details'],
                    rollback_data=result.get('rollback_data'),
                    can_rollback=result['details'].get('can_be_reversed', False)
                )
            else:
                return ExecutionResult(
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    action='release_funds',
                    status=ExecutionStatus.FAILED.value,
                    message=result['message'],
                    error_message=result.get('error_details', result['message']),
                    execution_time=0.0,
                    timestamp=time.time(),
                    metadata=result.get('metadata', {})
                )
                
        except Exception as exc:
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='release_funds',
                status=ExecutionStatus.FAILED.value,
                message="Release funds execution failed",
                error_message=f"Error releasing funds: {exc}",
                execution_time=0.0,
                timestamp=time.time(),
                metadata={'error_type': type(exc).__name__}
            )


    # ------------------------------------------------------------------
    async def _execute_unknown_action(self, action: MatchedAction) -> ExecutionResult:
        """
        Handle unknown actions using the action executor's validation
        """
        # Use the action executor to handle unsupported actions
        result = await action_executor.execute_action(action.national_id, action.matched_action)
        
        return ExecutionResult(
            national_id=action.national_id,
            customer_id=action.customer_id,
            action=action.matched_action,
            status=ExecutionStatus.UNKNOWN_ACTION.value,
            message=result['message'],
            execution_time=0.0,
            timestamp=time.time(),
            result_data={
                'original_action': action.original_action,
                'supported_actions': result.get('supported_actions', ['freeze_funds', 'release_funds']),
                'error_type': result.get('error_type', 'unsupported_action')
            },
            metadata={
                'requires_manual_review': True,
                'original_context': action.context,
                'page_number': action.page_number
            }
        )

    # ------------------------------------------------------------------
    async def _execute_real_freeze_funds(self, action: MatchedAction) -> ExecutionResult:
        """
        Real implementation for freezing funds (production mode)
        """
        try:
            # In production, this would:
            # 1. Connect to banking system API
            # 2. Update customer account status to frozen
            # 3. Log the action in audit system
            # 4. Send notifications if required
            
            # For now, return a placeholder for real implementation
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='freeze_funds',
                status=ExecutionStatus.SUCCESS.value,
                message=f"Production freeze funds executed for customer {action.customer_id}",
                execution_time=0.0,
                timestamp=time.time(),
                result_data={
                    'action_performed': 'freeze_funds',
                    'customer_id': action.customer_id,
                    'production_mode': True
                },
                metadata={'production_execution': True}
            )
            
        except Exception as exc:
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='freeze_funds',
                status=ExecutionStatus.FAILED.value,
                message="Production freeze funds failed",
                error_message=f"Production error: {exc}",
                execution_time=0.0,
                timestamp=time.time(),
                metadata={'production_error': True}
            )

    # ------------------------------------------------------------------
    async def _execute_real_release_funds(self, action: MatchedAction) -> ExecutionResult:
        """
        Real implementation for releasing funds (production mode)
        """
        try:
            # In production, this would:
            # 1. Connect to banking system API
            # 2. Update customer account status to active
            # 3. Release held funds
            # 4. Log the action in audit system
            # 5. Send notifications if required
            
            # For now, return a placeholder for real implementation
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='release_funds',
                status=ExecutionStatus.SUCCESS.value,
                message=f"Production release funds executed for customer {action.customer_id}",
                execution_time=0.0,
                timestamp=time.time(),
                result_data={
                    'action_performed': 'release_funds',
                    'customer_id': action.customer_id,
                    'production_mode': True
                },
                metadata={'production_execution': True}
            )
            
        except Exception as exc:
            return ExecutionResult(
                national_id=action.national_id,
                customer_id=action.customer_id,
                action='release_funds',
                status=ExecutionStatus.FAILED.value,
                message="Production release funds failed",
                error_message=f"Production error: {exc}",
                execution_time=0.0,
                timestamp=time.time(),
                metadata={'production_error': True}
            )

    # ------------------------------------------------------------------
    async def _create_execution_audit_log(
        self, 
        job_id: str, 
        execution_results: List[ExecutionResult], 
        stats: Dict[str, Any]
    ) -> None:
        """
        Create comprehensive audit log for all executions
        """
        try:
            increment_database_query()
            
            audit_data = {
                'job_id': job_id,
                'execution_summary': {
                    'total_actions': stats['total_executed'],
                    'successful_actions': stats['successful'],
                    'failed_actions': stats['failed'],
                    'freeze_funds_executed': stats['freeze_funds_executed'],
                    'release_funds_executed': stats['release_funds_executed'],
                    'unknown_actions': stats['unknown_actions']
                },
                'execution_details': [
                    {
                        'national_id': result.national_id,
                        'customer_id': result.customer_id,
                        'action': result.action,
                        'status': result.status,
                        'message': result.message,
                        'execution_time': result.execution_time,
                        'timestamp': result.timestamp
                    }
                    for result in execution_results
                ],
                'total_execution_time': stats['total_execution_time'],
                'audit_timestamp': time.time(),
                'agent_name': self.NAME
            }

            ProcessingLogModel.create_log_entry(
                job_id=job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'execution_audit',
                    'audit_data': audit_data
                }
            )

            self.log_debug("Execution audit log created", job_id=job_id)

        except Exception as exc:
            self.log_error(f"Failed to create execution audit log: {exc}")

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    async def _simulate_delay(self, min_seconds: float, max_seconds: float) -> None:
        """Simulate processing delay for realistic testing"""
        import random
        import asyncio
        
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)

    def _should_simulate_success(self, success_rate: float) -> bool:
        """Determine if simulation should succeed based on success rate"""
        import random
        return random.random() < success_rate

    def _log_execution_results(self, stats: Dict[str, Any], duration: float) -> None:
        """Log detailed execution results"""
        total = stats['total_executed']
        
        if total == 0:
            return

        success_rate = (stats['successful'] / total) * 100
        failure_rate = (stats['failed'] / total) * 100
        
        self.log_info(
            "Execution Statistics",
            total_executed=total,
            successful=f"{stats['successful']} ({success_rate:.1f}%)",
            failed=f"{stats['failed']} ({failure_rate:.1f}%)",
            freeze_funds=stats['freeze_funds_executed'],
            release_funds=stats['release_funds_executed'],
            unknown_actions=stats['unknown_actions'],
            total_execution_time=f"{stats['total_execution_time']:.2f}s",
            average_time_per_action=f"{stats['total_execution_time']/total:.2f}s" if total > 0 else "0s",
            processing_duration=f"{duration:.2f}s"
        )

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution agent statistics"""
        return {
            'supported_actions': self.supported_actions,
            'simulation_mode': self.config['simulate_execution'],
            'max_concurrent_executions': self.config['max_concurrent_executions'],
            'execution_timeout': self.config['execution_timeout'],
            'audit_enabled': self.config['audit_all_executions'],
            'agent_name': self.NAME
        }

# Export the agent class
__all__ = ['ExecutionAgent', 'ExecutionStatus']

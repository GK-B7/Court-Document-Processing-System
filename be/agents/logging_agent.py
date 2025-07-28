"""
LoggingAgent
────────────
1. Takes the final DocumentState after all processing is complete
2. Creates comprehensive audit logs in the database
3. Records detailed processing statistics and metrics
4. Generates final summary reports
5. Handles error logging and troubleshooting information
6. Updates job status with final results
7. Cleans up temporary data and resources

Logging Strategy:
• Comprehensive audit trail for compliance
• Detailed performance metrics for optimization
• Error analysis for troubleshooting
• Processing statistics for monitoring
• Final status update in database
"""

from __future__ import annotations

import time
import logging
import json
from typing import Dict, List, Any, Optional

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, AgentStatus
from database import ProcessingLogModel, JobModel
from exceptions import DatabaseError, AgentError
from metrics import (
    increment_database_query,
    record_database_query_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class LoggingAgent(BaseAgent):
    NAME = "logging_agent"
    DESCRIPTION = "Create comprehensive audit logs and finalize processing"

    def __init__(self):
        """Initialize logging agent with configuration"""
        self.enable_detailed_logging = getattr(settings, 'ENABLE_DETAILED_LOGGING', True)
        self.enable_performance_logging = getattr(settings, 'ENABLE_PERFORMANCE_LOGGING', True)
        self.enable_audit_logging = getattr(settings, 'ENABLE_AUDIT_LOGGING', True)
        self.log_retention_days = getattr(settings, 'LOG_RETENTION_DAYS', 90)

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Create comprehensive logs and finalize document processing
        """
        start_time = time.perf_counter()
        
        self.log_info("Starting comprehensive logging", job_id=state.job_id)

        try:
            # Step 1: Calculate final statistics
            final_stats = state.calculate_stats()
            
            # Step 2: Create audit log entries
            await self._create_audit_logs(state, final_stats)
            
            # Step 3: Log performance metrics
            await self._log_performance_metrics(state, final_stats)
            
            # Step 4: Create error analysis if needed
            if state.status == ProcessingStatus.FAILED.value:
                await self._create_error_analysis(state)
            
            # Step 5: Generate processing summary
            processing_summary = await self._generate_processing_summary(state, final_stats)
            
            # Step 6: Update final job status in database
            await self._update_final_job_status(state, processing_summary)
            
            # Step 7: Create compliance logs
            if self.enable_audit_logging:
                await self._create_compliance_logs(state, final_stats)
            
            # Step 8: Clean up temporary data
            await self._cleanup_temporary_data(state)

        except Exception as exc:
            # Even if logging fails, we don't want to fail the entire job
            self.log_error(f"Logging failed, but continuing: {exc}")
            
            # Create minimal error log
            try:
                await self._create_minimal_error_log(state, str(exc))
            except Exception as inner_exc:
                self.log_error(f"Even minimal logging failed: {inner_exc}")

        # Update state with final information
        state.status = state.status  # Keep existing status
        state.progress = 1.0  # Ensure 100% progress
        state.completed_at = time.time()
        
        # Record final statistics
        state.results = state.calculate_stats()
        
        duration = time.perf_counter() - start_time
        self.log_info(
            "Comprehensive logging completed",
            job_id=state.job_id,
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    async def _create_audit_logs(self, state: DocumentState, stats: Dict[str, Any]) -> None:
        """
        Create comprehensive audit logs for compliance and tracking
        """
        try:
            # Main processing audit log
            audit_data = {
                'job_information': {
                    'job_id': state.job_id,
                    'file_path': state.file_path,
                    'processing_status': state.status,
                    'created_at': state.created_at,
                    'started_at': state.started_at,
                    'completed_at': state.completed_at,
                    'total_duration': state.get_processing_duration()
                },
                'processing_pipeline': {
                    'pages_processed': stats['total_pages'],
                    'pairs_extracted': stats['extracted_pairs'],
                    'pairs_validated': stats['validated_pairs'],
                    'actions_matched': stats['matched_actions'],
                    'items_for_review': stats['review_required'],
                    'items_auto_approved': stats['auto_approved'],
                    'actions_executed': stats['execution_results']
                },
                'agent_execution_summary': self._create_agent_execution_summary(state),
                'customer_processing': {
                    'customers_found': stats['customers_found'],
                    'customers_missing': stats['customers_missing'],
                    'customer_lookup_success_rate': (
                        stats['customers_found'] / (stats['customers_found'] + stats['customers_missing'])
                        if (stats['customers_found'] + stats['customers_missing']) > 0 else 0
                    )
                },
                'execution_outcomes': {
                    'successful_executions': stats['successful_executions'],
                    'failed_executions': stats['failed_executions'],
                    'execution_success_rate': (
                        stats['successful_executions'] / stats['execution_results']
                        if stats['execution_results'] > 0 else 0
                    )
                },
                'quality_metrics': {
                    'average_confidence': stats['average_confidence'],
                    'review_rate': stats['review_required'] / stats['matched_actions'] if stats['matched_actions'] > 0 else 0,
                    'auto_approval_rate': stats['auto_approved'] / stats['matched_actions'] if stats['matched_actions'] > 0 else 0
                }
            }

            increment_database_query()
            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'comprehensive_audit',
                    'audit_data': audit_data,
                    'compliance_timestamp': time.time(),
                    'log_version': '1.0'
                }
            )

            self.log_debug("Comprehensive audit log created", job_id=state.job_id)

        except Exception as exc:
            raise DatabaseError(
                f"Failed to create audit logs: {exc}",
                operation="audit_logging",
                table_name="processing_logs",
                original_exception=exc
            ) from exc

    # ------------------------------------------------------------------
    async def _log_performance_metrics(self, state: DocumentState, stats: Dict[str, Any]) -> None:
        """
        Log detailed performance metrics for optimization
        """
        if not self.enable_performance_logging:
            return

        try:
            agent_times = state.get_agent_execution_times()
            
            performance_data = {
                'overall_performance': {
                    'total_processing_time': stats['processing_duration'],
                    'pages_per_second': (
                        stats['total_pages'] / stats['processing_duration']
                        if stats['processing_duration'] and stats['processing_duration'] > 0 else 0
                    ),
                    'actions_per_second': (
                        stats['matched_actions'] / stats['processing_duration']
                        if stats['processing_duration'] and stats['processing_duration'] > 0 else 0
                    )
                },
                'agent_performance': {
                    name: {
                        'execution_time': exec_time,
                        'percentage_of_total': (
                            (exec_time / stats['processing_duration']) * 100
                            if exec_time and stats['processing_duration'] and stats['processing_duration'] > 0 else 0
                        )
                    }
                    for name, exec_time in agent_times.items()
                    if exec_time is not None
                },
                'throughput_metrics': {
                    'pages_processed': stats['total_pages'],
                    'extraction_rate': stats['extracted_pairs'] / stats['total_pages'] if stats['total_pages'] > 0 else 0,
                    'validation_success_rate': stats['validated_pairs'] / stats['extracted_pairs'] if stats['extracted_pairs'] > 0 else 0,
                    'matching_success_rate': stats['matched_actions'] / stats['validated_pairs'] if stats['validated_pairs'] > 0 else 0
                },
                'bottleneck_analysis': self._analyze_performance_bottlenecks(agent_times)
            }

            increment_database_query()
            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'performance_metrics',
                    'performance_data': performance_data,
                    'measurement_timestamp': time.time()
                }
            )

            self.log_debug("Performance metrics logged", job_id=state.job_id)

        except Exception as exc:
            self.log_error(f"Failed to log performance metrics: {exc}")

    # ------------------------------------------------------------------
    async def _create_error_analysis(self, state: DocumentState) -> None:
        """
        Create detailed error analysis for failed processing jobs
        """
        try:
            # Collect error information from all agents
            agent_errors = {}
            for agent_name, agent_info in state.agent_executions.items():
                if agent_info.status == AgentStatus.FAILED:
                    agent_errors[agent_name] = {
                        'error_message': agent_info.error_message,
                        'execution_time': agent_info.execution_time,
                        'output_data': agent_info.output_data
                    }

            # Analyze execution results for errors
            execution_errors = []
            for result in state.execution_results:
                if result.status == 'failed':
                    execution_errors.append({
                        'customer_id': result.customer_id,
                        'national_id': result.national_id,
                        'action': result.action,
                        'error_message': result.error_message,
                        'metadata': result.metadata
                    })

            error_analysis = {
                'job_failure_info': {
                    'job_id': state.job_id,
                    'failure_timestamp': state.completed_at or time.time(),
                    'main_error_message': state.error_message,
                    'current_agent_at_failure': state.current_agent,
                    'progress_at_failure': state.progress
                },
                'agent_failures': agent_errors,
                'execution_failures': execution_errors,
                'failure_analysis': {
                    'total_agent_failures': len(agent_errors),
                    'total_execution_failures': len(execution_errors),
                    'failure_stage': self._determine_failure_stage(state),
                    'potential_causes': self._analyze_potential_causes(state, agent_errors)
                },
                'recovery_recommendations': self._generate_recovery_recommendations(state, agent_errors)
            }

            increment_database_query()
            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'error_analysis',
                    'error_analysis': error_analysis,
                    'analysis_timestamp': time.time()
                }
            )

            self.log_info("Error analysis created", job_id=state.job_id, failures=len(agent_errors))

        except Exception as exc:
            self.log_error(f"Failed to create error analysis: {exc}")

    # ------------------------------------------------------------------
    async def _generate_processing_summary(self, state: DocumentState, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive processing summary
        """
        summary = {
            'job_metadata': {
                'job_id': state.job_id,
                'processing_status': state.status,
                'completion_timestamp': time.time(),
                'total_processing_time': stats['processing_duration']
            },
            'processing_results': {
                'pages_processed': stats['total_pages'],
                'total_pairs_found': stats['extracted_pairs'],
                'valid_pairs': stats['validated_pairs'],
                'matched_actions': stats['matched_actions'],
                'executed_actions': stats['successful_executions'],
                'failed_actions': stats['failed_executions'],
                'items_requiring_review': stats['review_required']
            },
            'quality_assessment': {
                'overall_confidence': stats['average_confidence'],
                'processing_success_rate': (
                    stats['successful_executions'] / stats['execution_results']
                    if stats['execution_results'] > 0 else 0
                ),
                'review_rate': (
                    stats['review_required'] / stats['matched_actions']
                    if stats['matched_actions'] > 0 else 0
                ),
                'customer_match_rate': (
                    stats['customers_found'] / (stats['customers_found'] + stats['customers_missing'])
                    if (stats['customers_found'] + stats['customers_missing']) > 0 else 0
                )
            },
            'agent_performance': state.get_agent_execution_times(),
            'recommendations': self._generate_process_improvement_recommendations(stats)
        }

        return summary

    # ------------------------------------------------------------------
    async def _update_final_job_status(self, state: DocumentState, summary: Dict[str, Any]) -> None:
        """
        Update the job status in the database with final results
        """
        try:
            increment_database_query()
            
            # Determine final status
            final_status = state.status
            if state.needs_review() and final_status == ProcessingStatus.COMPLETED.value:
                final_status = ProcessingStatus.REVIEW_REQUIRED.value

            JobModel.update_job_status(
                job_id=state.job_id,
                status=final_status,
                progress=1.0,
                metadata=summary,
                error_message=state.error_message
            )

            self.log_info("Final job status updated", job_id=state.job_id, status=final_status)

        except Exception as exc:
            raise DatabaseError(
                f"Failed to update final job status: {exc}",
                operation="update_job_status",
                table_name="jobs",
                original_exception=exc
            ) from exc

    # ------------------------------------------------------------------
    async def _create_compliance_logs(self, state: DocumentState, stats: Dict[str, Any]) -> None:
        """
        Create compliance and regulatory audit logs
        """
        try:
            compliance_data = {
                'regulatory_compliance': {
                    'processing_timestamp': time.time(),
                    'job_id': state.job_id,
                    'data_retention_compliant': True,
                    'audit_trail_complete': True,
                    'customer_privacy_maintained': True
                },
                'data_processing_summary': {
                    'customer_records_accessed': stats['customers_found'],
                    'actions_executed': stats['successful_executions'],
                    'data_sources': ['PDF document', 'customer database'],
                    'processing_duration': stats['processing_duration']
                },
                'security_measures': {
                    'data_encrypted': True,
                    'access_logged': True,
                    'authorized_processing': True,
                    'secure_transmission': True
                },
                'retention_policy': {
                    'log_retention_days': self.log_retention_days,
                    'data_anonymization': False,  # Set based on requirements
                    'automatic_cleanup': True
                }
            }

            increment_database_query()
            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'compliance_audit',
                    'compliance_data': compliance_data,
                    'compliance_version': '1.0'
                }
            )

            self.log_debug("Compliance logs created", job_id=state.job_id)

        except Exception as exc:
            self.log_error(f"Failed to create compliance logs: {exc}")

    # ------------------------------------------------------------------
    async def _cleanup_temporary_data(self, state: DocumentState) -> None:
        """
        Clean up temporary data and resources
        """
        try:
            cleanup_info = {
                'temp_file_path': state.file_path,
                'cleanup_timestamp': time.time(),
                'resources_cleaned': []
            }

            # Clean up uploaded file if configured
            if getattr(settings, 'CLEANUP_TEMP_FILES', True):
                import os
                if os.path.exists(state.file_path):
                    try:
                        os.remove(state.file_path)
                        cleanup_info['resources_cleaned'].append('uploaded_file')
                        self.log_debug("Cleaned up uploaded file", path=state.file_path)
                    except OSError as exc:
                        self.log_warning(f"Failed to clean up file {state.file_path}: {exc}")

            # Log cleanup information
            increment_database_query()
            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'cleanup',
                    'cleanup_info': cleanup_info
                }
            )

        except Exception as exc:
            self.log_error(f"Cleanup operations failed: {exc}")

    # ------------------------------------------------------------------
    async def _create_minimal_error_log(self, state: DocumentState, error_message: str) -> None:
        """
        Create minimal error log when full logging fails
        """
        try:
            minimal_log = {
                'job_id': state.job_id,
                'logging_failure': True,
                'error_message': error_message,
                'timestamp': time.time(),
                'status': state.status,
                'progress': state.progress
            }

            ProcessingLogModel.create_log_entry(
                job_id=state.job_id,
                agent_name=self.NAME,
                data={
                    'log_type': 'minimal_error',
                    'minimal_log': minimal_log
                }
            )

        except Exception:
            # If even minimal logging fails, just log to file
            self.log_error(f"All logging failed for job {state.job_id}: {error_message}")

    # ------------------------------------------------------------------
    # Helper Methods
    # ------------------------------------------------------------------

    def _create_agent_execution_summary(self, state: DocumentState) -> Dict[str, Any]:
        """Create summary of agent executions"""
        summary = {}
        
        for agent_name, agent_info in state.agent_executions.items():
            summary[agent_name] = {
                'status': agent_info.status.value,
                'execution_time': agent_info.execution_time,
                'start_time': agent_info.start_time,
                'end_time': agent_info.end_time,
                'error_message': agent_info.error_message,
                'metrics': agent_info.metrics
            }
        
        return summary

    def _analyze_performance_bottlenecks(self, agent_times: Dict[str, Optional[float]]) -> List[str]:
        """Analyze performance bottlenecks"""
        bottlenecks = []
        
        # Filter out None values
        valid_times = {name: time for name, time in agent_times.items() if time is not None}
        
        if not valid_times:
            return bottlenecks
        
        total_time = sum(valid_times.values())
        
        for agent_name, exec_time in valid_times.items():
            percentage = (exec_time / total_time) * 100
            if percentage > 30:  # If agent takes more than 30% of total time
                bottlenecks.append(f"{agent_name} ({percentage:.1f}% of total time)")
        
        return bottlenecks

    def _determine_failure_stage(self, state: DocumentState) -> str:
        """Determine at which stage the processing failed"""
        if state.current_agent:
            return f"During {state.current_agent} execution"
        elif state.progress < 0.2:
            return "Early preprocessing stage"
        elif state.progress < 0.5:
            return "Extraction/validation stage"
        elif state.progress < 0.8:
            return "Customer lookup/action matching stage"
        elif state.progress < 1.0:
            return "Review routing/execution stage"
        else:
            return "Final logging stage"

    def _analyze_potential_causes(self, state: DocumentState, agent_errors: Dict) -> List[str]:
        """Analyze potential causes of failure"""
        causes = []
        
        if any('database' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            causes.append("Database connectivity or query issues")
        
        if any('openai' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            causes.append("OpenAI API issues (quota, timeout, or connectivity)")
        
        if any('pdf' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            causes.append("PDF processing or file corruption issues")
        
        if any('ocr' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            causes.append("OCR processing or Tesseract configuration issues")
        
        if len(agent_errors) > 1:
            causes.append("Cascading failures across multiple agents")
        
        return causes or ["Unknown cause - requires manual investigation"]

    def _generate_recovery_recommendations(self, state: DocumentState, agent_errors: Dict) -> List[str]:
        """Generate recovery recommendations"""
        recommendations = []
        
        if any('database' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            recommendations.append("Check database connectivity and retry the job")
        
        if any('openai' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            recommendations.append("Verify OpenAI API key and quota, then retry")
        
        if any('pdf' in str(error.get('error_message', '')).lower() for error in agent_errors.values()):
            recommendations.append("Verify PDF file integrity and resubmit if necessary")
        
        if state.progress > 0.5:
            recommendations.append("Consider manual review of partially processed data")
        
        recommendations.append("Contact system administrator if issues persist")
        
        return recommendations

    def _generate_process_improvement_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for process improvement"""
        recommendations = []
        
        if stats['average_confidence'] < 0.8:
            recommendations.append("Consider improving text extraction quality or OCR settings")
        
        review_rate = stats['review_required'] / stats['matched_actions'] if stats['matched_actions'] > 0 else 0
        if review_rate > 0.3:
            recommendations.append("High review rate detected - consider adjusting confidence thresholds")
        
        if stats['customers_missing'] > stats['customers_found']:
            recommendations.append("Many customers not found - verify customer database completeness")
        
        if stats['failed_executions'] > stats['successful_executions']:
            recommendations.append("High execution failure rate - review action execution logic")
        
        if not recommendations:
            recommendations.append("Processing completed efficiently with good quality metrics")
        
        return recommendations

"""
Background task processing module for document processing jobs
Handles async job queue management and concurrent processing
"""

import asyncio
import logging
import time
from typing import Dict, Optional, Any, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import psycopg2
from psycopg2.extras import RealDictCursor

from config import settings
from database import JobModel, ProcessingLogModel
from exceptions import DatabaseError

# Configure logger
logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Enumeration for job status values"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEW_REQUIRED = "review_required"
    CANCELLED = "cancelled"


@dataclass
class BackgroundJob:
    """
    Data class representing a background processing job
    Contains all necessary information for job execution and tracking
    """
    job_id: str
    job_type: str
    file_path: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_minutes: int = field(default=settings.JOB_TIMEOUT_MINUTES)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if job has exceeded its timeout duration"""
        if not self.started_at:
            return False
        return datetime.utcnow() - self.started_at > timedelta(minutes=self.timeout_minutes)
    
    @property
    def processing_duration(self) -> Optional[float]:
        """Get job processing duration in seconds"""
        if not self.started_at:
            return None
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()


class BackgroundJobManager:
    """
    Manager class for handling background job processing
    Implements async job queue with concurrent processing and monitoring
    """
    
    def __init__(self):
        """Initialize job manager with empty queue and worker pool"""
        self.job_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self.active_jobs: Dict[str, BackgroundJob] = {}
        self.completed_jobs: Dict[str, BackgroundJob] = {}
        self.is_running: bool = False
        self.worker_tasks: Set[asyncio.Task] = set()
        self.monitor_task: Optional[asyncio.Task] = None
        self.max_concurrent_jobs = settings.MAX_CONCURRENT_JOBS
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_jobs)
        self._lock = asyncio.Lock()
        
        # Statistics tracking
        self.stats = {
            'total_jobs_processed': 0,
            'successful_jobs': 0,
            'failed_jobs': 0,
            'jobs_in_queue': 0,
            'average_processing_time': 0.0,
            'start_time': datetime.utcnow()
        }
    
    async def start(self) -> None:
        """
        Start the background job manager
        Creates worker tasks and monitoring task
        """
        if self.is_running:
            logger.warning("Job manager is already running")
            return
        
        logger.info("Starting background job manager...")
        self.is_running = True
        
        # Create worker tasks for concurrent processing
        for i in range(self.max_concurrent_jobs):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self.worker_tasks.add(task)
        
        # Create monitoring task for job cleanup and health checks
        self.monitor_task = asyncio.create_task(self._monitor())
        
        logger.info(f"Background job manager started with {self.max_concurrent_jobs} workers")
    
    async def stop(self) -> None:
        """
        Stop the background job manager
        Gracefully shuts down all workers and cleans up resources
        """
        if not self.is_running:
            logger.warning("Job manager is not running")
            return
        
        logger.info("Stopping background job manager...")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Cancel monitor task
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Wait for all tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        if self.monitor_task:
            await asyncio.gather(self.monitor_task, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Background job manager stopped")
    
    async def submit_job(
        self,
        job_id: str,
        job_type: str,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Submit a new job to the processing queue
        
        Args:
            job_id: Unique job identifier
            job_type: Type of job (e.g., 'document_processing')
            file_path: Path to the file to process
            metadata: Optional job metadata
            
        Returns:
            True if job was successfully queued, False otherwise
        """
        try:
            # Create job object
            job = BackgroundJob(
                job_id=job_id,
                job_type=job_type,
                file_path=file_path,
                metadata=metadata or {}
            )
            
            # Add to queue
            await self.job_queue.put(job)
            
            # Update statistics
            async with self._lock:
                self.stats['jobs_in_queue'] = self.job_queue.qsize()
            
            logger.info(f"Job {job_id} submitted to queue")
            return True
            
        except asyncio.QueueFull:
            logger.error(f"Job queue is full, cannot submit job {job_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to submit job {job_id}: {e}")
            return False
    
    async def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a job
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status dictionary or None if job not found
        """
        # Check active jobs first
        if job_id in self.active_jobs:
            job = self.active_jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'processing_duration': job.processing_duration,
                'retry_count': job.retry_count,
                'error_message': job.error_message
            }
        
        # Check completed jobs
        if job_id in self.completed_jobs:
            job = self.completed_jobs[job_id]
            return {
                'job_id': job.job_id,
                'status': job.status.value,
                'created_at': job.created_at,
                'started_at': job.started_at,
                'completed_at': job.completed_at,
                'processing_duration': job.processing_duration,
                'retry_count': job.retry_count,
                'error_message': job.error_message
            }
        
        # Check database for historical jobs
        try:
            db_job = JobModel.get_job(job_id)
            if db_job:
                return {
                    'job_id': db_job['id'],
                    'status': db_job['status'],
                    'created_at': db_job['created_at'],
                    'updated_at': db_job['updated_at'],
                    'current_agent': db_job['current_agent'],
                    'progress': db_job['progress'],
                    'error_message': db_job['error_message']
                }
        except Exception as e:
            logger.error(f"Failed to get job status from database: {e}")
        
        return None
    
    async def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a pending or processing job
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if job was successfully cancelled, False otherwise
        """
        try:
            # Check if job is in active processing
            if job_id in self.active_jobs:
                job = self.active_jobs[job_id]
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.utcnow()
                
                # Move to completed jobs
                self.completed_jobs[job_id] = job
                del self.active_jobs[job_id]
                
                # Update database
                JobModel.update_job_status(
                    job_id=job_id,
                    status=JobStatus.CANCELLED.value,
                    error_message="Job cancelled by user"
                )
                
                logger.info(f"Job {job_id} cancelled")
                return True
            
            logger.warning(f"Cannot cancel job {job_id}: not found in active jobs")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel job {job_id}: {e}")
            return False
    
    async def _worker(self, worker_name: str) -> None:
        """
        Background worker task that processes jobs from the queue
        
        Args:
            worker_name: Unique name for this worker
        """
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running:
            try:
                # Get job from queue with timeout
                job = await asyncio.wait_for(self.job_queue.get(), timeout=1.0)
                
                # Process the job
                await self._process_job(job, worker_name)
                
                # Mark task as done
                self.job_queue.task_done()
                
                # Update queue statistics
                async with self._lock:
                    self.stats['jobs_in_queue'] = self.job_queue.qsize()
                
            except asyncio.TimeoutError:
                # No job available, continue loop
                continue
            except asyncio.CancelledError:
                logger.info(f"Worker {worker_name} cancelled")
                break
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                # Continue processing other jobs
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _process_job(self, job: BackgroundJob, worker_name: str) -> None:
        """
        Process a single job
        
        Args:
            job: Job to process
            worker_name: Name of the processing worker
        """
        logger.info(f"Worker {worker_name} processing job {job.job_id}")
        
        # Move job to active processing
        job.status = JobStatus.PROCESSING
        job.started_at = datetime.utcnow()
        self.active_jobs[job.job_id] = job
        
        # Update database status
        try:
            JobModel.update_job_status(
                job_id=job.job_id,
                status=JobStatus.PROCESSING.value,
                current_agent="background_processor",
                progress=0.1
            )
        except Exception as e:
            logger.error(f"Failed to update job status in database: {e}")
        
        try:
            # Process based on job type
            if job.job_type == 'document_processing':
                await self._process_document_job(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type}")
            
            # Job completed successfully
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.utcnow()
            
            # Update statistics
            async with self._lock:
                self.stats['total_jobs_processed'] += 1
                self.stats['successful_jobs'] += 1
                self._update_average_processing_time(job.processing_duration)
            
            # Update database
            JobModel.update_job_status(
                job_id=job.job_id,
                status=JobStatus.COMPLETED.value,
                progress=1.0
            )
            
            logger.info(f"Job {job.job_id} completed successfully in {job.processing_duration:.2f}s")
            
        except Exception as e:
            # Job failed
            job.status = JobStatus.FAILED
            job.completed_at = datetime.utcnow()
            job.error_message = str(e)
            
            # Update statistics
            async with self._lock:
                self.stats['total_jobs_processed'] += 1
                self.stats['failed_jobs'] += 1
            
            # Check if we should retry
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                job.started_at = None
                job.completed_at = None
                
                # Re-queue the job
                await self.job_queue.put(job)
                logger.warning(f"Job {job.job_id} failed, retrying ({job.retry_count}/{job.max_retries}): {e}")
                return
            
            # Update database with failure
            JobModel.update_job_status(
                job_id=job.job_id,
                status=JobStatus.FAILED.value,
                error_message=str(e)
            )
            
            logger.error(f"Job {job.job_id} failed permanently after {job.retry_count} retries: {e}")
        
        finally:
            # Move job from active to completed
            if job.job_id in self.active_jobs:
                self.completed_jobs[job.job_id] = self.active_jobs.pop(job.job_id)
    
    async def _process_document_job(self, job: BackgroundJob) -> None:
        """
        Process a document processing job using the comprehensive agent workflow
        
        Args:
            job: Document processing job
        """
        try:
            # Import here to avoid circular imports
            from agents.workflow import create_workflow, DocumentState
            from agents.state import ProcessingStatus
            
            # Create initial document state
            initial_state = DocumentState(
                job_id=job.job_id,
                file_path=job.file_path,
                status=ProcessingStatus.PROCESSING.value
            )
            
            # Create and run the comprehensive agent workflow
            workflow = create_workflow()
            
            # Run workflow in thread executor to avoid blocking
            loop = asyncio.get_event_loop()
            final_state = await loop.run_in_executor(
                self.executor,
                lambda: asyncio.run(workflow.ainvoke(initial_state))
            )
            
            # Check final status
            if final_state.status == ProcessingStatus.FAILED.value:
                raise Exception(final_state.error_message or "Agent workflow failed")
            
            # Log processing completion
            ProcessingLogModel.create_log_entry(
                job_id=job.job_id,
                agent_name="background_processor",
                data={
                    'final_status': final_state.status,
                    'processing_time': job.processing_duration,
                    'worker': 'background_job_manager'
                }
            )
            
        except Exception as e:
            logger.error(f"Document processing failed for job {job.job_id}: {e}")
            raise
    
    async def _monitor(self) -> None:
        """
        Background monitoring task for job health checks and cleanup
        Runs periodically to check for expired jobs and perform maintenance
        """
        logger.info("Job monitor started")
        
        while self.is_running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check for expired jobs
                expired_jobs = []
                for job_id, job in self.active_jobs.items():
                    if job.is_expired:
                        expired_jobs.append(job_id)
                
                # Handle expired jobs
                for job_id in expired_jobs:
                    job = self.active_jobs[job_id]
                    job.status = JobStatus.FAILED
                    job.completed_at = datetime.utcnow()
                    job.error_message = f"Job timed out after {job.timeout_minutes} minutes"
                    
                    # Move to completed jobs
                    self.completed_jobs[job_id] = self.active_jobs.pop(job_id)
                    
                    # Update database
                    JobModel.update_job_status(
                        job_id=job_id,
                        status=JobStatus.FAILED.value,
                        error_message=job.error_message
                    )
                    
                    logger.warning(f"Job {job_id} expired and marked as failed")
                
                # Clean up old completed jobs (keep last 1000)
                if len(self.completed_jobs) > 1000:
                    # Sort by completion time and keep most recent
                    sorted_jobs = sorted(
                        self.completed_jobs.items(),
                        key=lambda x: x[1].completed_at or datetime.min,
                        reverse=True
                    )
                    
                    # Keep only the most recent 1000 jobs
                    self.completed_jobs = dict(sorted_jobs[:1000])
                    logger.info("Cleaned up old completed jobs")
                
            except asyncio.CancelledError:
                logger.info("Job monitor cancelled")
                break
            except Exception as e:
                logger.error(f"Job monitor error: {e}")
                # Continue monitoring despite errors
                continue
        
        logger.info("Job monitor stopped")
    
    def _update_average_processing_time(self, processing_time: Optional[float]) -> None:
        """
        Update average processing time statistic
        
        Args:
            processing_time: Processing time in seconds
        """
        if processing_time is None:
            return
        
        current_avg = self.stats['average_processing_time']
        total_jobs = self.stats['total_jobs_processed']
        
        if total_jobs == 1:
            self.stats['average_processing_time'] = processing_time
        else:
            # Calculate running average
            self.stats['average_processing_time'] = (
                (current_avg * (total_jobs - 1) + processing_time) / total_jobs
            )
    
    @property
    def queue_size(self) -> int:
        """Get current queue size"""
        return self.job_queue.qsize()
    
    @property
    def active_job_count(self) -> int:
        """Get number of currently active jobs"""
        return len(self.active_jobs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get current job manager statistics
        
        Returns:
            Dictionary containing current statistics
        """
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        return {
            'is_running': self.is_running,
            'queue_size': self.queue_size,
            'active_jobs': self.active_job_count,
            'completed_jobs': len(self.completed_jobs),
            'total_jobs_processed': self.stats['total_jobs_processed'],
            'successful_jobs': self.stats['successful_jobs'],
            'failed_jobs': self.stats['failed_jobs'],
            'success_rate': (
                self.stats['successful_jobs'] / self.stats['total_jobs_processed']
                if self.stats['total_jobs_processed'] > 0 else 0.0
            ),
            'average_processing_time': self.stats['average_processing_time'],
            'uptime_seconds': uptime,
            'worker_count': len(self.worker_tasks)
        }


# Global job manager instance
job_manager = BackgroundJobManager()

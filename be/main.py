"""
FastAPI main application file
Entry point for the Document Processing API with comprehensive agent workflow
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from database import JobModel
from config import settings
from agents.workflow import create_workflow, DocumentState
from agents.state import ProcessingStatus
from background import BackgroundJobManager
from exceptions import PDFError, LLMError, DatabaseError

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(getattr(settings, 'LOG_FILE', './logs/app.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=getattr(settings, 'APP_VERSION', '1.0.0'),
    description="API for processing documents with LLM-powered extraction and action execution",
    debug=settings.DEBUG
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global job manager for background processing
job_manager = BackgroundJobManager()

# Pydantic models for API responses
class JobResponse(BaseModel):
    """Response model for job creation"""
    job_id: str
    status: str
    message: str
    created_at: datetime

class ActionExecutionResult(BaseModel):
    """Model for action execution results"""
    action_name: str
    customer_id: str
    national_id: Optional[str] = None
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None
    executed_at: Optional[str] = None
    execution_type: Optional[str] = None

class StatusResponse(BaseModel):
    """Response model for job status"""
    job_id: str
    status: str
    progress: float
    results: Optional[Dict]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime
    # New fields for action execution results
    executed_actions: Optional[List[ActionExecutionResult]] = []
    failed_actions: Optional[List[ActionExecutionResult]] = []
    pending_actions: Optional[List[Dict]] = []
    action_execution_summary: Optional[Dict] = None

class ReviewItem(BaseModel):
    """Model for items requiring human review"""
    id: str
    job_id: str
    national_id: str
    customer_id: Optional[str]
    original_action: str
    matched_action: Optional[str]
    confidence: Optional[float]
    page_number: Optional[int]
    context: Optional[str]
    review_reason: Optional[str]
    priority: int
    created_at: datetime

class ReviewSubmission(BaseModel):
    """Model for review submissions"""
    approved: bool
    corrected_id: Optional[str] = None
    corrected_action: Optional[str] = None
    comments: Optional[str] = None

class MetricsResponse(BaseModel):
    """Response model for system metrics"""
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    jobs_in_review: int
    average_processing_time: float
    current_queue_size: int
    total_customers: int
    supported_actions: List[str]

# Database connection helper
def get_db_connection():
    """
    Create and return a PostgreSQL database connection using psycopg2
    Returns connection with RealDictCursor for easier data handling
    """
    try:
        conn = psycopg2.connect(
            host=settings.DATABASE_HOST,
            port=settings.DATABASE_PORT,
            database=settings.DATABASE_NAME,
            user=settings.DATABASE_USER,
            password=settings.DATABASE_PASSWORD,
            cursor_factory=RealDictCursor
        )
        return conn
    except psycopg2.Error as e:
        logger.error(f"Database connection failed: {e}")
        raise DatabaseError(f"Failed to connect to database: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Application startup tasks
    Initialize database tables, ChromaDB, and background job manager
    """
    logger.info("Starting Document Processing API...")
    try:
        # Verify existing tables
        await verify_existing_tables()
        # Initialize supporting tables (jobs, review_queue, processing_logs)
        await initialize_supporting_tables()
        # Start background job manager
        await job_manager.start()
        # Initialize ChromaDB with actions from database
        from vector_store import initialize_action_store
        initialize_action_store()
        logger.info("Application startup completed successfully")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown tasks
    Clean up background tasks and close connections
    """
    logger.info("Shutting down Document Processing API...")
    await job_manager.stop()
    logger.info("Application shutdown completed")

async def verify_existing_tables():
    """
    Verify that the required customers and actions tables exist
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Check if customers table exists and get structure
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'customers'
                ORDER BY ordinal_position
            """)
            customer_columns = cursor.fetchall()
            if not customer_columns:
                raise DatabaseError("Customers table not found. Please ensure it exists with national_id and customer_id columns.")

            # Check if actions table exists
            cursor.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'actions'
                ORDER BY ordinal_position
            """)
            action_columns = cursor.fetchall()
            if not action_columns:
                raise DatabaseError("Actions table not found. Please ensure it exists with action_name and description columns.")

            # Get counts
            cursor.execute("SELECT COUNT(*) as count FROM customers")
            customer_count = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) as count FROM actions")
            action_count = cursor.fetchone()['count']
            logger.info(f"Existing tables verified: {customer_count} customers, {action_count} actions")
    except psycopg2.Error as e:
        logger.error(f"Failed to verify existing tables: {e}")
        raise DatabaseError(f"Failed to verify existing tables: {e}")
    finally:
        conn.close()

async def initialize_supporting_tables():
    """
    Initialize supporting tables (jobs, review_queue, processing_logs)
    Does NOT touch existing customers and actions tables
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Create jobs table with UUID
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_id VARCHAR(100) UNIQUE NOT NULL,
                    filename VARCHAR(255) NOT NULL,
                    file_path VARCHAR(500) NOT NULL,
                    status VARCHAR(50) NOT NULL DEFAULT 'pending',
                    progress DECIMAL(5,2) DEFAULT 0.00,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    started_at TIMESTAMP NULL,
                    completed_at TIMESTAMP NULL,
                    error_message TEXT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb
                )
            """)

            # Create review_queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS review_queue (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_id VARCHAR(100) NOT NULL,
                    national_id VARCHAR(20) NOT NULL,
                    customer_id VARCHAR(20),
                    original_action TEXT NOT NULL,
                    matched_action VARCHAR(50),
                    confidence DECIMAL(5,2),
                    page_number INTEGER,
                    context TEXT,
                    review_reason TEXT,
                    status VARCHAR(50) DEFAULT 'pending',
                    priority INTEGER DEFAULT 5,
                    assigned_to VARCHAR(100),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    reviewed_at TIMESTAMP NULL,
                    reviewed_by VARCHAR(100),
                    review_decision VARCHAR(50),
                    review_notes TEXT,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    CONSTRAINT fk_review_jobs FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)

            # Create processing_logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    job_id VARCHAR(100) NOT NULL,
                    agent_name VARCHAR(100) NOT NULL,
                    log_level VARCHAR(20) DEFAULT 'INFO',
                    message TEXT NOT NULL,
                    data JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_logs_jobs FOREIGN KEY (job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                )
            """)

            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_job_id ON review_queue(job_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_job_id ON processing_logs(job_id)")
            conn.commit()
            logger.info("Supporting database tables initialized successfully")
    except psycopg2.Error as e:
        logger.error(f"Database initialization failed: {e}")
        conn.rollback()
        raise DatabaseError(f"Failed to initialize database: {e}")
    finally:
        conn.close()

# Enhanced helper functions for action tracking
async def execute_approved_actions(job_id: str, approved_items: list):
    """
    Execute actions for approved items and track execution results
    """
    execution_results = {
        'executed_actions': [],
        'failed_actions': [],
        'total_executed': 0,
        'total_failed': 0
    }

    try:
        from services.action_executor import action_executor
        for item in approved_items:
            # Handle both dictionary and object types
            def get_item_value(item, key, default=None):
                if hasattr(item, key):
                    return getattr(item, key, default)
                elif isinstance(item, dict):
                    return item.get(key, default)
                else:
                    return default

            customer_id = get_item_value(item, 'customer_id')
            action_name = get_item_value(item, 'matched_action')
            national_id = get_item_value(item, 'national_id')

            if customer_id and action_name:
                try:
                    # Execute the action
                    result = await action_executor.execute_action(customer_id, action_name)
                    # Track successful execution
                    execution_results['executed_actions'].append({
                        'action_name': action_name,
                        'customer_id': customer_id,
                        'national_id': national_id,
                        'status': 'success',
                        'result': result,
                        'executed_at': datetime.utcnow().isoformat()
                    })
                    execution_results['total_executed'] += 1
                    logger.info(f"Successfully executed {action_name} for customer {customer_id}")
                except Exception as action_error:
                    logger.error(f"Failed to execute {action_name} for customer {customer_id}: {action_error}")
                    # Track failed execution
                    execution_results['failed_actions'].append({
                        'action_name': action_name,
                        'customer_id': customer_id,
                        'national_id': national_id,
                        'status': 'failed',
                        'error': str(action_error),
                        'attempted_at': datetime.utcnow().isoformat()
                    })
                    execution_results['total_failed'] += 1
    except Exception as e:
        logger.error(f"Failed to execute approved actions: {e}")
        execution_results['error'] = str(e)

    return execution_results

async def populate_review_queue(job_id: str, review_items: list):
    """
    Populate the review_queue table with items requiring human review
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            for item in review_items:
                # Handle both dictionary and object types
                def get_item_value(item, key, default=None):
                    if hasattr(item, key):
                        return getattr(item, key, default)
                    elif isinstance(item, dict):
                        return item.get(key, default)
                    else:
                        return default

                # Determine priority based on confidence and risk factors
                confidence = get_item_value(item, 'confidence', 0)
                risk_factors = get_item_value(item, 'risk_factors', [])
                priority = determine_priority_from_values(confidence, risk_factors)

                cursor.execute("""
                    INSERT INTO review_queue (
                        job_id, national_id, customer_id, original_action,
                        matched_action, confidence, page_number, context,
                        review_reason, priority, status, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    job_id,
                    get_item_value(item, 'national_id'),
                    get_item_value(item, 'customer_id'),
                    get_item_value(item, 'original_action'),
                    get_item_value(item, 'matched_action'),
                    get_item_value(item, 'confidence'),
                    get_item_value(item, 'page_number'),
                    get_item_value(item, 'context'),
                    ', '.join(get_item_value(item, 'review_reasons', [])),
                    priority,
                    'pending',
                    datetime.utcnow()
                ))
            conn.commit()
            logger.info(f"Successfully added {len(review_items)} items to review queue")
    except psycopg2.Error as e:
        logger.error(f"Failed to populate review queue: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def determine_priority_from_values(confidence, risk_factors):
    """
    Determine priority based on confidence score and risk factors
    """
    if confidence < 0.3 or (risk_factors and 'high_risk_customer' in risk_factors):
        return 1  # Critical
    elif confidence < 0.5:
        return 2  # High
    elif confidence < 0.7:
        return 3  # Medium
    else:
        return 4  # Low

# Helper function to extract customer mapping statistics
def extract_customer_mapping_stats(validated_pairs):
    """Extract customer mapping statistics from validated pairs"""
    if not validated_pairs:
        return 0, 0, []

    customers_found = []
    customers_not_found = []
    for pair in validated_pairs:
        if hasattr(pair, 'customer_found') and pair.customer_found:
            customers_found.append(pair.national_id)
        else:
            customers_not_found.append(pair.national_id)

    return len(customers_found), len(customers_not_found), customers_not_found

# Enhanced process_document function with detailed action tracking and fixed case detection
async def process_document(job_id: str, file_path: str):
    """
    Background task to process document through agent workflow.
    Fixed case handling:
    1. National IDs extracted but none map to a customer (customer_mapping_failed)
    2. No national IDs extracted
    3. National IDs found, customers found, but no valid actions identified
    4. Regular review or auto-approval flow
    """
    logger.info(f"Starting document processing for job {job_id}")
    try:
        # Update job status to processing
        JobModel.update_job_status(job_id, ProcessingStatus.PROCESSING.value, 0.1)

        # Initialize document state
        initial_state = DocumentState(
            job_id=job_id,
            file_path=file_path,
            status=ProcessingStatus.PROCESSING.value
        )

        try:
            from agents.workflow import create_workflow
            workflow = create_workflow()
            # Execute workflow
            result = await workflow.ainvoke(initial_state)

            # --- Extract all relevant workflow result fields ---
            auto_completed_items = []
            review_items = []
            auto_approved_items = []
            extracted_pairs_count = 0
            unmatched_national_ids = []

            def get_item_value(item, key, default=None):
                if hasattr(item, key):
                    return getattr(item, key, default)
                elif isinstance(item, dict):
                    return item.get(key, default)
                else:
                    return default

            # Extract lists safely
            if hasattr(result, 'auto_completed_pairs'):
                auto_completed_items = result.auto_completed_pairs or []
            elif hasattr(result, 'auto_completed'):
                auto_completed_items = result.auto_completed or []
            elif isinstance(result, dict):
                auto_completed_items = result.get('auto_completed', [])

            if hasattr(result, 'review_required'):
                review_items = result.review_required or []
            elif isinstance(result, dict):
                review_items = result.get('review_required', [])

            if hasattr(result, 'auto_approved'):
                auto_approved_items = result.auto_approved or []
            elif isinstance(result, dict):
                auto_approved_items = result.get('auto_approved', [])

            if hasattr(result, 'unmatched_national_ids'):
                unmatched_national_ids = [i for i in (result.unmatched_national_ids or []) if i]
            elif isinstance(result, dict):
                unmatched_national_ids = [i for i in (result.get('unmatched_national_ids', []) or []) if i]

            # Extract customer mapping statistics using the helper function
            if hasattr(result, 'validated_pairs') and result.validated_pairs:
                customers_found_count, customers_not_found_count, not_found_ids = extract_customer_mapping_stats(result.validated_pairs)
                extracted_pairs_count = len(result.validated_pairs)
                all_extracted_ids = [get_item_value(i, 'national_id') for i in result.validated_pairs if get_item_value(i, 'national_id')]
            elif hasattr(result, 'extracted_pairs') and result.extracted_pairs:
                customers_found_count, customers_not_found_count, not_found_ids = extract_customer_mapping_stats(result.extracted_pairs)
                extracted_pairs_count = len(result.extracted_pairs)
                all_extracted_ids = [get_item_value(i, 'national_id') for i in result.extracted_pairs if get_item_value(i, 'national_id')]
            elif isinstance(result, dict):
                extracted_pairs = result.get('extracted_pairs', []) or result.get('validated_pairs', [])
                customers_found_count, customers_not_found_count, not_found_ids = extract_customer_mapping_stats(extracted_pairs)
                extracted_pairs_count = len(extracted_pairs)
                all_extracted_ids = [get_item_value(i, 'national_id') for i in extracted_pairs if get_item_value(i, 'national_id')]
            else:
                customers_found_count, customers_not_found_count, not_found_ids = 0, 0, []
                all_extracted_ids = []

            logger.info(f"Extracted counts - extracted_pairs: {extracted_pairs_count}, customers_found: {customers_found_count}, customers_not_found: {customers_not_found_count}, auto_completed: {len(auto_completed_items)}, review: {len(review_items)}, auto_approved: {len(auto_approved_items)}")

            # ------ CASE 1: National IDs extracted, but none map to a customer ------
            if (
                extracted_pairs_count > 0
                and customers_not_found_count == extracted_pairs_count  # All customers not found
                and customers_found_count == 0  # No customers found
                and not review_items
                and not auto_approved_items
                and len(auto_completed_items) == 0  # No genuinely auto-completed items
            ):
                logger.info("CASE 1: National IDs extracted, but none map to customers (customer_mapping_failed)")
                job_metadata = {
                    'processing_completed_at': datetime.utcnow().isoformat(),
                    'extracted_pairs': extracted_pairs_count,
                    'actions_summary': {
                        'processing_outcome': 'customer_mapping_failed_no_actions',
                        'no_action_reason': 'customer_not_in_system',
                        'outcome_details': {
                            'message': 'National IDs extracted but no matching customer records were found',
                            'description': (
                                'The document contains valid national-id numbers but '
                                'none of them correspond to a customer in the core system'
                            ),
                            'case_type': 'customer_mapping_failed',
                            'national_ids': not_found_ids,
                            'possible_reasons': [
                                'Customer not yet onboarded',
                                'National-ID typo in the document',
                                'Stale/legacy national-id not present in master DB',
                            ],
                        },
                        'total_actions_found': 0,
                        'auto_approved_count': 0,
                        'review_required_count': 0,
                        'auto_completed_count': 0,
                        'executed_actions': [],
                        'failed_actions': [],
                        'pending_reviews': [],
                        'completed_items': []
                    },
                }

                JobModel.update_job_status(
                    job_id,
                    ProcessingStatus.COMPLETED.value,
                    1.0,
                    None,
                    job_metadata,
                )
                logger.info(f"Job {job_id} completed – national IDs not mapped to customers")
                return

            # ------ CASE 2: No national IDs extracted at all ------
            if extracted_pairs_count == 0 and not review_items and not auto_approved_items and not auto_completed_items:
                logger.info("CASE 2: No national IDs extracted from PDF")
                job_metadata = {
                    'processing_completed_at': datetime.utcnow().isoformat(),
                    'extracted_pairs': 0,
                    'actions_summary': {
                        'processing_outcome': 'no_national_ids_found',
                        'no_action_reason': 'no_national_ids_found',
                        'outcome_details': {
                            'message': 'No national IDs found in document',
                            'description': 'The document was processed but no valid national ID numbers could be extracted from the text',
                            'case_type': 'no_national_ids_extracted',
                            'possible_reasons': [
                                'Document does not contain any national ID numbers',
                                'National IDs are in unrecognizable format',
                                'Document text quality is too poor for extraction',
                                'Document is not in supported language or format'
                            ]
                        },
                        'total_actions_found': 0,
                        'auto_approved_count': 0,
                        'review_required_count': 0,
                        'auto_completed_count': 0,
                        'executed_actions': [],
                        'failed_actions': [],
                        'pending_reviews': [],
                        'completed_items': []
                    }
                }
                JobModel.update_job_status(job_id, ProcessingStatus.COMPLETED.value, 1.0, None, job_metadata)
                logger.info(f"Job {job_id} completed - no national IDs found in PDF")
                return

            # ------ CASE 3: National IDs found, customers found, but no valid actions identified ------
            elif (
                extracted_pairs_count > 0
                and customers_found_count > 0  # At least some customers found
                and not review_items
                and not auto_approved_items
                and len(auto_completed_items) == 0
            ):
                logger.info("CASE 3: National IDs found, customers found, but no valid actions identified")
                job_metadata = {
                    'processing_completed_at': datetime.utcnow().isoformat(),
                    'extracted_pairs': extracted_pairs_count,
                    'actions_summary': {
                        'processing_outcome': 'no_valid_actions_found',
                        'no_action_reason': 'no_valid_actions_found',
                        'outcome_details': {
                            'message': 'No valid banking actions found in document',
                            'description': 'National IDs were extracted and customers found but no recognizable banking actions could be identified in the document text',
                            'case_type': 'no_valid_actions_identified',
                            'extracted_pairs_count': extracted_pairs_count,
                            'customers_found': customers_found_count,
                            'possible_reasons': [
                                'Document contains no actionable banking requests',
                                'Banking actions are not in recognizable format',
                                'Actions mentioned are not supported by the system',
                                'Low confidence scores for all potential action matches'
                            ]
                        },
                        'total_actions_found': 0,
                        'auto_approved_count': 0,
                        'review_required_count': 0,
                        'auto_completed_count': 0,
                        'executed_actions': [],
                        'failed_actions': [],
                        'pending_reviews': [],
                        'completed_items': []
                    }
                }
                JobModel.update_job_status(job_id, ProcessingStatus.COMPLETED.value, 1.0, None, job_metadata)
                logger.info(f"Job {job_id} completed - no valid actions found")
                return

            # -- CONTINUE WITH REGULAR PROCESSING FOR REVIEW/AUTO-APPROVED CASES --
            job_metadata = {
                'processing_completed_at': datetime.utcnow().isoformat(),
                'extracted_pairs': len(review_items) + len(auto_approved_items),
                'actions_summary': {
                    'total_actions_found': len(review_items) + len(auto_approved_items),
                    'auto_approved_count': len(auto_approved_items),
                    'review_required_count': len(review_items),
                    'executed_actions': [],
                    'failed_actions': [],
                    'pending_reviews': [],
                    'completed_items': [],
                    'no_action_reason': None,
                    'processing_outcome': None
                }
            }

            # Handle review required
            if review_items:
                await populate_review_queue(job_id, review_items)
                job_metadata['actions_summary']['processing_outcome'] = 'review_required'
                job_metadata['actions_summary']['pending_reviews'] = [
                    {
                        'action_name': get_item_value(item, 'matched_action') or get_item_value(item, 'action_name'),
                        'customer_id': get_item_value(item, 'customer_id'),
                        'national_id': get_item_value(item, 'national_id'),
                        'confidence': get_item_value(item, 'confidence'),
                        'status': 'pending_review'
                    }
                    for item in review_items
                ]
                JobModel.update_job_status(job_id, ProcessingStatus.REVIEW_REQUIRED.value, 0.8, None, job_metadata)
                logger.info(f"Added {len(review_items)} items to review queue for job {job_id}")

            # Handle auto-approved
            elif auto_approved_items:
                execution_results = await execute_approved_actions(job_id, auto_approved_items)
                job_metadata['actions_summary']['executed_actions'] = execution_results['executed_actions']
                job_metadata['actions_summary']['failed_actions'] = execution_results['failed_actions']
                job_metadata['actions_summary']['total_executed'] = execution_results['total_executed']
                job_metadata['actions_summary']['total_failed'] = execution_results['total_failed']

                if execution_results['total_executed'] > 0:
                    job_metadata['actions_summary']['processing_outcome'] = 'actions_executed'
                elif execution_results['total_failed'] > 0:
                    job_metadata['actions_summary']['processing_outcome'] = 'actions_failed'

                JobModel.update_job_status(job_id, ProcessingStatus.COMPLETED.value, 1.0, None, job_metadata)
                logger.info(f"Auto-executed {execution_results['total_executed']} actions for job {job_id}")

        except Exception as workflow_error:
            logger.error(f"Workflow execution failed: {workflow_error}")
            JobModel.update_job_status(
                job_id,
                ProcessingStatus.FAILED.value,
                0.0,
                str(workflow_error),
                {
                    'error_details': str(workflow_error),
                    'failed_at': datetime.utcnow().isoformat(),
                    'no_action_reason': 'processing_failed',
                    'processing_outcome': 'failed',
                    'actions_summary': {
                        'outcome_details': {
                            'message': 'Document processing failed',
                            'description': 'An error occurred during document processing',
                            'case_type': 'processing_error'
                        }
                    }
                }
            )

    except Exception as e:
        logger.error(f"Document processing failed for job {job_id}: {e}")
        JobModel.update_job_status(
            job_id,
            ProcessingStatus.FAILED.value,
            0.0,
            str(e),
            {
                'error_details': str(e),
                'failed_at': datetime.utcnow().isoformat(),
                'no_action_reason': 'processing_failed',
                'processing_outcome': 'failed',
                'actions_summary': {
                    'outcome_details': {
                        'message': 'Document processing failed',
                        'description': 'A system error occurred during document processing',
                        'case_type': 'system_error'
                    }
                }
            }
        )

# Helper function to get action execution results for status endpoint
async def get_action_execution_results(job_id: str, job_metadata: Dict):
    """
    Extract and format action execution results from job metadata
    """
    executed_actions = []
    failed_actions = []
    pending_actions = []
    action_execution_summary = None

    if job_metadata and 'actions_summary' in job_metadata:
        actions_summary = job_metadata['actions_summary']
        
        # Extract executed actions
        if 'executed_actions' in actions_summary:
            for action in actions_summary['executed_actions']:
                executed_actions.append(ActionExecutionResult(
                    action_name=action.get('action_name', ''),
                    customer_id=action.get('customer_id', ''),
                    national_id=action.get('national_id'),
                    status=action.get('status', 'unknown'),
                    result=action.get('result'),
                    executed_at=action.get('executed_at'),
                    execution_type=action.get('execution_type', 'auto')
                ))
        
        # Extract reviewed/approved actions
        if 'reviewed_actions' in actions_summary:
            for action in actions_summary['reviewed_actions']:
                executed_actions.append(ActionExecutionResult(
                    action_name=action.get('action_name', ''),
                    customer_id=action.get('customer_id', ''),
                    national_id=action.get('national_id'),
                    status=action.get('status', 'unknown'),
                    result=action.get('result'),
                    executed_at=action.get('executed_at'),
                    execution_type=action.get('execution_type', 'human_approved')
                ))
        
        # Extract failed actions
        if 'failed_actions' in actions_summary:
            for action in actions_summary['failed_actions']:
                failed_actions.append(ActionExecutionResult(
                    action_name=action.get('action_name', ''),
                    customer_id=action.get('customer_id', ''),
                    national_id=action.get('national_id'),
                    status=action.get('status', 'failed'),
                    error=action.get('error'),
                    executed_at=action.get('attempted_at'),
                    execution_type=action.get('execution_type', 'auto')
                ))
        
        # Extract pending reviews as pending actions
        if 'pending_reviews' in actions_summary:
            pending_actions = actions_summary['pending_reviews']
        
        # Create execution summary
        action_execution_summary = {
            'total_actions_found': actions_summary.get('total_actions_found', 0),
            'total_executed': len(executed_actions),
            'total_failed': len(failed_actions),
            'total_pending': len(pending_actions),
            'processing_outcome': actions_summary.get('processing_outcome'),
            'auto_approved_count': actions_summary.get('auto_approved_count', 0),
            'review_required_count': actions_summary.get('review_required_count', 0)
        }
        
        # Add outcome details if available
        if 'outcome_details' in actions_summary:
            action_execution_summary['outcome_details'] = actions_summary['outcome_details']
    
    return executed_actions, failed_actions, pending_actions, action_execution_summary

# API Endpoints
@app.post("/upload", response_model=JobResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a PDF document
    Validates file, saves it, and starts comprehensive agent processing
    """
    # Validate file type and size
    if not file.filename.lower().endswith(tuple(settings.ALLOWED_FILE_TYPES)):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed types: {settings.ALLOWED_FILE_TYPES}"
        )

    # Check file size
    file_content = await file.read()
    if len(file_content) > settings.MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # Ensure upload directory exists
    os.makedirs(settings.UPLOAD_DIRECTORY, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(settings.UPLOAD_DIRECTORY, f"{job_id}_{file.filename}")
    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except IOError as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Create job record in database
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO jobs (job_id, filename, file_path, status, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (job_id, file.filename, file_path, ProcessingStatus.PENDING.value, datetime.utcnow()))
            conn.commit()
    except psycopg2.Error as e:
        logger.error(f"Failed to create job record: {e}")
        raise DatabaseError(f"Failed to create job record: {e}")
    finally:
        conn.close()

    # Start background processing
    background_tasks.add_task(process_document, job_id, file_path)
    logger.info(f"Document upload successful. Job ID: {job_id}")

    return JobResponse(
        job_id=job_id,
        status=ProcessingStatus.PENDING.value,
        message="Document uploaded successfully. Processing started.",
        created_at=datetime.utcnow()
    )

@app.get("/status/{job_id}", response_model=StatusResponse)
async def get_job_status(job_id: str):
    """
    Get the current status of a processing job
    Returns detailed status including progress and action execution results
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT job_id, status, progress, metadata,
                       error_message, created_at,
                       COALESCE(completed_at, started_at, created_at) as updated_at
                FROM jobs WHERE job_id = %s
            """, (job_id,))
            job = cursor.fetchone()

            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Extract action execution results from metadata
            executed_actions, failed_actions, pending_actions, action_execution_summary = await get_action_execution_results(
                job_id, job['metadata']
            )

            return StatusResponse(
                job_id=job['job_id'],
                status=job['status'],
                progress=float(job['progress'] or 0.0),
                results=job['metadata'],
                error=job['error_message'],
                created_at=job['created_at'],
                updated_at=job['updated_at'],
                # New action execution fields
                #executed_actions=executed_actions,
                failed_actions=failed_actions,
                pending_actions=pending_actions,
                action_execution_summary=action_execution_summary
            )

    except psycopg2.Error as e:
        logger.error(f"Failed to get job status: {e}")
        raise DatabaseError(f"Failed to get job status: {e}")
    finally:
        conn.close()

@app.get("/review_queue", response_model=List[ReviewItem])
async def get_review_queue():
    """
    Get all items currently in the review queue
    Returns list of items requiring human review
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, job_id, national_id, customer_id, original_action,
                       matched_action, confidence, page_number, context,
                       review_reason, priority, created_at
                FROM review_queue
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
            """)
            items = cursor.fetchall()

            return [ReviewItem(
                id=str(item['id']),
                job_id=item['job_id'],
                national_id=item['national_id'],
                customer_id=item['customer_id'],
                original_action=item['original_action'],
                matched_action=item['matched_action'],
                confidence=float(item['confidence']) if item['confidence'] else None,
                page_number=item['page_number'],
                context=item['context'],
                review_reason=item['review_reason'],
                priority=item['priority'],
                created_at=item['created_at']
            ) for item in items]

    except psycopg2.Error as e:
        logger.error(f"Failed to get review queue: {e}")
        raise DatabaseError(f"Failed to get review queue: {e}")
    finally:
        conn.close()

@app.post("/review/{review_id}")
async def submit_review(review_id: str, review: ReviewSubmission):
    """
    Submit human review for items in the review queue
    Updates review status and continues processing if approved
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Get the review item details first
            cursor.execute("""
                SELECT job_id, national_id, customer_id, matched_action, status
                FROM review_queue
                WHERE id = %s AND status = 'pending'
            """, (review_id,))
            review_item = cursor.fetchone()

            if not review_item:
                raise HTTPException(status_code=404, detail="Review item not found or already processed")

            job_id = review_item['job_id']
            customer_id = review_item['customer_id']
            action_name = review_item['matched_action']
            national_id = review_item['national_id']

            # Update review queue item
            cursor.execute("""
                UPDATE review_queue
                SET status = 'reviewed',
                    review_decision = %s,
                    review_notes = %s,
                    reviewed_at = %s,
                    reviewed_by = %s,
                    metadata = metadata || %s::jsonb
                WHERE id = %s
            """, (
                'approved' if review.approved else 'rejected',
                review.comments,
                datetime.utcnow(),
                'system_user',
                json.dumps({"corrected_id": review.corrected_id or '', "corrected_action": review.corrected_action or ''}),
                review_id
            ))

            execution_result = None

            # If approved, execute the action and track result
            if review.approved:
                try:
                    # Use corrected values if provided
                    final_customer_id = review.corrected_id or customer_id
                    final_action = review.corrected_action or action_name

                    if final_customer_id and final_action:
                        from services.action_executor import action_executor
                        result = await action_executor.execute_action(final_customer_id, final_action)

                        execution_result = {
                            'action_name': final_action,
                            'customer_id': final_customer_id,
                            'national_id': national_id,
                            'status': 'success',
                            'result': result,
                            'executed_at': datetime.utcnow().isoformat(),
                            'review_id': review_id,
                            'execution_type': 'human_approved'
                        }
                        logger.info(f"Executed {final_action} for customer {final_customer_id}: {result}")
                except Exception as e:
                    logger.error(f"Failed to execute approved action: {e}")
                    execution_result = {
                        'action_name': final_action,
                        'customer_id': final_customer_id,
                        'national_id': national_id,
                        'status': 'failed',
                        'error': str(e),
                        'attempted_at': datetime.utcnow().isoformat(),
                        'review_id': review_id,
                        'execution_type': 'human_approved'
                    }

            # Update job metadata with the executed action
            if execution_result:
                # Get current job metadata
                cursor.execute("""
                    SELECT metadata FROM jobs WHERE job_id = %s
                """, (job_id,))
                job_row = cursor.fetchone()
                current_metadata = job_row['metadata'] if job_row and job_row['metadata'] else {}

                # Initialize actions_summary if it doesn't exist
                if 'actions_summary' not in current_metadata:
                    current_metadata['actions_summary'] = {
                        'executed_actions': [],
                        'failed_actions': [],
                        'reviewed_actions': [],
                        'total_executed': 0,
                        'total_failed': 0
                    }

                # Add the execution result to the appropriate list
                if execution_result['status'] == 'success':
                    current_metadata['actions_summary']['reviewed_actions'] = current_metadata['actions_summary'].get('reviewed_actions', [])
                    current_metadata['actions_summary']['reviewed_actions'].append(execution_result)
                    current_metadata['actions_summary']['total_executed'] = current_metadata['actions_summary'].get('total_executed', 0) + 1
                else:
                    current_metadata['actions_summary']['failed_actions'] = current_metadata['actions_summary'].get('failed_actions', [])
                    current_metadata['actions_summary']['failed_actions'].append(execution_result)
                    current_metadata['actions_summary']['total_failed'] = current_metadata['actions_summary'].get('total_failed', 0) + 1

                # Update the job metadata
                cursor.execute("""
                    UPDATE jobs
                    SET metadata = %s
                    WHERE job_id = %s
                """, (
                    json.dumps(current_metadata),
                    job_id
                ))

                # Also log the execution in processing_logs
                cursor.execute("""
                    INSERT INTO processing_logs (job_id, agent_name, message, log_level, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    job_id,
                    'ReviewAgent',
                    f"Action {final_action} {'executed successfully' if execution_result['status'] == 'success' else 'failed'} for customer {final_customer_id} after human approval",
                    'INFO' if execution_result['status'] == 'success' else 'ERROR',
                    datetime.utcnow()
                ))

            conn.commit()

            # Check if all review items for this job are completed
            cursor.execute("""
                SELECT COUNT(*) as pending_count
                FROM review_queue
                WHERE job_id = %s AND status = 'pending'
            """, (job_id,))
            pending_count = cursor.fetchone()['pending_count']

            # If no pending reviews left, update job status to completed
            if pending_count == 0:
                JobModel.update_job_status(
                    job_id,
                    ProcessingStatus.COMPLETED.value,
                    1.0,
                    None,
                    None  # Don't overwrite the metadata we just updated
                )
                logger.info(f"Job {job_id} completed - all reviews processed")

            return {
                "message": "Review submitted successfully",
                "job_id": job_id,
                "approved": review.approved,
                "remaining_reviews": pending_count,
                "execution_result": execution_result
            }

    except HTTPException:
        raise
    except psycopg2.Error as e:
        logger.error(f"Failed to submit review: {e}")
        conn.rollback()
        raise DatabaseError(f"Failed to submit review: {e}")
    finally:
        conn.close()

@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """
    Get system metrics and statistics
    Returns processing statistics and current system status
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # Get job counts by status
            cursor.execute("""
                SELECT status, COUNT(*) as count
                FROM jobs
                GROUP BY status
            """)
            status_counts = {row['status']: row['count'] for row in cursor.fetchall()}

            # Get average processing time for completed jobs
            cursor.execute("""
                SELECT AVG(EXTRACT(EPOCH FROM (completed_at - created_at))) as avg_time
                FROM jobs
                WHERE status = 'completed' AND completed_at IS NOT NULL
            """)
            avg_time_result = cursor.fetchone()
            avg_processing_time = float(avg_time_result['avg_time'] or 0.0)

            # Get review queue size
            cursor.execute("""
                SELECT COUNT(*) as count
                FROM review_queue
                WHERE status = 'pending'
            """)
            review_count = cursor.fetchone()['count']

            # Get customer count
            cursor.execute("SELECT COUNT(*) as count FROM customers")
            customer_count = cursor.fetchone()['count']

            # Get supported actions
            cursor.execute("SELECT action_name FROM actions ORDER BY action_name")
            actions = [row['action_name'] for row in cursor.fetchall()]

            return MetricsResponse(
                total_jobs=sum(status_counts.values()),
                successful_jobs=status_counts.get('completed', 0),
                failed_jobs=status_counts.get('failed', 0),
                jobs_in_review=review_count,
                average_processing_time=avg_processing_time,
                current_queue_size=job_manager.queue_size if hasattr(job_manager, 'queue_size') else 0,
                total_customers=customer_count,
                supported_actions=actions
            )

    except psycopg2.Error as e:
        logger.error(f"Failed to get metrics: {e}")
        raise DatabaseError(f"Failed to get metrics: {e}")
    finally:
        conn.close()

@app.get("/customers/search")
async def search_customers(national_id: Optional[str] = None, customer_id: Optional[str] = None):
    """
    Search customers by national_id or customer_id
    """
    if not national_id and not customer_id:
        raise HTTPException(status_code=400, detail="Either national_id or customer_id must be provided")

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            if national_id:
                cursor.execute("""
                    SELECT customer_id, national_id::text as national_id
                    FROM customers
                    WHERE national_id::text = %s
                """, (str(national_id),))
            else:
                cursor.execute("""
                    SELECT customer_id, national_id::text as national_id
                    FROM customers
                    WHERE customer_id = %s
                """, (customer_id,))

            result = cursor.fetchone()
            if not result:
                raise HTTPException(status_code=404, detail="Customer not found")

            return {
                "customer_id": result['customer_id'],
                "national_id": result['national_id'],
                "found": True
            }

    except psycopg2.Error as e:
        logger.error(f"Failed to search customers: {e}")
        raise DatabaseError(f"Failed to search customers: {e}")
    finally:
        conn.close()

@app.get("/actions")
async def get_supported_actions():
    """
    Get all supported actions from the database
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT action_name, description
                FROM actions
                ORDER BY action_name
            """)
            actions = cursor.fetchall()

            return {
                "actions": [
                    {
                        "action_name": action['action_name'],
                        "description": action['description']
                    } for action in actions
                ],
                "count": len(actions)
            }

    except psycopg2.Error as e:
        logger.error(f"Failed to get actions: {e}")
        raise DatabaseError(f"Failed to get actions: {e}")
    finally:
        conn.close()

@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring
    Verifies database connectivity and system status
    """
    try:
        # Test database connection and verify tables
        conn = get_db_connection()
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            # Check existing tables
            cursor.execute("SELECT COUNT(*) as count FROM customers")
            customer_count = cursor.fetchone()['count']
            cursor.execute("SELECT COUNT(*) as count FROM actions")
            action_count = cursor.fetchone()['count']
        conn.close()

        return {
            "status": "healthy",
            "timestamp": datetime.utcnow(),
            "database": "connected",
            "customers_count": customer_count,
            "actions_count": action_count,
            "job_manager": "running" if hasattr(job_manager, 'is_running') and job_manager.is_running else "initialized"
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )

@app.get("/supported-actions")
async def get_supported_actions_endpoint():
    """
    Get list of supported actions for execution
    """
    try:
        from services.action_executor import get_supported_actions
        actions = get_supported_actions()
        return {
            "supported_actions": actions,
            "count": len(actions)
        }
    except Exception as e:
        logger.error(f"Failed to get supported actions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get supported actions: {e}")

@app.get("/jobs")
async def get_all_jobs(
    status: Optional[str] = None,
    limit: Optional[int] = 50,
    offset: Optional[int] = 0
):
    """
    Get all jobs with optional filtering
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Base query
            query = """
                SELECT job_id, filename, file_path, status, progress,
                       error_message, metadata, created_at, started_at, completed_at
                FROM jobs
            """
            params = []

            # Add status filter if provided
            if status and status != 'all':
                query += " WHERE status = %s"
                params.append(status)

            # Add ordering and pagination
            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cursor.execute(query, params)
            jobs = cursor.fetchall()

            # Convert to list of dicts for JSON serialization
            jobs_list = []
            for job in jobs:
                job_dict = dict(job)
                # Convert datetime objects to ISO strings
                for field in ['created_at', 'started_at', 'completed_at']:
                    if job_dict[field]:
                        job_dict[field] = job_dict[field].isoformat()
                jobs_list.append(job_dict)

            return {
                "jobs": jobs_list,
                "total": len(jobs_list),
                "limit": limit,
                "offset": offset
            }

    except psycopg2.Error as e:
        logger.error(f"Failed to get jobs: {e}")
        raise DatabaseError(f"Failed to get jobs: {e}")
    finally:
        conn.close()

@app.get("/jobs/{job_id}")
async def get_job_details(job_id: str):
    """
    Get detailed information about a specific job
    """
    try:
        conn = get_db_connection()
        with conn.cursor() as cursor:
            # Get job details
            cursor.execute("""
                SELECT job_id, filename, file_path, status, progress,
                       error_message, metadata, created_at, started_at, completed_at
                FROM jobs
                WHERE job_id = %s
            """, (job_id,))
            job = cursor.fetchone()

            if not job:
                raise HTTPException(status_code=404, detail="Job not found")

            # Convert to dict and handle datetime serialization
            job_dict = dict(job)
            for field in ['created_at', 'started_at', 'completed_at']:
                if job_dict[field]:
                    job_dict[field] = job_dict[field].isoformat()

            # Get associated review items
            cursor.execute("""
                SELECT id, national_id, customer_id, original_action,
                       matched_action, confidence, review_reason, status,
                       priority, created_at
                FROM review_queue
                WHERE job_id = %s
                ORDER BY priority ASC, created_at ASC
            """, (job_id,))
            review_items = cursor.fetchall()

            # Convert review items to list of dicts
            review_items_list = []
            for item in review_items:
                item_dict = dict(item)
                item_dict['id'] = str(item_dict['id'])
                if item_dict['created_at']:
                    item_dict['created_at'] = item_dict['created_at'].isoformat()
                review_items_list.append(item_dict)

            # Get processing logs
            cursor.execute("""
                SELECT agent_name, message, log_level, created_at
                FROM processing_logs
                WHERE job_id = %s
                ORDER BY created_at DESC
                LIMIT 20
            """, (job_id,))
            logs = cursor.fetchall()

            # Convert logs to list of dicts
            logs_list = []
            for log in logs:
                log_dict = dict(log)
                if log_dict['created_at']:
                    log_dict['created_at'] = log_dict['created_at'].isoformat()
                logs_list.append(log_dict)

            # Get action execution results
            executed_actions, failed_actions, pending_actions, action_execution_summary = await get_action_execution_results(
                job_id, job_dict['metadata']
            )

            return {
                "job": job_dict,
                "review_items": review_items_list,
                "logs": logs_list,
                "executed_actions": [action.dict() for action in executed_actions],
                "failed_actions": [action.dict() for action in failed_actions],
                "pending_actions": pending_actions,
                "action_execution_summary": action_execution_summary
            }

    except HTTPException:
        raise
    except psycopg2.Error as e:
        logger.error(f"Failed to get job details: {e}")
        raise DatabaseError(f"Failed to get job details: {e}")
    finally:
        conn.close()

# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )

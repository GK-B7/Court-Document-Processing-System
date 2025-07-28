"""
Database Models and Connection Management
Handles PostgreSQL database operations for the Document Processing API
Works with existing customers and actions tables + supporting tables
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from contextlib import contextmanager
from datetime import datetime
import json

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from config import settings
from exceptions import DatabaseError

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Database Connection Manager
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Database connection manager with connection pooling
    """
    
    def __init__(self):
        """Initialize database manager with connection pool"""
        self.pool = None
        self._initialize_connection_pool()
    
    def _initialize_connection_pool(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = ThreadedConnectionPool(
                minconn=getattr(settings, 'DATABASE_MIN_CONNECTIONS', 5),
                maxconn=getattr(settings, 'DATABASE_MAX_CONNECTIONS', 20),
                host=settings.DATABASE_HOST,
                port=settings.DATABASE_PORT,
                database=settings.DATABASE_NAME,
                user=settings.DATABASE_USER,
                password=settings.DATABASE_PASSWORD,
                cursor_factory=RealDictCursor
            )
            logger.info("Database connection pool initialized successfully")
        except Exception as exc:
            logger.error(f"Failed to initialize database connection pool: {exc}")
            raise DatabaseError(f"Database initialization failed: {exc}") from exc
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Automatically handles connection cleanup and error handling
        """
        conn = None
        try:
            conn = self.pool.getconn()
            if conn:
                yield conn
            else:
                raise DatabaseError("Failed to get database connection from pool")
        except Exception as e:
            if conn:
                conn.rollback()
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if conn:
                self.pool.putconn(conn)
    
    def execute_query(
        self, 
        query: str, 
        params: Optional[tuple] = None, 
        fetch: bool = False
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Execute a database query with proper error handling
        """
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    
                    if fetch:
                        return cursor.fetchall()
                    else:
                        conn.commit()
                        return None
        except Exception as exc:
            logger.error(f"Query execution failed: {exc}")
            raise DatabaseError(f"Database operation failed: {exc}") from exc

# Global database manager instance
db_manager = DatabaseManager()

# ─────────────────────────────────────────────────────────────────────────────
# Database Models (Updated for correct column names)
# ─────────────────────────────────────────────────────────────────────────────

class CustomerModel:
    """Model for customer database operations (works with existing table)"""
    
    @staticmethod
    def get_customer_by_id(customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Get customer by customer_id
        """
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT customer_id, national_id
                        FROM customers 
                        WHERE customer_id::text = %s
                    """, (str(customer_id),))
                    
                    customer = cursor.fetchone()
                    return dict(customer) if customer else None
                
        except psycopg2.Error as e:
            logger.error(f"Failed to get customer by ID {customer_id}: {e}")
            return None

    @staticmethod
    def get_customer_by_national_id(national_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by national ID from existing customers table"""
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Work with existing customers table structure
                    cursor.execute("""
                        SELECT 
                            customer_id,
                            national_id::text as national_id
                        FROM customers 
                        WHERE national_id::text = %s
                    """, (str(national_id),))
                    
                    result = cursor.fetchone()
                    if result:
                        return {
                            'id': result['customer_id'],  # Use customer_id as id
                            'customer_id': result['customer_id'],
                            'national_id': result['national_id'],
                            'name': result['customer_id'],  # Use customer_id as display name
                            'email': None,
                            'phone': None,
                            'address': None,
                            'status': 'active',  # Default status
                            'account_balance': 1000.0,  # Default balance
                            'created_at': None,
                            'updated_at': None
                        }
                    return None
                    
        except Exception as exc:
            logger.error(f"Error getting customer by national_id {national_id}: {exc}")
            return None
    
    @staticmethod
    def batch_get_customers_by_national_ids(national_ids: List[str]) -> Dict[str, str]:
        """Batch get customers by national IDs from existing table"""
        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("""
                        SELECT 
                            national_id::text as national_id,
                            customer_id
                        FROM customers 
                        WHERE national_id::text = ANY(%s)
                    """, (national_ids,))
                    
                    results = cursor.fetchall()
                    return {
                        str(row['national_id']): row['customer_id'] 
                        for row in results
                    }
                    
        except Exception as exc:
            logger.error(f"Error batch getting customers: {exc}")
            return {}

class JobModel:
    """Model for job database operations (uses supporting tables)"""
    
    @staticmethod
    def create_job(job_id: str, filename: str, file_path: str) -> str:
        """Create a new job record"""
        try:
            db_manager.execute_query(
                """
                INSERT INTO jobs (job_id, filename, file_path, status, created_at)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (job_id, filename, file_path, 'pending', datetime.utcnow())
            )
            return job_id
        except Exception as exc:
            logger.error(f"Failed to create job: {exc}")
            raise DatabaseError(f"Failed to create job: {exc}") from exc
    
    @staticmethod
    def update_job_status(
        job_id: str, 
        status: str, 
        progress: Optional[float] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update job status and progress"""
        try:
            update_fields = ["status = %s"]
            params = [status]
            
            if progress is not None:
                update_fields.append("progress = %s")
                params.append(progress)
            
            if error_message is not None:
                update_fields.append("error_message = %s")
                params.append(error_message)
            
            if metadata is not None:
                update_fields.append("metadata = %s")
                params.append(json.dumps(metadata))
            
            # Add timestamps based on status
            if status == 'processing':
                update_fields.append("started_at = %s")
                params.append(datetime.utcnow())
            elif status in ['completed', 'failed']:
                update_fields.append("completed_at = %s")
                params.append(datetime.utcnow())
            
            params.append(job_id)
            
            db_manager.execute_query(
                f"""
                UPDATE jobs 
                SET {', '.join(update_fields)}
                WHERE job_id = %s
                """,
                tuple(params)
            )
        except Exception as exc:
            logger.error(f"Failed to update job status: {exc}")
            raise DatabaseError(f"Failed to update job status: {exc}") from exc
    
    @staticmethod
    def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID"""
        try:
            result = db_manager.execute_query(
                """
                SELECT job_id, filename, file_path, status, progress, 
                       error_message, metadata, created_at, started_at, completed_at
                FROM jobs 
                WHERE job_id = %s
                """,
                (job_id,),
                fetch=True
            )
            return result[0] if result else None
        except Exception as exc:
            logger.error(f"Failed to get job: {exc}")
            return None

class ReviewQueueModel:
    """Model for review queue operations (fixed column names)"""
    
    @staticmethod
    def create_review_item(
        job_id: str,
        national_id: str,
        customer_id: Optional[str],
        original_action: str,  # Changed from extracted_action
        matched_action: Optional[str] = None,
        confidence: Optional[float] = None,
        page_number: Optional[int] = None,
        context: Optional[str] = None,
        review_reason: Optional[str] = None,
        priority: int = 5
    ) -> str:
        """Create a review queue item with correct column names"""
        try:
            result = db_manager.execute_query(
                """
                INSERT INTO review_queue (
                    job_id, national_id, customer_id, original_action, 
                    matched_action, confidence, page_number, context, 
                    review_reason, priority, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    job_id, national_id, customer_id, original_action,
                    matched_action, confidence, page_number, context,
                    review_reason, priority, datetime.utcnow()
                ),
                fetch=True
            )
            return str(result[0]['id']) if result else None
        except Exception as exc:
            logger.error(f"Failed to create review item: {exc}")
            raise DatabaseError(f"Failed to create review item: {exc}") from exc
    
    @staticmethod
    def get_pending_reviews() -> List[Dict[str, Any]]:
        """Get all pending review items"""
        try:
            result = db_manager.execute_query(
                """
                SELECT id, job_id, national_id, customer_id, original_action,
                       matched_action, confidence, page_number, context,
                       review_reason, priority, created_at
                FROM review_queue 
                WHERE status = 'pending'
                ORDER BY priority ASC, created_at ASC
                """,
                fetch=True
            )
            return result or []
        except Exception as exc:
            logger.error(f"Failed to get pending reviews: {exc}")
            return []
    
    @staticmethod
    def update_review_decision(
        review_id: str,
        decision: str,
        reviewed_by: str,
        notes: Optional[str] = None
    ):
        """Update review decision"""
        try:
            db_manager.execute_query(
                """
                UPDATE review_queue 
                SET status = 'reviewed',
                    review_decision = %s,
                    reviewed_by = %s,
                    review_notes = %s,
                    reviewed_at = %s
                WHERE id = %s
                """,
                (decision, reviewed_by, notes, datetime.utcnow(), review_id)
            )
        except Exception as exc:
            logger.error(f"Failed to update review decision: {exc}")
            raise DatabaseError(f"Failed to update review decision: {exc}") from exc

class ProcessingLogModel:
    """Model for processing logs (fixed column names)"""
    
    @staticmethod
    def create_log_entry(
        job_id: str, 
        agent_name: str, 
        data: Dict[str, Any],
        log_level: str = 'INFO',
        message: str = ''
    ):
        """Create a log entry with correct column names"""
        try:
            db_manager.execute_query(
                """
                INSERT INTO processing_logs (job_id, agent_name, log_level, message, data, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    job_id, 
                    agent_name, 
                    log_level, 
                    message or f"{agent_name} processing", 
                    json.dumps(data) if isinstance(data, dict) else data,
                    datetime.utcnow()
                )
            )
        except Exception as exc:
            logger.error(f"Failed to create log entry: {exc}")
            raise DatabaseError(f"Failed to create log entry: {exc}") from exc
    
    @staticmethod
    def get_logs_by_job_id(job_id: str) -> List[Dict[str, Any]]:
        """Get all logs for a specific job"""
        try:
            result = db_manager.execute_query(
                """
                SELECT id, job_id, agent_name, log_level, message, data, created_at
                FROM processing_logs 
                WHERE job_id = %s
                ORDER BY created_at ASC
                """,
                (job_id,),
                fetch=True
            )
            return result or []
        except Exception as exc:
            logger.error(f"Failed to get logs for job {job_id}: {exc}")
            return []

class ActionModel:
    """Model for actions (works with existing actions table)"""
    
    @staticmethod
    def get_all_actions() -> List[Dict[str, Any]]:
        """Get all actions from existing actions table"""
        try:
            result = db_manager.execute_query(
                """
                SELECT action_name, description 
                FROM actions 
                ORDER BY action_name
                """,
                fetch=True
            )
            return result or []
        except Exception as exc:
            logger.error(f"Failed to get actions: {exc}")
            return []
    
    @staticmethod
    def is_valid_action(action_name: str) -> bool:
        """Check if an action exists in the actions table"""
        try:
            result = db_manager.execute_query(
                """
                SELECT 1 FROM actions WHERE action_name = %s
                """,
                (action_name,),
                fetch=True
            )
            return len(result) > 0 if result else False
        except Exception as exc:
            logger.error(f"Failed to validate action {action_name}: {exc}")
            return False

# ─────────────────────────────────────────────────────────────────────────────
# Database Health and Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def test_database_connection() -> bool:
    """Test database connectivity"""
    try:
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT 1")
                return True
    except Exception as exc:
        logger.error(f"Database connection test failed: {exc}")
        return False

def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        stats = {}
        with db_manager.get_connection() as conn:
            with conn.cursor() as cursor:
                # Get table counts
                cursor.execute("SELECT COUNT(*) as count FROM customers")
                stats['customers_count'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM actions")
                stats['actions_count'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM jobs")
                stats['jobs_count'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM review_queue WHERE status = 'pending'")
                stats['pending_reviews'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM processing_logs")
                stats['log_entries'] = cursor.fetchone()['count']
                
        return stats
    except Exception as exc:
        logger.error(f"Failed to get database stats: {exc}")
        return {'error': str(exc)}

def cleanup_old_records(days_to_keep: int = 30) -> int:
    """Clean up old completed jobs and logs"""
    try:
        result = db_manager.execute_query(
            """
            WITH deleted_jobs AS (
                DELETE FROM jobs 
                WHERE status IN ('completed', 'failed') 
                AND created_at < NOW() - INTERVAL '%s days'
                RETURNING job_id
            )
            SELECT COUNT(*) as deleted_count FROM deleted_jobs
            """,
            (days_to_keep,),
            fetch=True
        )
        return result[0]['deleted_count'] if result else 0
    except Exception as exc:
        logger.error(f"Failed to cleanup old records: {exc}")
        return 0

# Export all models and functions
__all__ = [
    'DatabaseManager',
    'db_manager',
    'CustomerModel',
    'JobModel', 
    'ReviewQueueModel',
    'ProcessingLogModel',
    'ActionModel',
    'test_database_connection',
    'get_database_stats',
    'cleanup_old_records',
    'DatabaseError'
]

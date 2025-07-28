"""
Custom exception classes for the Document Processing API
Provides specific exception types for different error scenarios with proper error codes
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

# Configure logger
logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """
    Enumeration of error codes for consistent error handling
    Maps to HTTP status codes where applicable
    """
    # General errors (500-599)
    INTERNAL_SERVER_ERROR = 500
    SERVICE_UNAVAILABLE = 503
    
    # Client errors (400-499)
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    VALIDATION_ERROR = 422
    TOO_MANY_REQUESTS = 429
    
    # Database errors (600-699)
    DATABASE_CONNECTION_ERROR = 600
    DATABASE_QUERY_ERROR = 601
    DATABASE_TRANSACTION_ERROR = 602
    DATABASE_INTEGRITY_ERROR = 603
    
    # File processing errors (700-799)
    FILE_NOT_FOUND = 700
    FILE_INVALID_FORMAT = 701
    FILE_TOO_LARGE = 702
    FILE_CORRUPTED = 703
    PDF_EXTRACTION_ERROR = 704
    OCR_ERROR = 705
    
    # LLM service errors (800-899)
    LLM_API_ERROR = 800
    LLM_TIMEOUT = 801
    LLM_QUOTA_EXCEEDED = 802
    LLM_INVALID_RESPONSE = 803
    EMBEDDING_ERROR = 804
    
    # Agent workflow errors (900-999)
    AGENT_EXECUTION_ERROR = 900
    WORKFLOW_TIMEOUT = 901
    STATE_VALIDATION_ERROR = 902
    AGENT_COMMUNICATION_ERROR = 903
    
    # Vector store errors (1000-1099)
    VECTOR_STORE_ERROR = 1000
    SIMILARITY_SEARCH_ERROR = 1001
    EMBEDDING_STORAGE_ERROR = 1002
    
    # Background job errors (1100-1199)
    JOB_QUEUE_FULL = 1100
    JOB_TIMEOUT = 1101
    JOB_CANCELLED = 1102
    JOB_NOT_FOUND = 1103


class BaseDocumentProcessingError(Exception):
    """
    Base exception class for all document processing errors
    Provides common functionality for error handling and logging
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: ErrorCode = ErrorCode.INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize base exception
        
        Args:
            message: Human-readable error message
            error_code: Specific error code for this exception
            details: Additional error details dictionary
            original_exception: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.original_exception = original_exception
        
        # Log the error
        if original_exception:
            logger.error(f"{self.__class__.__name__}: {message}", exc_info=original_exception)
        else:
            logger.error(f"{self.__class__.__name__}: {message}")
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for API responses
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code.value,
            'message': self.message,
            'details': self.details,
            'original_error': str(self.original_exception) if self.original_exception else None
        }
    
    @property
    def http_status_code(self) -> int:
        """
        Get appropriate HTTP status code for this exception
        
        Returns:
            HTTP status code
        """
        # Map error codes to HTTP status codes
        if 400 <= self.error_code.value < 500:
            return self.error_code.value
        elif 500 <= self.error_code.value < 600:
            return self.error_code.value
        else:
            # Custom error codes default to 500
            return 500


class ValidationError(BaseDocumentProcessingError):
    """
    Exception raised for data validation errors
    Used when input data doesn't meet required criteria
    """
    
    def __init__(
        self, 
        message: str, 
        field_name: Optional[str] = None,
        invalid_value: Optional[Any] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error
        
        Args:
            message: Error message
            field_name: Name of the field that failed validation
            invalid_value: The invalid value that caused the error
            details: Additional validation details
        """
        validation_details = details or {}
        if field_name:
            validation_details['field_name'] = field_name
        if invalid_value is not None:
            validation_details['invalid_value'] = str(invalid_value)
        
        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=validation_details
        )


class DatabaseError(BaseDocumentProcessingError):
    """
    Exception raised for database-related errors
    Includes connection issues, query failures, and transaction problems
    """
    
    def __init__(
        self, 
        message: str, 
        operation: Optional[str] = None,
        table_name: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize database error
        
        Args:
            message: Error message
            operation: Database operation that failed (e.g., 'INSERT', 'SELECT')
            table_name: Name of the table involved in the operation
            original_exception: Original database exception
        """
        details = {}
        if operation:
            details['operation'] = operation
        if table_name:
            details['table_name'] = table_name
        
        # Determine specific error code based on exception type
        error_code = ErrorCode.DATABASE_QUERY_ERROR
        if original_exception:
            exception_str = str(original_exception).lower()
            if 'connection' in exception_str or 'connect' in exception_str:
                error_code = ErrorCode.DATABASE_CONNECTION_ERROR
            elif 'transaction' in exception_str or 'rollback' in exception_str:
                error_code = ErrorCode.DATABASE_TRANSACTION_ERROR
            elif 'constraint' in exception_str or 'unique' in exception_str:
                error_code = ErrorCode.DATABASE_INTEGRITY_ERROR
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class PDFError(BaseDocumentProcessingError):
    """
    Exception raised for PDF processing errors
    Covers text extraction, OCR failures, and file format issues
    """
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        page_number: Optional[int] = None,
        operation: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize PDF error
        
        Args:
            message: Error message
            file_path: Path to the PDF file that caused the error
            page_number: Page number where the error occurred
            operation: PDF operation that failed (e.g., 'text_extraction', 'ocr')
            original_exception: Original PDF processing exception
        """
        details = {}
        if file_path:
            details['file_path'] = file_path
        if page_number is not None:
            details['page_number'] = page_number
        if operation:
            details['operation'] = operation
        
        # Determine specific error code based on operation
        error_code = ErrorCode.PDF_EXTRACTION_ERROR
        if operation == 'ocr':
            error_code = ErrorCode.OCR_ERROR
        elif 'corrupted' in message.lower() or 'invalid' in message.lower():
            error_code = ErrorCode.FILE_CORRUPTED
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class LLMError(BaseDocumentProcessingError):
    """
    Exception raised for Large Language Model API errors
    Includes API failures, timeout errors, and quota exceeded scenarios
    """
    
    def __init__(
        self, 
        message: str, 
        model_name: Optional[str] = None,
        api_call_type: Optional[str] = None,
        tokens_used: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize LLM error
        
        Args:
            message: Error message
            model_name: Name of the LLM model that failed
            api_call_type: Type of API call (e.g., 'completion', 'embedding')
            tokens_used: Number of tokens used in the failed request
            original_exception: Original LLM API exception
        """
        details = {}
        if model_name:
            details['model_name'] = model_name
        if api_call_type:
            details['api_call_type'] = api_call_type
        if tokens_used is not None:
            details['tokens_used'] = tokens_used
        
        # Determine specific error code based on error message
        error_code = ErrorCode.LLM_API_ERROR
        message_lower = message.lower()
        if 'timeout' in message_lower:
            error_code = ErrorCode.LLM_TIMEOUT
        elif 'quota' in message_lower or 'limit' in message_lower:
            error_code = ErrorCode.LLM_QUOTA_EXCEEDED
        elif 'invalid' in message_lower or 'malformed' in message_lower:
            error_code = ErrorCode.LLM_INVALID_RESPONSE
        elif api_call_type == 'embedding':
            error_code = ErrorCode.EMBEDDING_ERROR
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class AgentError(BaseDocumentProcessingError):
    """
    Exception raised for agent workflow errors
    Covers agent execution failures, communication issues, and workflow problems
    """
    
    def __init__(
        self, 
        message: str, 
        agent_name: Optional[str] = None,
        job_id: Optional[str] = None,
        state_data: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize agent error
        
        Args:
            message: Error message
            agent_name: Name of the agent that failed
            job_id: Job ID associated with the failure
            state_data: Current state data when the error occurred
            original_exception: Original agent exception
        """
        details = {}
        if agent_name:
            details['agent_name'] = agent_name
        if job_id:
            details['job_id'] = job_id
        if state_data:
            details['state_data'] = state_data
        
        # Determine specific error code based on context
        error_code = ErrorCode.AGENT_EXECUTION_ERROR
        message_lower = message.lower()
        if 'timeout' in message_lower:
            error_code = ErrorCode.WORKFLOW_TIMEOUT
        elif 'state' in message_lower or 'validation' in message_lower:
            error_code = ErrorCode.STATE_VALIDATION_ERROR
        elif 'communication' in message_lower or 'message' in message_lower:
            error_code = ErrorCode.AGENT_COMMUNICATION_ERROR
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class VectorStoreError(BaseDocumentProcessingError):
    """
    Exception raised for vector store and embedding errors
    Covers ChromaDB operations and similarity search failures
    """
    
    def __init__(
        self, 
        message: str, 
        collection_name: Optional[str] = None,
        operation: Optional[str] = None,
        query_text: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize vector store error
        
        Args:
            message: Error message
            collection_name: Name of the ChromaDB collection
            operation: Vector store operation that failed
            query_text: Query text that caused the error
            original_exception: Original vector store exception
        """
        details = {}
        if collection_name:
            details['collection_name'] = collection_name
        if operation:
            details['operation'] = operation
        if query_text:
            details['query_text'] = query_text[:100]  # Truncate long texts
        
        # Determine specific error code based on operation
        error_code = ErrorCode.VECTOR_STORE_ERROR
        if operation == 'similarity_search':
            error_code = ErrorCode.SIMILARITY_SEARCH_ERROR
        elif operation in ['add', 'update', 'delete']:
            error_code = ErrorCode.EMBEDDING_STORAGE_ERROR
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


class BackgroundJobError(BaseDocumentProcessingError):
    """
    Exception raised for background job processing errors
    Covers job queue issues, timeouts, and execution failures
    """
    
    def __init__(
        self, 
        message: str, 
        job_id: Optional[str] = None,
        job_type: Optional[str] = None,
        queue_size: Optional[int] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize background job error
        
        Args:
            message: Error message
            job_id: ID of the job that failed
            job_type: Type of job (e.g., 'document_processing')
            queue_size: Current queue size when error occurred
            original_exception: Original job processing exception
        """
        details = {}
        if job_id:
            details['job_id'] = job_id
        if job_type:
            details['job_type'] = job_type
        if queue_size is not None:
            details['queue_size'] = queue_size
        
        # Determine specific error code based on message
        error_code = ErrorCode.JOB_QUEUE_FULL
        message_lower = message.lower()
        if 'timeout' in message_lower:
            error_code = ErrorCode.JOB_TIMEOUT
        elif 'cancelled' in message_lower:
            error_code = ErrorCode.JOB_CANCELLED
        elif 'not found' in message_lower:
            error_code = ErrorCode.JOB_NOT_FOUND
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )

class WorkflowError(Exception):
    """Workflow execution errors"""
    
    def __init__(
        self, 
        message: str, 
        job_id: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.job_id = job_id
        self.original_exception = original_exception
        super().__init__(message)

class FileError(BaseDocumentProcessingError):
    """
    Exception raised for general file processing errors
    Covers file access, format validation, and size issues
    """
    
    def __init__(
        self, 
        message: str, 
        file_path: Optional[str] = None,
        file_size: Optional[int] = None,
        max_size: Optional[int] = None,
        file_type: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize file error
        
        Args:
            message: Error message
            file_path: Path to the problematic file
            file_size: Size of the file in bytes
            max_size: Maximum allowed file size
            file_type: Detected or expected file type
            original_exception: Original file operation exception
        """
        details = {}
        if file_path:
            details['file_path'] = file_path
        if file_size is not None:
            details['file_size'] = file_size
        if max_size is not None:
            details['max_size'] = max_size
        if file_type:
            details['file_type'] = file_type
        
        # Determine specific error code based on context
        error_code = ErrorCode.FILE_NOT_FOUND
        message_lower = message.lower()
        if 'too large' in message_lower or 'size' in message_lower:
            error_code = ErrorCode.FILE_TOO_LARGE
        elif 'format' in message_lower or 'type' in message_lower:
            error_code = ErrorCode.FILE_INVALID_FORMAT
        elif 'corrupted' in message_lower or 'damaged' in message_lower:
            error_code = ErrorCode.FILE_CORRUPTED
        
        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            original_exception=original_exception
        )


# Convenience functions for raising common exceptions
def raise_validation_error(message: str, field_name: str = None, invalid_value: Any = None) -> None:
    """Raise a validation error with consistent formatting"""
    raise ValidationError(message, field_name=field_name, invalid_value=invalid_value)


def raise_database_error(message: str, operation: str = None, original_exception: Exception = None) -> None:
    """Raise a database error with consistent formatting"""
    raise DatabaseError(message, operation=operation, original_exception=original_exception)


def raise_pdf_error(message: str, file_path: str = None, page_number: int = None, operation: str = None) -> None:
    """Raise a PDF processing error with consistent formatting"""
    raise PDFError(message, file_path=file_path, page_number=page_number, operation=operation)


def raise_llm_error(message: str, model_name: str = None, api_call_type: str = None) -> None:
    """Raise an LLM API error with consistent formatting"""
    raise LLMError(message, model_name=model_name, api_call_type=api_call_type)


def raise_agent_error(message: str, agent_name: str = None, job_id: str = None) -> None:
    """Raise an agent workflow error with consistent formatting"""
    raise AgentError(message, agent_name=agent_name, job_id=job_id)


def raise_vector_store_error(message: str, operation: str = None, collection_name: str = None) -> None:
    """Raise a vector store error with consistent formatting"""
    raise VectorStoreError(message, operation=operation, collection_name=collection_name)


def raise_job_error(message: str, job_id: str = None, job_type: str = None) -> None:
    """Raise a background job error with consistent formatting"""
    raise BackgroundJobError(message, job_id=job_id, job_type=job_type)


def raise_file_error(message: str, file_path: str = None, file_size: int = None) -> None:
    """Raise a file processing error with consistent formatting"""
    raise FileError(message, file_path=file_path, file_size=file_size)


# Exception handler decorator for automatic error logging and conversion
def handle_exceptions(logger_name: str = __name__):
    """
    Decorator for automatic exception handling and logging
    
    Args:
        logger_name: Name of the logger to use for error logging
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except BaseDocumentProcessingError:
                # Re-raise our custom exceptions as-is
                raise
            except Exception as e:
                # Convert unknown exceptions to our base exception
                func_logger = logging.getLogger(logger_name)
                func_logger.error(f"Unhandled exception in {func.__name__}: {e}", exc_info=True)
                raise BaseDocumentProcessingError(
                    message=f"Unexpected error in {func.__name__}: {str(e)}",
                    original_exception=e
                )
        return wrapper
    return decorator

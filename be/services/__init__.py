"""
Services Module
──────────────
Centralized service layer for the Document Processing API providing clean,
reusable services for core business operations.

Available Services:
• PDFService - PDF text extraction and OCR operations
• LLMService - Large Language Model operations and text processing  
• CustomerService - Customer lookup, validation, and business rules

Each service provides:
• Clean, documented APIs
• Comprehensive error handling
• Performance monitoring and metrics
• Caching and optimization
• Configuration management
• Statistics and reporting
"""

# Import all service classes
from .pdf_service import (
    PDFService,
    PageContent,
    DocumentInfo,
    ProcessingResult,
    pdf_service,
    extract_text_from_pdf,
    extract_text_from_page,
    validate_pdf_file,
    get_pdf_service_stats
)

from .llm_service import (
    LLMService,
    LLMResponse,
    ExtractionRequest,
    ExtractionResult,
    TokenUsage,
    LLMOperation,
    llm_service,
    extract_id_action_pairs,
    generate_text_embeddings,
    chat_with_gpt,
    count_text_tokens,
    get_llm_service_stats
)

from .customer_service import (
    CustomerService,
    CustomerInfo,
    ActionValidation,
    BatchLookupResult,
    CustomerStatus,
    ValidationResult,
    customer_service,
    get_customer_info,
    batch_lookup_customers,
    validate_customer_action,
    get_customer_service_stats
)

# Version information
__version__ = '1.0.0'
__author__ = 'Document Processing Team'
__description__ = 'Comprehensive service layer for document processing operations'

# Service registry for easy access
SERVICES = {
    'pdf': pdf_service,
    'llm': llm_service,
    'customer': customer_service
}

# Export all public classes and functions
__all__ = [
    # PDF Service
    'PDFService',
    'PageContent', 
    'DocumentInfo',
    'ProcessingResult',
    'pdf_service',
    'extract_text_from_pdf',
    'extract_text_from_page',
    'validate_pdf_file',
    'get_pdf_service_stats',
    
    # LLM Service
    'LLMService',
    'LLMResponse',
    'ExtractionRequest',
    'ExtractionResult',
    'TokenUsage',
    'LLMOperation',
    'llm_service',
    'extract_id_action_pairs',
    'generate_text_embeddings',
    'chat_with_gpt',
    'count_text_tokens',
    'get_llm_service_stats',
    
    # Customer Service
    'CustomerService',
    'CustomerInfo',
    'ActionValidation',
    'BatchLookupResult',
    'CustomerStatus',
    'ValidationResult',
    'CustomerServiceError',
    'customer_service',
    'get_customer_info',
    'batch_lookup_customers',
    'validate_customer_action',
    'get_customer_service_stats',
    
    # Registry
    'SERVICES'
]

def get_service(service_name: str):
    """
    Get service instance by name
    
    Args:
        service_name: Name of the service ('pdf', 'llm', 'customer')
        
    Returns:
        Service instance
        
    Raises:
        KeyError: If service name not found
    """
    if service_name not in SERVICES:
        raise KeyError(f"Service '{service_name}' not found. Available services: {list(SERVICES.keys())}")
    
    return SERVICES[service_name]

def get_all_service_stats() -> dict:
    """
    Get statistics from all services
    
    Returns:
        Dictionary with statistics from all services
    """
    try:
        return {
            'pdf_service': get_pdf_service_stats(),
            'llm_service': get_llm_service_stats(),
            'customer_service': get_customer_service_stats(),
            'timestamp': time.time(),
            'services_available': list(SERVICES.keys())
        }
    except Exception as exc:
        return {
            'error': f"Failed to get service stats: {exc}",
            'timestamp': time.time(),
            'services_available': list(SERVICES.keys())
        }

def health_check_services() -> dict:
    """
    Perform health check on all services
    
    Returns:
        Dictionary with health status of all services
    """
    health_status = {
        'overall_status': 'healthy',
        'services': {},
        'timestamp': time.time()
    }
    
    # Check PDF service
    try:
        pdf_stats = get_pdf_service_stats()
        health_status['services']['pdf_service'] = {
            'status': 'healthy',
            'documents_processed': pdf_stats.get('documents_processed', 0),
            'uptime': 'available'
        }
    except Exception as exc:
        health_status['services']['pdf_service'] = {
            'status': 'unhealthy',
            'error': str(exc)
        }
        health_status['overall_status'] = 'degraded'
    
    # Check LLM service
    try:
        llm_stats = get_llm_service_stats()
        health_status['services']['llm_service'] = {
            'status': 'healthy',
            'total_requests': llm_stats.get('total_requests', 0),
            'success_rate': llm_stats.get('success_rate', 0),
            'uptime': 'available'
        }
    except Exception as exc:
        health_status['services']['llm_service'] = {
            'status': 'unhealthy',
            'error': str(exc)
        }
        health_status['overall_status'] = 'degraded'
    
    # Check Customer service
    try:
        customer_stats = get_customer_service_stats()
        health_status['services']['customer_service'] = {
            'status': 'healthy',
            'total_lookups': customer_stats.get('total_lookups', 0),
            'success_rate': customer_stats.get('success_rate', 0),
            'uptime': 'available'
        }
    except Exception as exc:
        health_status['services']['customer_service'] = {
            'status': 'unhealthy',
            'error': str(exc)
        }
        health_status['overall_status'] = 'degraded'
    
    # Set overall status based on individual service health
    unhealthy_services = [
        name for name, status in health_status['services'].items()
        if status.get('status') != 'healthy'
    ]
    
    if len(unhealthy_services) >= len(health_status['services']):
        health_status['overall_status'] = 'unhealthy'
    elif unhealthy_services:
        health_status['overall_status'] = 'degraded'
    
    return health_status

def reset_all_service_stats() -> dict:
    """
    Reset statistics for all services
    
    Returns:
        Dictionary with reset confirmation
    """
    reset_results = {}
    
    try:
        pdf_service.reset_stats()
        reset_results['pdf_service'] = 'reset_successful'
    except Exception as exc:
        reset_results['pdf_service'] = f'reset_failed: {exc}'
    
    try:
        llm_service.reset_stats()
        reset_results['llm_service'] = 'reset_successful'
    except Exception as exc:
        reset_results['llm_service'] = f'reset_failed: {exc}'
    
    try:
        customer_service.reset_stats()
        reset_results['customer_service'] = 'reset_successful'
    except Exception as exc:
        reset_results['customer_service'] = f'reset_failed: {exc}'
    
    return {
        'reset_results': reset_results,
        'timestamp': time.time()
    }

# Service configuration utilities
def configure_services(**kwargs):
    """
    Configure all services with provided settings
    
    Args:
        **kwargs: Configuration parameters for services
    """
    # This could be used to reconfigure services at runtime
    # For now, services use configuration from settings during initialization
    pass

def get_service_info() -> dict:
    """
    Get information about all available services
    
    Returns:
        Dictionary with service information
    """
    return {
        'services': {
            'pdf_service': {
                'description': 'PDF text extraction and OCR operations',
                'class': 'PDFService',
                'features': ['text_extraction', 'ocr_fallback', 'image_preprocessing', 'metadata_extraction'],
                'status': 'available'
            },
            'llm_service': {
                'description': 'Large Language Model operations and text processing',
                'class': 'LLMService', 
                'features': ['structured_extraction', 'embeddings', 'chat_completion', 'token_counting'],
                'status': 'available'
            },
            'customer_service': {
                'description': 'Customer lookup, validation, and business rules',
                'class': 'CustomerService',
                'features': ['customer_lookup', 'batch_operations', 'business_validation', 'caching'],
                'status': 'available'
            }
        },
        'version': __version__,
        'description': __description__,
        'total_services': len(SERVICES)
    }

# Performance monitoring utilities
class ServiceMetricsCollector:
    """Collect and aggregate metrics from all services"""
    
    def __init__(self):
        self.collection_history = []
    
    def collect_metrics(self) -> dict:
        """Collect current metrics from all services"""
        metrics = {
            'timestamp': time.time(),
            'pdf_service': get_pdf_service_stats(),
            'llm_service': get_llm_service_stats(),
            'customer_service': get_customer_service_stats()
        }
        
        # Store in history (keep last 100 collections)
        self.collection_history.append(metrics)
        if len(self.collection_history) > 100:
            self.collection_history.pop(0)
        
        return metrics
    
    def get_aggregated_metrics(self, time_window_minutes: int = 60) -> dict:
        """Get aggregated metrics over a time window"""
        current_time = time.time()
        cutoff_time = current_time - (time_window_minutes * 60)
        
        # Filter metrics within time window
        recent_metrics = [
            m for m in self.collection_history
            if m['timestamp'] >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No metrics available for the specified time window'}
        
        # Aggregate metrics
        aggregated = {
            'time_window_minutes': time_window_minutes,
            'data_points': len(recent_metrics),
            'start_time': recent_metrics[0]['timestamp'],
            'end_time': recent_metrics[-1]['timestamp'],
            'services': {}
        }
        
        # Aggregate each service's metrics
        for service_name in ['pdf_service', 'llm_service', 'customer_service']:
            service_metrics = [m[service_name] for m in recent_metrics if service_name in m]
            if service_metrics:
                aggregated['services'][service_name] = self._aggregate_service_metrics(service_metrics)
        
        return aggregated
    
    def _aggregate_service_metrics(self, metrics_list: list) -> dict:
        """Aggregate metrics for a specific service"""
        if not metrics_list:
            return {}
        
        # Get numeric metrics that can be aggregated
        numeric_keys = []
        for key, value in metrics_list[0].items():
            if isinstance(value, (int, float)):
                numeric_keys.append(key)
        
        aggregated = {}
        for key in numeric_keys:
            values = [m[key] for m in metrics_list if key in m and isinstance(m[key], (int, float))]
            if values:
                aggregated[f'{key}_min'] = min(values)
                aggregated[f'{key}_max'] = max(values)
                aggregated[f'{key}_avg'] = sum(values) / len(values)
                aggregated[f'{key}_total'] = sum(values)
        
        return aggregated

# Global metrics collector
metrics_collector = ServiceMetricsCollector()

# Additional utility imports
import time
import logging

# Configure module logger
logger = logging.getLogger(__name__)

# Log module initialization
logger.info(f"Services module initialized with {len(SERVICES)} services: {list(SERVICES.keys())}")

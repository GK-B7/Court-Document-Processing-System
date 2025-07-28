"""
Metrics and Performance Monitoring for Document Processing API
Provides comprehensive tracking of system performance and usage statistics
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Metrics Storage
# ─────────────────────────────────────────────────────────────────────────────

class MetricsCollector:
    """Thread-safe metrics collector"""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._counters = defaultdict(int)
        self._timers = defaultdict(list)
        self._histograms = defaultdict(lambda: deque(maxlen=1000))
        
    def increment(self, metric_name: str, value: int = 1):
        """Increment a counter metric"""
        with self._lock:
            self._counters[metric_name] += value
    
    def record_time(self, metric_name: str, duration: float):
        """Record a timing metric"""
        with self._lock:
            self._timers[metric_name].append(duration)
            self._histograms[metric_name].append(duration)
    
    def get_counter(self, metric_name: str) -> int:
        """Get counter value"""
        with self._lock:
            return self._counters[metric_name]
    
    def get_timer_stats(self, metric_name: str) -> Dict[str, float]:
        """Get timer statistics"""
        with self._lock:
            times = self._timers[metric_name]
            if not times:
                return {"count": 0, "avg": 0.0, "min": 0.0, "max": 0.0, "total": 0.0}
            
            return {
                "count": len(times),
                "avg": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "total": sum(times)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics"""
        with self._lock:
            metrics = {
                "counters": dict(self._counters),
                "timers": {name: self.get_timer_stats(name) for name in self._timers}
            }
            return metrics

# Global metrics collector
_metrics = MetricsCollector()

# ─────────────────────────────────────────────────────────────────────────────
# Counter Metrics
# ─────────────────────────────────────────────────────────────────────────────

def increment_document_processed(success: bool = True):
    """Increment document processed counter"""
    _metrics.increment("documents_processed_total")
    if success:
        _metrics.increment("documents_processed_success")
    else:
        _metrics.increment("documents_processed_failed")

def increment_llm_api_call():
    """Increment LLM API call counter"""
    _metrics.increment("llm_api_calls_total")

def increment_database_query():
    """Increment database query counter"""
    _metrics.increment("database_queries_total")

def increment_vector_query():
    """Increment vector store query counter"""
    _metrics.increment("vector_queries_total")

def increment_vector_search():
    """Increment vector search counter"""
    _metrics.increment("vector_searches_total")

def increment_ocr_operation():
    """Increment OCR operation counter"""
    _metrics.increment("ocr_operations_total")

def increment_pdf_processed():
    """Increment PDF processed counter"""
    _metrics.increment("pdf_processed_total")

def increment_extraction_operation():
    """Increment extraction operation counter"""
    _metrics.increment("extraction_operations_total")

def increment_validation_operation():
    """Increment validation operation counter"""
    _metrics.increment("validation_operations_total")

def increment_customer_lookup():
    """Increment customer lookup counter"""
    _metrics.increment("customer_lookups_total")

def increment_action_matching():
    """Increment action matching counter"""
    _metrics.increment("action_matching_total")

def increment_execution_operation():
    """Increment execution operation counter"""
    _metrics.increment("execution_operations_total")

# Review-related metrics
def increment_review_item_created():
    """Increment review item created counter"""
    _metrics.increment("review_items_created_total")

def increment_review_item_approved():
    """Increment review item approved counter"""
    _metrics.increment("review_items_approved_total")

def increment_review_item_rejected():
    """Increment review item rejected counter"""
    _metrics.increment("review_items_rejected_total")

def increment_human_review_required():
    """Increment human review required counter"""
    _metrics.increment("human_reviews_required_total")

def increment_auto_approved():
    """Increment auto approved counter"""
    _metrics.increment("auto_approved_total")


# ─────────────────────────────────────────────────────────────────────────────
# Timing Metrics
# ─────────────────────────────────────────────────────────────────────────────

def record_document_processing_time(duration: float):
    """Record document processing time"""
    _metrics.record_time("document_processing_duration", duration)

def record_llm_response_time(duration: float):
    """Record LLM API response time"""
    _metrics.record_time("llm_response_duration", duration)

def record_database_query_time(duration: float):
    """Record database query time"""
    _metrics.record_time("database_query_duration", duration)

def record_vector_query_time(duration: float):
    """Record vector store query time"""
    _metrics.record_time("vector_query_duration", duration)

def record_ocr_processing_time(duration: float):
    """Record OCR processing time"""
    _metrics.record_time("ocr_processing_duration", duration)

def record_pdf_processing_time(duration: float):
    """Record PDF processing time"""
    _metrics.record_time("pdf_processing_duration", duration)

def record_extraction_time(duration: float):
    """Record extraction processing time"""
    _metrics.record_time("extraction_duration", duration)

def record_validation_time(duration: float):
    """Record validation processing time"""
    _metrics.record_time("validation_duration", duration)

def record_customer_lookup_time(duration: float):
    """Record customer lookup time"""
    _metrics.record_time("customer_lookup_duration", duration)

def record_action_matching_time(duration: float):
    """Record action matching time"""
    _metrics.record_time("action_matching_duration", duration)

def record_execution_time(duration: float):
    """Record execution time"""
    _metrics.record_time("execution_duration", duration)

# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

def get_all_metrics() -> Dict[str, Any]:
    """Get all collected metrics"""
    return _metrics.get_all_metrics()

def get_counter_value(metric_name: str) -> int:
    """Get specific counter value"""
    return _metrics.get_counter(metric_name)

def get_timer_stats(metric_name: str) -> Dict[str, float]:
    """Get specific timer statistics"""
    return _metrics.get_timer_stats(metric_name)

def reset_metrics():
    """Reset all metrics (for testing)"""
    global _metrics
    _metrics = MetricsCollector()

def get_system_health() -> Dict[str, Any]:
    """Get system health metrics"""
    import psutil
    
    try:
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0,
            "timestamp": time.time()
        }
    except Exception as exc:
        logger.error(f"Failed to get system health: {exc}")
        return {
            "error": str(exc),
            "timestamp": time.time()
        }

# Context manager for timing operations
class timer:
    """Context manager for timing operations"""
    
    def __init__(self, metric_name: str):
        self.metric_name = metric_name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.perf_counter() - self.start_time
            _metrics.record_time(self.metric_name, duration)

# ─────────────────────────────────────────────────────────────────────────────
# Counter Metrics
# ─────────────────────────────────────────────────────────────────────────────

def increment_document_processed(success: bool = True):
    """Increment document processed counter"""
    _metrics.increment("documents_processed_total")
    if success:
        _metrics.increment("documents_processed_success")
    else:
        _metrics.increment("documents_processed_failed")

def increment_llm_api_call():
    """Increment LLM API call counter"""
    _metrics.increment("llm_api_calls_total")

def increment_database_query():
    """Increment database query counter"""
    _metrics.increment("database_queries_total")

def increment_vector_query():
    """Increment vector store query counter"""
    _metrics.increment("vector_queries_total")

def increment_vector_search():
    """Increment vector search counter"""
    _metrics.increment("vector_searches_total")

def increment_ocr_operation():
    """Increment OCR operation counter"""
    _metrics.increment("ocr_operations_total")

def increment_pdf_processed():
    """Increment PDF processed counter"""
    _metrics.increment("pdf_processed_total")

def increment_extraction_operation():
    """Increment extraction operation counter"""
    _metrics.increment("extraction_operations_total")

def increment_validation_operation():
    """Increment validation operation counter"""
    _metrics.increment("validation_operations_total")

def increment_customer_lookup():
    """Increment customer lookup counter"""
    _metrics.increment("customer_lookups_total")

def increment_action_matching():
    """Increment action matching counter"""
    _metrics.increment("action_matching_total")

def increment_execution_operation():
    """Increment execution operation counter"""
    _metrics.increment("execution_operations_total")


# ─────────────────────────────────────────────────────────────────────────────
# Export all functions
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Counter functions
    'increment_document_processed',
    'increment_llm_api_call',
    'increment_database_query',
    'increment_vector_query',
    'increment_vector_search',
    'increment_ocr_operation',
    'increment_pdf_processed',
    'increment_extraction_operation',
    'increment_validation_operation',
    'increment_customer_lookup',
    'increment_action_matching',
    'increment_execution_operation',
    
    # Review-related counters
    'increment_review_item_created',
    'increment_review_item_approved',
    'increment_review_item_rejected',
    'increment_human_review_required',
    'increment_auto_approved',
    
    # Timing functions
    'record_document_processing_time',
    'record_llm_response_time',
    'record_database_query_time',
    'record_vector_query_time',
    'record_ocr_processing_time',
    'record_pdf_processing_time',
    'record_extraction_time',
    'record_validation_time',
    'record_customer_lookup_time',
    'record_action_matching_time',
    'record_execution_time',
    
    # Utility functions
    'get_all_metrics',
    'get_counter_value',
    'get_timer_stats',
    'reset_metrics',
    'get_system_health',
    'timer'
]
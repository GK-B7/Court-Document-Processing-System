"""
Customer Service Module
──────────────────────
Centralized service for customer-related operations using only existing database schema:
- customers table: national_id, customer_id
- actions table: freeze_funds, release_funds only

Features:
• Customer lookup by National ID (minimal data)
• Business rule validation for 2 actions only
• Simple risk assessment without complex data
• Handles only: freeze_funds, release_funds, unknown_action
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import asyncio

from config import settings
from database import db_manager
from psycopg2.extras import RealDictCursor
from exceptions import DatabaseError, ValidationError
from metrics import (
    increment_database_query,
    record_database_query_time,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# Customer validation configuration
CUSTOMER_CONFIG = {
    'cache_ttl': getattr(settings, 'CUSTOMER_CACHE_TTL', 300),  # 5 minutes
    'batch_size': getattr(settings, 'CUSTOMER_BATCH_SIZE', 100),
    'max_concurrent_lookups': getattr(settings, 'MAX_CONCURRENT_CUSTOMER_LOOKUPS', 10)
}

# Business rules for only your 2 actions
ACTION_BUSINESS_RULES = {
    'freeze_funds': {
        'description': 'Freeze all funds in the customer account',
        'approval_required': False,
        'risk_level': 'medium'
    },
    'release_funds': {
        'description': 'Release all held or restricted funds back to the customer account',
        'approval_required': True,  # Release always needs approval
        'risk_level': 'high'
    },
    'unknown_action': {
        'description': 'Action not recognized or not supported',
        'approval_required': True,
        'risk_level': 'critical'
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Enums and Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class CustomerStatus(Enum):
    """Customer account status - simplified since we don't have status column"""
    ACTIVE = "active"  # Default status for all customers
    NOT_FOUND = "not_found"

class ValidationResult(Enum):
    """Customer validation result enumeration"""
    VALID = "valid"
    INVALID = "invalid"
    REQUIRES_REVIEW = "requires_review"

@dataclass
class CustomerInfo:
    """Minimal customer information based on your database schema"""
    customer_id: Optional[str]  # Your customer_id (CUST001, etc.)
    national_id: str
    name: str  # Will use customer_id as name since no name column
    
    # Default values since columns don't exist
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    status: str = CustomerStatus.ACTIVE.value
    account_balance: float = 1000.0  # Fixed default
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    # Processing flags
    found: bool = False
    last_activity: Optional[str] = None
    risk_score: float = 0.2  # Low default risk
    verification_status: str = "verified"  # Default verified
    account_flags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize customer data"""
        self.national_id = self.national_id.strip()
        self.name = self.name.strip() if self.name else self.customer_id or "Unknown"

@dataclass
class ActionValidation:
    """Result of customer action validation for 2 actions only"""
    is_valid: bool
    customer_info: CustomerInfo
    action: str  # freeze_funds, release_funds, or unknown_action
    validation_result: ValidationResult
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    required_approvals: List[str] = field(default_factory=list)
    business_rules_applied: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BatchLookupResult:
    """Result of batch customer lookup operation"""
    found_customers: Dict[str, CustomerInfo]
    missing_customers: Set[str]
    total_requested: int
    lookup_time: float
    success_rate: float
    errors: List[str] = field(default_factory=list)

# ─────────────────────────────────────────────────────────────────────────────
# Customer Cache
# ─────────────────────────────────────────────────────────────────────────────

class CustomerCache:
    """Simple in-memory cache for customer data"""
    
    def __init__(self, ttl: int = 300):
        """Initialize cache with TTL in seconds"""
        self.cache: Dict[str, Tuple[CustomerInfo, float]] = {}
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, national_id: str) -> Optional[CustomerInfo]:
        """Get customer from cache if not expired"""
        if national_id in self.cache:
            customer_info, timestamp = self.cache[national_id]
            if time.time() - timestamp < self.ttl:
                self.hits += 1
                return customer_info
            else:
                del self.cache[national_id]
        
        self.misses += 1
        return None
    
    def set(self, national_id: str, customer_info: CustomerInfo) -> None:
        """Cache customer information"""
        self.cache[national_id] = (customer_info, time.time())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        return {
            'cache_size': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / total_requests if total_requests > 0 else 0,
            'ttl': self.ttl
        }

# ─────────────────────────────────────────────────────────────────────────────
# Customer Service Class
# ─────────────────────────────────────────────────────────────────────────────

class CustomerService:
    """
    Customer service for lookup and validation using only your database schema
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize customer service with configuration"""
        self.config = {**CUSTOMER_CONFIG, **(config or {})}
        
        # Initialize cache
        self.cache = CustomerCache(ttl=self.config['cache_ttl'])
        
        # Initialize semaphore for concurrent operations
        self.lookup_semaphore = asyncio.Semaphore(self.config['max_concurrent_lookups'])
        
        # Statistics tracking
        self.stats = {
            'total_lookups': 0,
            'successful_lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_checks': 0,
            'total_lookup_time': 0.0,
            'batch_operations': 0
        }
        
        logger.info("Customer Service initialized with minimal schema support")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Customer Operations
    # ─────────────────────────────────────────────────────────────────────────
    
    async def get_customer(
        self, 
        national_id: str,
        use_cache: bool = True
    ) -> CustomerInfo:
        """
        Get customer information by National ID - using only existing columns
        """
        start_time = time.perf_counter()
        national_id = str(national_id).strip()
        
        try:
            # Check cache first
            if use_cache:
                cached_customer = self.cache.get(national_id)
                if cached_customer:
                    self.stats['cache_hits'] += 1
                    return cached_customer
                self.stats['cache_misses'] += 1
            
            # Lookup from database
            async with self.lookup_semaphore:
                customer_data = await self._lookup_customer_from_db(national_id)
            
            if customer_data:
                # Create CustomerInfo with only available data
                customer_info = CustomerInfo(
                    customer_id=customer_data['customer_id'],
                    national_id=customer_data['national_id'],
                    name=customer_data['customer_id'],  # Use customer_id as display name
                    found=True,
                    metadata={
                        'source': 'database',
                        'lookup_time': time.time()
                    }
                )
                
                # Cache the result
                if use_cache:
                    self.cache.set(national_id, customer_info)
                
                self.stats['successful_lookups'] += 1
                
            else:
                # Customer not found
                customer_info = CustomerInfo(
                    customer_id=None,
                    national_id=national_id,
                    name="Not Found",
                    found=False,
                    status=CustomerStatus.NOT_FOUND.value
                )
            
            # Update statistics
            lookup_time = time.perf_counter() - start_time
            self.stats['total_lookups'] += 1
            self.stats['total_lookup_time'] += lookup_time
            
            return customer_info
            
        except Exception as exc:
            logger.error(f"Customer lookup failed for {national_id}: {exc}")
            # Return not found customer instead of raising
            return CustomerInfo(
                customer_id=None,
                national_id=national_id,
                name="Error",
                found=False,
                status=CustomerStatus.NOT_FOUND.value,
                metadata={'error': str(exc)}
            )
    
    async def batch_get_customers(
        self, 
        national_ids: List[str],
        use_cache: bool = True
    ) -> BatchLookupResult:
        """
        Batch lookup multiple customers using only existing schema
        """
        start_time = time.perf_counter()
        
        if not national_ids:
            return BatchLookupResult(
                found_customers={},
                missing_customers=set(),
                total_requested=0,
                lookup_time=0.0,
                success_rate=1.0
            )
        
        try:
            # Remove duplicates and normalize
            unique_ids = list(set(str(id).strip() for id in national_ids if str(id).strip()))
            
            found_customers = {}
            missing_customers = set()
            
            # Batch lookup from database
            db_customers = await self._batch_lookup_from_db(unique_ids)
            
            # Process results
            for national_id in unique_ids:
                if national_id in db_customers:
                    customer_data = db_customers[national_id]
                    customer_info = CustomerInfo(
                        customer_id=customer_data['customer_id'],
                        national_id=customer_data['national_id'],
                        name=customer_data['customer_id'],  # Use customer_id as name
                        found=True
                    )
                    found_customers[national_id] = customer_info
                    
                    # Cache the result
                    if use_cache:
                        self.cache.set(national_id, customer_info)
                else:
                    missing_customers.add(national_id)
            
            # Calculate statistics
            lookup_time = time.perf_counter() - start_time
            success_rate = len(found_customers) / len(unique_ids) if unique_ids else 1.0
            
            # Update service statistics
            self.stats['batch_operations'] += 1
            self.stats['total_lookups'] += len(unique_ids)
            self.stats['successful_lookups'] += len(found_customers)
            self.stats['total_lookup_time'] += lookup_time
            
            return BatchLookupResult(
                found_customers=found_customers,
                missing_customers=missing_customers,
                total_requested=len(unique_ids),
                lookup_time=lookup_time,
                success_rate=success_rate
            )
            
        except Exception as exc:
            lookup_time = time.perf_counter() - start_time
            logger.error(f"Batch customer lookup failed: {exc}")
            
            return BatchLookupResult(
                found_customers={},
                missing_customers=set(national_ids),
                total_requested=len(national_ids),
                lookup_time=lookup_time,
                success_rate=0.0,
                errors=[str(exc)]
            )
    
    async def validate_customer_for_action(
        self, 
        national_id: str, 
        action: str,
        context: Optional[Dict[str, Any]] = None
    ) -> ActionValidation:
        """
        Validate customer for one of 3 cases: freeze_funds, release_funds, unknown_action
        """
        start_time = time.perf_counter()
        self.stats['validation_checks'] += 1
        
        try:
            # Get customer information
            customer_info = await self.get_customer(national_id)
            
            # Normalize action name
            action = action.lower().strip()
            
            # Only validate supported actions
            if action not in ['freeze_funds', 'release_funds', 'unknown_action']:
                action = 'unknown_action'
            
            # Get business rules for action
            business_rules = ACTION_BUSINESS_RULES.get(action, ACTION_BUSINESS_RULES['unknown_action'])
            
            # Perform validation
            validation_result = await self._validate_action_simple(
                customer_info, action, business_rules, context
            )
            
            return validation_result
            
        except Exception as exc:
            logger.error(f"Customer validation failed for {national_id}:{action}: {exc}")
            
            return ActionValidation(
                is_valid=False,
                customer_info=CustomerInfo(customer_id=None, national_id=national_id, name="Error"),
                action=action,
                validation_result=ValidationResult.INVALID,
                reasons=[f"Validation error: {exc}"],
                business_rules_applied={}
            )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _lookup_customer_from_db(self, national_id: str) -> Optional[Dict[str, Any]]:
        """Lookup single customer from database using only existing columns"""
        try:
            increment_database_query()
            start_time = time.perf_counter()
            
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Query only existing columns
                    cursor.execute("""
                        SELECT 
                            customer_id,
                            national_id::text as national_id
                        FROM customers 
                        WHERE national_id::text = %s
                    """, (str(national_id),))
                    
                    result = cursor.fetchone()
                    
                    query_time = time.perf_counter() - start_time
                    record_database_query_time(query_time)
                    
                    return dict(result) if result else None
                    
        except Exception as exc:
            logger.error(f"Database lookup failed for {national_id}: {exc}")
            return None
    
    async def _batch_lookup_from_db(self, national_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Batch lookup customers from database using only existing columns"""
        try:
            increment_database_query()
            start_time = time.perf_counter()
            
            with db_manager.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Use ANY for batch lookup
                    cursor.execute("""
                        SELECT 
                            customer_id,
                            national_id::text as national_id
                        FROM customers 
                        WHERE national_id::text = ANY(%s)
                    """, (national_ids,))
                    
                    results = cursor.fetchall()
                    
                    # Convert to dictionary
                    customers = {
                        row['national_id']: dict(row)
                        for row in results
                    }
                    
                    query_time = time.perf_counter() - start_time
                    record_database_query_time(query_time)
                    
                    return customers
                    
        except Exception as exc:
            logger.error(f"Batch database lookup failed: {exc}")
            return {}
    
    async def _validate_action_simple(
        self, 
        customer_info: CustomerInfo, 
        action: str,
        business_rules: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> ActionValidation:
        """Simple validation for only 3 action cases"""
        reasons = []
        warnings = []
        required_approvals = []
        is_valid = True
        validation_result = ValidationResult.VALID
        
        # Check if customer exists
        if not customer_info.found:
            is_valid = False
            reasons.append(f"Customer with National ID {customer_info.national_id} not found in database")
            validation_result = ValidationResult.INVALID
            
        elif action == 'freeze_funds':
            # Freeze funds validation
            validation_result = ValidationResult.VALID
            warnings.append("Funds will be frozen for this customer")
            
        elif action == 'release_funds':
            # Release funds always requires approval
            required_approvals.append("supervisor_approval")
            validation_result = ValidationResult.REQUIRES_REVIEW
            warnings.append("Release funds requires supervisor approval")
            
        elif action == 'unknown_action':
            # Unknown action always invalid
            is_valid = False
            reasons.append("Action not recognized. Only 'freeze_funds' and 'release_funds' are supported")
            validation_result = ValidationResult.INVALID
            
        else:
            # Fallback for any other action
            is_valid = False
            reasons.append(f"Unsupported action: {action}")
            validation_result = ValidationResult.INVALID
        
        # Create risk assessment
        risk_assessment = {
            'customer_risk_score': customer_info.risk_score,
            'action_risk_level': business_rules.get('risk_level', 'medium'),
            'combined_risk': 0.8 if action == 'release_funds' else 0.3,
            'requires_approval': business_rules.get('approval_required', False)
        }
        
        return ActionValidation(
            is_valid=is_valid,
            customer_info=customer_info,
            action=action,
            validation_result=validation_result,
            reasons=reasons,
            warnings=warnings,
            required_approvals=required_approvals,
            business_rules_applied=business_rules,
            risk_assessment=risk_assessment
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        cache_stats = self.cache.get_stats()
        
        total_requests = self.stats['total_lookups']
        avg_lookup_time = (
            self.stats['total_lookup_time'] / total_requests
            if total_requests > 0 else 0
        )
        
        return {
            **self.stats,
            'cache_statistics': cache_stats,
            'success_rate': (
                self.stats['successful_lookups'] / total_requests
                if total_requests > 0 else 0
            ),
            'average_lookup_time': avg_lookup_time,
            'supported_actions': ['freeze_funds', 'release_funds'],
            'database_schema': 'minimal (national_id, customer_id only)'
        }
    
    def reset_stats(self) -> None:
        """Reset service statistics"""
        self.stats = {
            'total_lookups': 0,
            'successful_lookups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'validation_checks': 0,
            'total_lookup_time': 0.0,
            'batch_operations': 0
        }
        self.cache = CustomerCache(ttl=self.config['cache_ttl'])


# ─────────────────────────────────────────────────────────────────────────────
# Global Service Instance and Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

# Global customer service instance
customer_service = CustomerService()

async def get_customer_info(national_id: str, use_cache: bool = True) -> CustomerInfo:
    """Convenience function to get customer information"""
    return await customer_service.get_customer(national_id, use_cache)

async def batch_lookup_customers(national_ids: List[str]) -> BatchLookupResult:
    """Convenience function for batch customer lookup"""
    return await customer_service.batch_get_customers(national_ids)

async def validate_customer_action(national_id: str, action: str) -> ActionValidation:
    """Convenience function to validate customer action (freeze_funds, release_funds, or unknown)"""
    return await customer_service.validate_customer_for_action(national_id, action)

def get_customer_service_stats() -> Dict[str, Any]:
    """Get customer service statistics"""
    return customer_service.get_service_stats()

def get_supported_actions() -> List[str]:
    """Get list of supported actions"""
    return ['freeze_funds', 'release_funds']

# Export main classes and functions
__all__ = [
    'CustomerService',
    'CustomerInfo',
    'ActionValidation',
    'BatchLookupResult',
    'CustomerStatus',
    'ValidationResult',
    'customer_service',
    'get_customer_info',
    'batch_lookup_customers',
    'validate_customer_action',
    'get_customer_service_stats',
    'get_supported_actions'
]

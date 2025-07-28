"""
CustomerLookupAgent
───────────────────

1. Takes validated_pairs from ValidationAgent
2. Extracts unique national IDs from validated pairs
3. Performs batch lookup against PostgreSQL customers table
4. FAILS immediately if no IDs or if any customer is missing (no review)
5. Populates state.customer_mappings and updates validated_pairs
6. Updates state.progress and logs statistics
"""

import time
import logging
from typing import List, Dict, Set, Tuple

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, ValidatedPair
from database import CustomerModel
from exceptions import DatabaseError, AgentError
from metrics import increment_database_query, record_database_query_time

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class CustomerLookupAgent(BaseAgent):
    NAME = "customer_lookup_agent"
    DESCRIPTION = "Batch lookup customers by National ID from database - fails on missing customers"
    
    def __init__(self):
        super().__init__()
        self.require_active_customer = getattr(settings, 'REQUIRE_ACTIVE_CUSTOMER', True)

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        start_time = time.perf_counter()

        # 1) No IDs extracted at all
        if not state.validated_pairs:
            error_message = (
                "No national ID numbers could be found in the document."
            )
            raise AgentError(error_message, agent_name=self.NAME, job_id=state.job_id)

        # Preserve original pairs
        if not hasattr(state, 'extracted_pairs') or not state.extracted_pairs:
            state.extracted_pairs = state.validated_pairs.copy()

        # 2) Extract unique IDs
        unique_ids = self._extract_unique_national_ids(state.validated_pairs)
        if not unique_ids:
            error_message = (
                "No valid national ID numbers were detected in the document."
            )
            raise AgentError(error_message, agent_name=self.NAME, job_id=state.job_id)

        # 3) Batch lookup
        try:
            customer_mappings = await self._batch_lookup_customers(unique_ids)
        except Exception as e:
            logger.error(f"System error during customer lookup: {e}")
            raise AgentError(
                "A system error occurred while looking up customer information. "
                "Please try again or contact technical support if the problem persists.",
                agent_name=self.NAME,
                job_id=state.job_id
            ) from e

        # 4) Detect missing customers and fail
        missing = [nid for nid, info in customer_mappings.items() if not info['found']]
        if missing:
            found = [nid for nid in unique_ids if nid not in missing]
            if len(missing) == 1:
                error_message = (
                    f"Customer with national ID '{missing[0]}' was not found"
                )
            elif len(missing) == len(unique_ids):
                error_message = (
                    f"None of the {len(missing)} customers with national ID found"
                )
            else:
                error_message = (
                    f"{len(missing)} of {len(unique_ids)} customers were not found. "
                    f"Missing: {', '.join(missing)}."
                )

            logger.error(
                "Customer mapping failed",
                extra={
                    "missing_ids": missing,
                    "found_ids": found,
                    "total_ids": len(unique_ids)
                }
            )
            raise AgentError(error_message, agent_name=self.NAME, job_id=state.job_id)

        # 5) All customers found → update pairs
        updated_pairs = self._update_pairs_with_customer_info(state.validated_pairs, customer_mappings)
        state.validated_pairs = updated_pairs
        state.customer_mappings = customer_mappings

        # No auto‐completion or review for this agent
        state.auto_completed_pairs = []
        state.auto_completed = []
        state.unmatched_national_ids = []

        # Set state for downstream
        state.processing_case = 'customers_found'
        state.case_details = {
            'message': 'All customers found successfully',
            'case_type': 'normal_processing',
            'total_customers': len(updated_pairs),
            'customer_ids': [p.customer_id for p in updated_pairs]
        }
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.75

        # Log stats
        self._log_lookup_stats(unique_ids, customer_mappings, updated_pairs, [])

        duration = time.perf_counter() - start_time
        record_database_query_time(duration)
        logger.info(
            "Customer lookup completed successfully",
            extra={
                'unique_ids': len(unique_ids),
                'customers_found': len(unique_ids),
                'duration': f"{duration:.2f}s"
            }
        )

        return state

    # ------------------------------------------------------------------
    def _extract_unique_national_ids(self, pairs: List[ValidatedPair]) -> List[str]:
        ids: Set[str] = set()
        for p in pairs:
            if p.national_id and p.national_id.strip():
                ids.add(p.national_id.strip())
        result = list(ids)
        logger.debug(f"Extracted {len(result)} unique national IDs")
        return result

    # ------------------------------------------------------------------
    async def _batch_lookup_customers(self, national_ids: List[str]) -> Dict[str, Dict]:
        increment_database_query()
        start = time.perf_counter()

        try:
            id_map = CustomerModel.batch_get_customers_by_national_ids(national_ids)
            mappings: Dict[str, Dict] = {}
            # Populate found
            for nid, cid in id_map.items():
                details = CustomerModel.get_customer_by_national_id(nid) or {}
                mappings[nid] = {
                    'customer_id': cid,
                    'name': details.get('name', ''),
                    'email': details.get('email', ''),
                    'status': details.get('status', 'unknown'),
                    'account_balance': details.get('account_balance', 0.0),
                    'found': True
                }
            # Mark missing
            for nid in national_ids:
                if nid not in mappings:
                    mappings[nid] = {
                        'customer_id': None,
                        'name': None,
                        'email': None,
                        'status': 'not_found',
                        'account_balance': 0.0,
                        'found': False
                    }
            return mappings

        except Exception as e:
            raise DatabaseError(
                f"Batch customer lookup failed: {e}",
                operation="batch_select",
                table_name="customers",
                original_exception=e
            ) from e
        finally:
            record_database_query_time(time.perf_counter() - start)

    # ------------------------------------------------------------------
    def _update_pairs_with_customer_info(
        self,
        pairs: List[ValidatedPair],
        mappings: Dict[str, Dict]
    ) -> List[ValidatedPair]:
        result: List[ValidatedPair] = []
        for p in pairs:
            info = mappings.get(p.national_id, {})
            vp = ValidatedPair(
                national_id=p.national_id,
                action=p.action,
                confidence=p.confidence,
                page_number=p.page_number,
                context=p.context,
                needs_review=False,
                validation_status=p.validation_status,
                original_text=p.original_text,
                extraction_metadata=p.extraction_metadata,
                customer_id=info.get('customer_id'),
                customer_found=info.get('found', False),
                customer_status=info.get('status', 'unknown'),
                customer_name=info.get('name', ''),
                customer_metadata={
                    'email': info.get('email', ''),
                    'account_balance': info.get('account_balance', 0.0)
                }
            )
            # Flag inactive if required
            if self.require_active_customer and vp.customer_status != 'active':
                vp.needs_review = True
                vp.validation_status = 'customer_inactive'
            result.append(vp)
        return result

    # ------------------------------------------------------------------
    def _log_lookup_stats(
        self,
        unique_ids: List[str],
        customer_mappings: Dict[str, Dict],
        updated_pairs: List[ValidatedPair],
        auto_completed_pairs: List[ValidatedPair]
    ) -> None:
        total = len(unique_ids)
        found = sum(1 for m in customer_mappings.values() if m['found'])
        missing = total - found
        logger.info(
            "Customer lookup stats",
            extra={
                'total_requested': total,
                'customers_found': found,
                'customers_missing': missing,
            }
        )

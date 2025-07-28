"""
ValidationAgent
───────────────
1. Takes extracted_pairs from ExtractionAgent
2. Validates National ID format using regex patterns
3. Deduplicates ID/Action pairs across pages
4. Calculates aggregate confidence scores
5. Filters out invalid or low-confidence pairs
6. Populates state.validated_pairs with clean data
7. Updates progress and provides validation statistics

Validation Rules:
• National ID format: 10-20 digits, configurable pattern
• Confidence threshold: configurable minimum
• Deduplication: same ID+action = keep highest confidence
• Action normalization: standardize action names
"""

from __future__ import annotations

import re
import time
import logging
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, IDActionPair, ValidatedPair
from exceptions import ValidationError, AgentError

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Validation Patterns and Constants
# ─────────────────────────────────────────────────────────────────────────────

# Default National ID pattern (10-20 digits)
DEFAULT_ID_PATTERN = r'^\d{10,20}$'

# Action normalization mapping
ACTION_NORMALIZATIONS = {
    'freeze_funds': ['freeze', 'freeze_account', 'block_funds', 'hold_funds', 'suspend_funds'],
    'release_funds': ['release', 'unfreeze', 'release_funds', 'unlock_funds', 'restore_funds'],
    'close_account': ['close', 'close_account', 'terminate_account', 'end_account'],
    'suspend_account': ['suspend', 'suspend_account', 'disable_account', 'pause_account'],
    'verify_identity': ['verify', 'verify_identity', 'confirm_identity', 'validate_identity'],
    'update_contact': ['update', 'update_contact', 'change_details', 'modify_info'],
    'flag_suspicious': ['flag', 'flag_suspicious', 'mark_suspicious', 'alert'],
    'set_limit': ['limit', 'set_limit', 'restrict', 'cap_amount'],
    'require_approval': ['approval', 'require_approval', 'manual_review', 'authorize'],
    'send_notice': ['notice', 'send_notice', 'notify', 'communicate']
}

# Minimum confidence threshold (can be overridden by settings)
MIN_CONFIDENCE = getattr(settings, 'CONFIDENCE_THRESHOLD', 0.7)


class ValidationAgent(BaseAgent):
    NAME = "validation_agent"
    DESCRIPTION = "Validate, deduplicate and normalize extracted ID/Action pairs"

    def __init__(self):
        """Initialize validation agent with patterns and thresholds"""
        # Load ID validation pattern from settings or use default
        self.id_pattern = re.compile(
            getattr(settings, 'NATIONAL_ID_PATTERN', DEFAULT_ID_PATTERN)
        )
        
        # Confidence thresholds
        self.min_confidence = MIN_CONFIDENCE
        self.review_threshold = getattr(settings, 'REVIEW_THRESHOLD', 0.8)
        
        # Build reverse action mapping for normalization
        self.action_mapping = self._build_action_mapping()

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Validate and normalize extracted ID/Action pairs
        """
        start_time = time.perf_counter()
        
        if not state.extracted_pairs:
            self.log_warning("No extracted pairs to validate")
            state.validated_pairs = []
            state.progress = 0.6  # Skip to next stage
            return state

        self.log_info("Starting validation", pairs=len(state.extracted_pairs))

        try:
            # Step 1: Validate individual pairs
            valid_pairs = self._validate_individual_pairs(state.extracted_pairs)
            
            # Step 2: Normalize actions
            normalized_pairs = self._normalize_actions(valid_pairs)
            
            # Step 3: Deduplicate pairs
            deduplicated_pairs = self._deduplicate_pairs(normalized_pairs)
            
            # Step 4: Calculate aggregate confidence scores
            final_pairs = self._calculate_aggregate_confidence(deduplicated_pairs)
            
            # Step 5: Filter by confidence threshold
            filtered_pairs = self._filter_by_confidence(final_pairs)
            
            # Convert to ValidatedPair objects
            validated_pairs = self._create_validated_pairs(filtered_pairs)

        except Exception as exc:
            raise AgentError(
                f"Validation failed: {exc}",
                agent_name=self.NAME,
                job_id=state.job_id,
                original_exception=exc
            ) from exc

        # Update state
        state.validated_pairs = validated_pairs
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.6  # 60% complete after validation

        # Log validation statistics
        self._log_validation_stats(state.extracted_pairs, validated_pairs)
        
        duration = time.perf_counter() - start_time
        self.log_info(
            "Validation completed",
            input_pairs=len(state.extracted_pairs),
            output_pairs=len(validated_pairs),
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    def _validate_individual_pairs(self, pairs: List[IDActionPair]) -> List[IDActionPair]:
        """
        Validate individual ID/Action pairs for format and basic requirements
        """
        valid_pairs = []
        validation_stats = {
            'invalid_id_format': 0,
            'missing_action': 0,
            'low_confidence': 0,
            'valid': 0
        }

        for pair in pairs:
            # Validate National ID format
            if not self.id_pattern.match(pair.national_id):
                validation_stats['invalid_id_format'] += 1
                self.log_debug(
                    "Invalid ID format",
                    id=pair.national_id,
                    page=pair.page_number
                )
                continue

            # Check for missing or empty action
            if not pair.action or pair.action.strip() == "":
                validation_stats['missing_action'] += 1
                self.log_debug(
                    "Missing action",
                    id=pair.national_id,
                    page=pair.page_number
                )
                continue

            # Check minimum confidence threshold
            if pair.confidence < self.min_confidence:
                validation_stats['low_confidence'] += 1
                self.log_debug(
                    "Low confidence pair",
                    id=pair.national_id,
                    action=pair.action,
                    confidence=pair.confidence
                )
                continue

            # Pair is valid
            validation_stats['valid'] += 1
            valid_pairs.append(pair)

        self.log_info("Individual validation completed", **validation_stats)
        return valid_pairs

    # ------------------------------------------------------------------
    def _normalize_actions(self, pairs: List[IDActionPair]) -> List[IDActionPair]:
        """
        Normalize action names to standard format
        """
        normalized_pairs = []
        normalization_stats = defaultdict(int)

        for pair in pairs:
            original_action = pair.action.lower().strip()
            normalized_action = self.action_mapping.get(original_action, original_action)
            
            # Create new pair with normalized action
            normalized_pair = IDActionPair(
                national_id=pair.national_id,
                action=normalized_action,
                confidence=pair.confidence,
                page_number=pair.page_number,
                context=pair.context,
                start_position=pair.start_position,
                end_position=pair.end_position,
                extraction_method=pair.extraction_method,
                raw_text=pair.raw_text
            )
            
            normalized_pairs.append(normalized_pair)
            
            # Track normalizations
            if original_action != normalized_action:
                normalization_stats[f"{original_action} -> {normalized_action}"] += 1

        if normalization_stats:
            self.log_debug("Action normalizations applied", normalizations=dict(normalization_stats))

        return normalized_pairs

    # ------------------------------------------------------------------
    def _deduplicate_pairs(self, pairs: List[IDActionPair]) -> List[IDActionPair]:
        """
        Remove duplicate ID/Action combinations, keeping highest confidence
        """
        # Group pairs by (national_id, action) tuple
        groups: Dict[Tuple[str, str], List[IDActionPair]] = defaultdict(list)
        
        for pair in pairs:
            key = (pair.national_id, pair.action)
            groups[key].append(pair)

        deduplicated_pairs = []
        duplicate_count = 0

        for key, group_pairs in groups.items():
            if len(group_pairs) == 1:
                # No duplicates
                deduplicated_pairs.append(group_pairs[0])
            else:
                # Multiple pairs - keep highest confidence
                best_pair = max(group_pairs, key=lambda p: p.confidence)
                deduplicated_pairs.append(best_pair)
                duplicate_count += len(group_pairs) - 1
                
                self.log_debug(
                    "Deduplicated pairs",
                    key=key,
                    kept_confidence=best_pair.confidence,
                    removed_count=len(group_pairs) - 1
                )

        self.log_info(
            "Deduplication completed",
            input_pairs=len(pairs),
            output_pairs=len(deduplicated_pairs),
            duplicates_removed=duplicate_count
        )

        return deduplicated_pairs

    # ------------------------------------------------------------------
    def _calculate_aggregate_confidence(self, pairs: List[IDActionPair]) -> List[IDActionPair]:
        """
        Calculate aggregate confidence scores for pairs that appeared on multiple pages
        """
        # For now, we'll use the confidence as-is since deduplication already picked the best
        # In future, could implement more sophisticated confidence aggregation
        
        for pair in pairs:
            # Ensure confidence is within valid range
            pair.confidence = max(0.0, min(1.0, pair.confidence))

        return pairs

    # ------------------------------------------------------------------
    def _filter_by_confidence(self, pairs: List[IDActionPair]) -> List[IDActionPair]:
        """
        Filter pairs by confidence threshold
        """
        filtered_pairs = []
        low_confidence_count = 0

        for pair in pairs:
            if pair.confidence >= self.min_confidence:
                filtered_pairs.append(pair)
            else:
                low_confidence_count += 1

        self.log_info(
            "Confidence filtering completed",
            threshold=self.min_confidence,
            kept=len(filtered_pairs),
            filtered_out=low_confidence_count
        )

        return filtered_pairs

    # ------------------------------------------------------------------
    def _create_validated_pairs(self, pairs: List[IDActionPair]) -> List[ValidatedPair]:
        """
        Convert IDActionPair objects to ValidatedPair objects
        """
        validated_pairs = []

        for pair in pairs:
            # Determine if pair needs human review
            needs_review = (
                pair.confidence < self.review_threshold or
                pair.action not in self.action_mapping.values()
            )

            validated_pair = ValidatedPair(
                national_id=pair.national_id,
                action=pair.action,
                confidence=pair.confidence,
                page_number=pair.page_number,
                context=pair.context,
                needs_review=needs_review,
                validation_status='validated',
                original_text=pair.raw_text,
                extraction_metadata={
                    'start_position': pair.start_position,
                    'end_position': pair.end_position,
                    'extraction_method': pair.extraction_method
                }
            )

            validated_pairs.append(validated_pair)

        return validated_pairs

    # ------------------------------------------------------------------
    def _build_action_mapping(self) -> Dict[str, str]:
        """
        Build reverse mapping from action variants to standard actions
        """
        mapping = {}
        
        for standard_action, variants in ACTION_NORMALIZATIONS.items():
            # Map standard action to itself
            mapping[standard_action] = standard_action
            
            # Map all variants to standard action
            for variant in variants:
                mapping[variant.lower().strip()] = standard_action

        return mapping

    # ------------------------------------------------------------------
    def _log_validation_stats(self, input_pairs: List[IDActionPair], output_pairs: List[ValidatedPair]) -> None:
        """
        Log detailed validation statistics
        """
        # Count pairs by confidence ranges
        confidence_ranges = {
            'high (>= 0.9)': 0,
            'medium (0.7-0.9)': 0,
            'low (< 0.7)': 0
        }

        review_needed = 0
        action_distribution = defaultdict(int)

        for pair in output_pairs:
            # Confidence ranges
            if pair.confidence >= 0.9:
                confidence_ranges['high (>= 0.9)'] += 1
            elif pair.confidence >= 0.7:
                confidence_ranges['medium (0.7-0.9)'] += 1
            else:
                confidence_ranges['low (< 0.7)'] += 1

            # Review needed
            if pair.needs_review:
                review_needed += 1

            # Action distribution
            action_distribution[pair.action] += 1

        self.log_info(
            "Validation statistics",
            input_count=len(input_pairs),
            output_count=len(output_pairs),
            retention_rate=f"{len(output_pairs)/len(input_pairs)*100:.1f}%",
            confidence_distribution=dict(confidence_ranges),
            review_needed=review_needed,
            action_distribution=dict(action_distribution)
        )

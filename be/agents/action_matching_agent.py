"""
ActionMatchingAgent
──────────────────
Semantic action matching using ChromaDB vector store for only your 2 database actions:
- freeze_funds: Freeze all funds in the customer account
- release_funds: Release all held or restricted funds back to the customer account

Process:
1. Takes validated_pairs from ValidationAgent
2. Uses semantic similarity matching via ChromaDB
3. Matches to freeze_funds, release_funds, or marks as unknown_action
4. Returns MatchedAction objects with confidence scores
5. Handles only 3 cases: freeze_funds, release_funds, unknown_action
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, ValidatedPair, MatchedAction
from vector_store import get_best_action_match, get_vector_store_stats
from exceptions import AgentError, VectorStoreError
from metrics import (
    increment_vector_query,
    record_vector_query_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# Similarity thresholds for your 2 actions only
HIGH_SIMILARITY_THRESHOLD = getattr(settings, 'HIGH_SIMILARITY_THRESHOLD', 0.85)
MEDIUM_SIMILARITY_THRESHOLD = getattr(settings, 'MEDIUM_SIMILARITY_THRESHOLD', 0.70)
LOW_SIMILARITY_THRESHOLD = getattr(settings, 'LOW_SIMILARITY_THRESHOLD', 0.60)

# Your supported actions only
SUPPORTED_ACTIONS = ['freeze_funds', 'release_funds']

# Keywords for simple matching fallback
ACTION_KEYWORDS = {
    'freeze_funds': ['freeze', 'hold', 'block', 'restrict', 'suspend', 'lock'],
    'release_funds': ['release', 'unfreeze', 'unblock', 'restore', 'remove', 'lift', 'unlock']
}

class MatchQuality(Enum):
    """Quality levels for action matching"""
    HIGH = "high"           # > 0.85 similarity
    MEDIUM = "medium"       # 0.70 - 0.85 similarity  
    LOW = "low"            # 0.60 - 0.70 similarity
    NO_MATCH = "no_match"  # < 0.60 similarity

class ActionMatchingAgent(BaseAgent):
    NAME = "action_matching_agent"
    DESCRIPTION = "Match extracted actions to freeze_funds, release_funds, or unknown_action using semantic similarity"

    def __init__(self):
        """Initialize action matching agent"""
        self.high_threshold = HIGH_SIMILARITY_THRESHOLD
        self.medium_threshold = MEDIUM_SIMILARITY_THRESHOLD
        self.low_threshold = LOW_SIMILARITY_THRESHOLD
        self.supported_actions = SUPPORTED_ACTIONS

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Match validated actions to your 2 database actions or mark as unknown
        """
        start_time = time.perf_counter()
        
        if not state.validated_pairs:
            self.log_warning("No validated pairs to match")
            state.matched_actions = []
            state.progress = 0.8  # Skip to next stage
            return state

        self.log_info("Starting action matching", pairs=len(state.validated_pairs))

        try:
            # Match each validated pair to an action
            matched_actions = []
            match_stats = {
                'total_processed': 0,
                'freeze_funds_matches': 0,
                'release_funds_matches': 0,
                'unknown_actions': 0,
                'high_confidence': 0,
                'medium_confidence': 0,
                'low_confidence': 0
            }

            for pair in state.validated_pairs:
                matched_action = await self._match_single_action(pair)
                matched_actions.append(matched_action)
                
                # Update statistics
                match_stats['total_processed'] += 1
                
                if matched_action.matched_action == 'freeze_funds':
                    match_stats['freeze_funds_matches'] += 1
                elif matched_action.matched_action == 'release_funds':
                    match_stats['release_funds_matches'] += 1
                else:
                    match_stats['unknown_actions'] += 1
                
                # Track confidence levels
                if matched_action.similarity_score >= self.high_threshold:
                    match_stats['high_confidence'] += 1
                elif matched_action.similarity_score >= self.medium_threshold:
                    match_stats['medium_confidence'] += 1
                else:
                    match_stats['low_confidence'] += 1

        except Exception as exc:
            raise AgentError(
                f"Action matching failed: {exc}",
                agent_name=self.NAME,
                job_id=state.job_id,
                original_exception=exc
            ) from exc

        # Update state
        state.matched_actions = matched_actions
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.8  # 80% complete after action matching

        # Log results
        duration = time.perf_counter() - start_time
        self._log_matching_results(match_stats, duration)
        
        self.log_info(
            "Action matching completed",
            total_actions=len(matched_actions),
            freeze_matches=match_stats['freeze_funds_matches'],
            release_matches=match_stats['release_funds_matches'],
            unknown=match_stats['unknown_actions'],
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    async def _match_single_action(self, pair: ValidatedPair) -> MatchedAction:
        """
        Match a single action to freeze_funds, release_funds, or unknown_action
        """
        original_action = pair.action.lower().strip()
        
        try:
            # Try semantic matching using vector store
            semantic_match = await self._semantic_action_match(original_action)
            
            if semantic_match and semantic_match['similarity'] >= self.low_threshold:
                # Valid semantic match found
                return self._create_matched_action(pair, semantic_match, "semantic")
            else:
                # Try simple keyword matching as fallback
                keyword_match = self._keyword_action_match(original_action)
                
                if keyword_match:
                    return self._create_matched_action(pair, keyword_match, "keyword")
                else:
                    # No match found - mark as unknown
                    return self._create_unknown_action(pair)
                    
        except Exception as exc:
            self.log_error(f"Action matching error for '{original_action}': {exc}")
            # Return unknown action on error
            return self._create_unknown_action(pair, error_message=str(exc))

    # ------------------------------------------------------------------
    async def _semantic_action_match(self, action_text: str) -> Optional[Dict[str, Any]]:
        """
        Use vector store to find semantic match for action
        """
        try:
            increment_vector_query()
            start_time = time.perf_counter()
            
            # Query vector store
            match_result = get_best_action_match(
                query_text=action_text,
                threshold=self.low_threshold
            )
            
            query_time = time.perf_counter() - start_time
            record_vector_query_time(query_time)
            
            if match_result and match_result['action_name'] in self.supported_actions:
                self.log_debug(
                    f"Semantic match found: '{action_text}' -> {match_result['action_name']} "
                    f"(similarity: {match_result['similarity']:.3f})"
                )
                return match_result
            else:
                self.log_debug(f"No semantic match found for: '{action_text}'")
                return None
                
        except Exception as exc:
            self.log_error(f"Semantic matching failed for '{action_text}': {exc}")
            return None

    # ------------------------------------------------------------------
    def _keyword_action_match(self, action_text: str) -> Optional[Dict[str, Any]]:
        """
        Simple keyword-based matching as fallback for your 2 actions
        """
        action_lower = action_text.lower()
        
        # Check for freeze keywords
        freeze_keywords = ACTION_KEYWORDS['freeze_funds']
        if any(keyword in action_lower for keyword in freeze_keywords):
            return {
                'action_name': 'freeze_funds',
                'similarity': 0.75,  # Medium confidence for keyword match
                'confidence': 0.75,
                'description': 'Freeze all funds in the customer account',
                'match_method': 'keyword'
            }
        
        # Check for release keywords
        release_keywords = ACTION_KEYWORDS['release_funds']
        if any(keyword in action_lower for keyword in release_keywords):
            return {
                'action_name': 'release_funds',
                'similarity': 0.75,  # Medium confidence for keyword match
                'confidence': 0.75,
                'description': 'Release all held or restricted funds back to the customer account',
                'match_method': 'keyword'
            }
        
        self.log_debug(f"No keyword match found for: '{action_text}'")
        return None

    # ------------------------------------------------------------------
    def _create_matched_action(
        self, 
        pair: ValidatedPair, 
        match_result: Dict[str, Any], 
        match_method: str
    ) -> MatchedAction:
        """
        Create MatchedAction object for successful match
        """
        action_name = match_result['action_name']
        similarity = match_result['similarity']
        
        # Determine if needs review based on confidence
        needs_review = similarity < self.medium_threshold
        match_status = "matched"
        
        return MatchedAction(
            national_id=pair.national_id,
            original_action=pair.action,
            matched_action=action_name,
            similarity_score=similarity,
            confidence=pair.confidence,  # Keep original extraction confidence
            page_number=pair.page_number,
            context=pair.context,
            needs_review=needs_review,
            match_status=match_status,
            match_method=match_method,
            customer_id=pair.customer_id,
            customer_found=pair.customer_found,
            customer_status=pair.customer_status,
            customer_name=pair.customer_name,
            matching_metadata={
                'semantic_match': match_result,
                'similarity_threshold_used': self.low_threshold,
                'match_quality': self._get_match_quality(similarity),
                'supported_action': True,
                'customer_metadata': pair.customer_metadata or {}
            }
        )

    # ------------------------------------------------------------------
    def _create_unknown_action(
        self, 
        pair: ValidatedPair, 
        error_message: Optional[str] = None
    ) -> MatchedAction:
        """
        Create MatchedAction object for unknown/unmatched actions
        """
        return MatchedAction(
            national_id=pair.national_id,
            original_action=pair.action,
            matched_action="unknown_action",
            similarity_score=0.0,
            confidence=pair.confidence,  # Keep original extraction confidence
            page_number=pair.page_number,
            context=pair.context,
            needs_review=True,  # Always needs review
            match_status="unknown_action",
            match_method="none",
            customer_id=pair.customer_id,
            customer_found=pair.customer_found,
            customer_status=pair.customer_status,
            customer_name=pair.customer_name,
            matching_metadata={
                'reason': 'No matching action found in supported actions',
                'supported_actions': self.supported_actions,
                'error_message': error_message,
                'requires_manual_review': True,
                'customer_metadata': pair.customer_metadata or {}
            }
        )

    # ------------------------------------------------------------------
    def _get_match_quality(self, similarity: float) -> MatchQuality:
        """Determine match quality based on similarity score"""
        if similarity >= self.high_threshold:
            return MatchQuality.HIGH
        elif similarity >= self.medium_threshold:
            return MatchQuality.MEDIUM
        elif similarity >= self.low_threshold:
            return MatchQuality.LOW
        else:
            return MatchQuality.NO_MATCH

    # ------------------------------------------------------------------
    def _log_matching_results(self, stats: Dict[str, Any], duration: float) -> None:
        """Log detailed matching results"""
        total = stats['total_processed']
        
        if total == 0:
            return
        
        freeze_rate = (stats['freeze_funds_matches'] / total) * 100
        release_rate = (stats['release_funds_matches'] / total) * 100
        unknown_rate = (stats['unknown_actions'] / total) * 100
        
        high_conf_rate = (stats['high_confidence'] / total) * 100
        medium_conf_rate = (stats['medium_confidence'] / total) * 100
        low_conf_rate = (stats['low_confidence'] / total) * 100
        
        self.log_info(
            "Action Matching Statistics",
            total_processed=total,
            freeze_funds_matches=f"{stats['freeze_funds_matches']} ({freeze_rate:.1f}%)",
            release_funds_matches=f"{stats['release_funds_matches']} ({release_rate:.1f}%)",
            unknown_actions=f"{stats['unknown_actions']} ({unknown_rate:.1f}%)",
            high_confidence=f"{stats['high_confidence']} ({high_conf_rate:.1f}%)",
            medium_confidence=f"{stats['medium_confidence']} ({medium_conf_rate:.1f}%)",
            low_confidence=f"{stats['low_confidence']} ({low_conf_rate:.1f}%)",
            processing_time=f"{duration:.2f}s",
            actions_per_second=f"{total/duration:.1f}" if duration > 0 else "N/A"
        )

    # ------------------------------------------------------------------
    def get_matching_stats(self) -> Dict[str, Any]:
        """Get current matching statistics from vector store"""
        try:
            vector_stats = get_vector_store_stats()
            return {
                'supported_actions': self.supported_actions,
                'similarity_thresholds': {
                    'high': self.high_threshold,
                    'medium': self.medium_threshold,
                    'low': self.low_threshold
                },
                'vector_store_stats': vector_stats,
                'agent_name': self.NAME
            }
        except Exception as exc:
            self.log_error(f"Failed to get matching stats: {exc}")
            return {
                'error': str(exc),
                'supported_actions': self.supported_actions
            }

    # ------------------------------------------------------------------
    def validate_action_support(self, action_name: str) -> bool:
        """Check if an action is supported"""
        return action_name in self.supported_actions

    # ------------------------------------------------------------------
    def get_action_examples(self, action_name: str) -> List[str]:
        """Get example phrases for an action"""
        if action_name not in self.supported_actions:
            return []
        
        keywords = ACTION_KEYWORDS.get(action_name, [])
        if not keywords:
            return []
        
        # Fix: Create examples properly using the keywords list
        examples = []
        for keyword in keywords[:3]:  # First 3 keywords
            examples.extend([
                f"{keyword} funds",
                f"{keyword} account",
                f"{keyword} customer funds"
            ])
        
        return examples


# Export the agent class
__all__ = ['ActionMatchingAgent']

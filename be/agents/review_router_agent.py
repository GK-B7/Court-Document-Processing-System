"""
ReviewRouterAgent
─────────────────
1. Takes matched_actions from ActionMatchingAgent
2. Analyzes each action for review requirements based on multiple criteria
3. Routes high-confidence items to automatic execution
4. Routes low-confidence/problematic items to human review queue
5. Creates review queue entries in database
6. Populates state.review_required and state.auto_approved lists
7. Updates progress and provides routing statistics

Review Criteria:
• Confidence thresholds (extraction + matching)
• Customer status and validation
• Action type risk assessment
• Account balance considerations
• Regulatory compliance checks
"""

from __future__ import annotations

import time
import logging
from typing import List, Dict, Optional, Tuple
from enum import Enum

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, MatchedAction, ReviewItem
from database import ReviewQueueModel
from exceptions import DatabaseError, AgentError
from metrics import (
    increment_review_item_created,
    increment_database_query,
    record_database_query_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Review Routing Constants and Enums
# ─────────────────────────────────────────────────────────────────────────────

class ReviewReason(Enum):
    """Reasons why an action might need human review"""
    LOW_CONFIDENCE = "low_confidence"
    UNKNOWN_ACTION = "unknown_action"
    CUSTOMER_NOT_FOUND = "customer_not_found"
    CUSTOMER_INACTIVE = "customer_inactive"
    HIGH_RISK_ACTION = "high_risk_action"
    LARGE_AMOUNT = "large_amount"
    CONFLICTING_ACTIONS = "conflicting_actions"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    SYSTEM_UNCERTAINTY = "system_uncertainty"

class ActionRiskLevel(Enum):
    """Risk levels for different actions"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

# Action risk mappings (simplified for your 2 actions)
ACTION_RISK_LEVELS = {
    'freeze_funds': ActionRiskLevel.MEDIUM,
    'release_funds': ActionRiskLevel.MEDIUM,
    'unknown_action': ActionRiskLevel.CRITICAL
}

# Thresholds
AUTO_APPROVE_THRESHOLD = getattr(settings, 'AUTO_APPROVE_THRESHOLD', 0.95)
REVIEW_THRESHOLD = getattr(settings, 'REVIEW_THRESHOLD', 0.8)
HIGH_VALUE_THRESHOLD = 50000.0  # USD threshold for high-value accounts

class ReviewRouterAgent(BaseAgent):
    NAME = "review_router_agent"
    DESCRIPTION = "Route actions to automatic execution or human review based on risk assessment"

    def __init__(self):
        """Initialize review router agent with thresholds and rules"""
        self.auto_approve_threshold = AUTO_APPROVE_THRESHOLD
        self.review_threshold = REVIEW_THRESHOLD
        self.high_value_threshold = HIGH_VALUE_THRESHOLD
        self.enable_risk_assessment = getattr(settings, 'ENABLE_RISK_ASSESSMENT', True)
        self.enable_regulatory_checks = getattr(settings, 'ENABLE_REGULATORY_CHECKS', True)

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Route matched actions to execution or review based on risk assessment
        """
        start_time = time.perf_counter()
        
        if not state.matched_actions:
            self.log_warning("No matched actions to route")
            state.review_required = []
            state.auto_approved = []
            state.progress = 0.95  # Skip to next stage
            return state

        self.log_info("Starting review routing", actions=len(state.matched_actions))

        try:
            # Step 1: Assess each action for review requirements
            routing_decisions = []
            for action in state.matched_actions:
                decision = await self._assess_action_for_review(action)
                routing_decisions.append(decision)
            
            # Step 2: Create review queue entries for items needing review
            review_items = await self._create_review_queue_entries(
                state.job_id, 
                routing_decisions
            )
            
            # Step 3: Separate actions into review and auto-approved lists
            auto_approved, review_required = self._separate_actions_by_decision(
                state.matched_actions, 
                routing_decisions,
                review_items
            )
            
            # Step 4: Handle conflicting actions
            auto_approved, additional_reviews = self._handle_conflicting_actions(auto_approved)
            review_required.extend(additional_reviews)

        except Exception as exc:
            raise AgentError(
                f"Review routing failed: {exc}",
                agent_name=self.NAME,
                job_id=state.job_id,
                original_exception=exc
            ) from exc

        # Update state
        state.review_required = review_required
        state.auto_approved = auto_approved
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.95  # 95% complete after routing

        # Log routing statistics
        self._log_routing_stats(state.matched_actions, auto_approved, review_required, routing_decisions)
        
        duration = time.perf_counter() - start_time
        self.log_info(
            "Review routing completed",
            total_actions=len(state.matched_actions),
            auto_approved=len(auto_approved),
            review_required=len(review_required),
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    async def _assess_action_for_review(self, action: MatchedAction) -> Dict:
        """
        Assess a single action to determine if it needs human review
        """
        review_reasons = []
        risk_factors = []
        should_review = False
        risk_level = ACTION_RISK_LEVELS.get(action.matched_action, ActionRiskLevel.MEDIUM)

        # 1. Confidence-based checks
        if action.confidence < self.review_threshold:
            review_reasons.append(ReviewReason.LOW_CONFIDENCE)
            should_review = True
            
        if action.similarity_score < self.review_threshold:
            review_reasons.append(ReviewReason.SYSTEM_UNCERTAINTY)
            should_review = True

        # 2. Action-specific checks
        if action.matched_action == 'unknown_action':
            review_reasons.append(ReviewReason.UNKNOWN_ACTION)
            should_review = True
            risk_level = ActionRiskLevel.CRITICAL

        # 3. Customer-related checks
        if not action.customer_found:
            review_reasons.append(ReviewReason.CUSTOMER_NOT_FOUND)
            should_review = True
            
        if action.customer_found and action.customer_status not in ['active']:
            review_reasons.append(ReviewReason.CUSTOMER_INACTIVE)
            should_review = True

        # 4. Risk-based checks
        if risk_level in [ActionRiskLevel.HIGH, ActionRiskLevel.CRITICAL]:
            review_reasons.append(ReviewReason.HIGH_RISK_ACTION)
            should_review = True

        # 5. Account balance checks (using default balance since we don't have real balance data)
        customer_balance = 1000.0  # Default balance from our customer service
        if customer_balance > self.high_value_threshold:
            review_reasons.append(ReviewReason.LARGE_AMOUNT)
            risk_factors.append(f"High account balance: ${customer_balance:,.2f}")
            should_review = True

        # 6. Regulatory compliance checks
        if self.enable_regulatory_checks:
            regulatory_review = self._check_regulatory_requirements(action)
            if regulatory_review:
                review_reasons.append(ReviewReason.REGULATORY_REQUIREMENT)
                risk_factors.extend(regulatory_review)
                should_review = True

        # 7. Override for very high confidence
        if (action.confidence >= self.auto_approve_threshold and 
            action.similarity_score >= self.auto_approve_threshold and
            risk_level == ActionRiskLevel.LOW and
            action.customer_found and
            action.customer_status == 'active'):
            # Override review requirement for very high confidence, low-risk actions
            should_review = False
            review_reasons = []

        return {
            'action': action,
            'should_review': should_review,
            'review_reasons': review_reasons,
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'confidence_score': action.confidence,
            'similarity_score': action.similarity_score
        }

    # ------------------------------------------------------------------
    def _check_regulatory_requirements(self, action: MatchedAction) -> List[str]:
        """
        Check if action requires review due to regulatory requirements
        """
        regulatory_issues = []
        customer_balance = 1000.0  # Default balance since we don't have real balance data
        
        # Large transaction reporting requirements
        if customer_balance > 10000 and action.matched_action in ['freeze_funds', 'release_funds']:
            regulatory_issues.append("Large transaction monitoring required")
        
        # Suspicious activity regulations
        if action.matched_action == 'flag_suspicious':
            regulatory_issues.append("SAR filing may be required")

        return regulatory_issues

    # ------------------------------------------------------------------
    async def _create_review_queue_entries(
        self, 
        job_id: str, 
        routing_decisions: List[Dict]
    ) -> List[ReviewItem]:
        """
        Create database entries for actions requiring human review
        """
        review_items = []
        start_time = time.perf_counter()

        try:
            for decision in routing_decisions:
                if not decision['should_review']:
                    continue
                
                action = decision['action']
                
                # Create review queue entry in database using correct parameter names
                increment_database_query()
                review_id = ReviewQueueModel.create_review_item(
                    job_id=job_id,
                    national_id=action.national_id,
                    customer_id=action.customer_id,
                    original_action=action.original_action,  # Fixed: use original_action not extracted_action
                    matched_action=action.matched_action,
                    confidence=action.confidence,
                    page_number=action.page_number,
                    context=action.context or '',
                    review_reason=', '.join([reason.value for reason in decision['review_reasons']]),
                    priority=self._calculate_review_priority_numeric(decision)
                )
                
                # Create ReviewItem object
                review_item = ReviewItem(
                    review_id=review_id,
                    job_id=job_id,
                    national_id=action.national_id,
                    original_action=action.original_action,
                    matched_action=action.matched_action,
                    confidence=action.confidence,
                    similarity_score=action.similarity_score,
                    page_number=action.page_number,
                    context=action.context,
                    customer_id=action.customer_id,
                    customer_name=action.customer_name,
                    customer_status=action.customer_status,
                    review_reasons=[reason.value for reason in decision['review_reasons']],
                    risk_level=decision['risk_level'].value,
                    risk_factors=decision['risk_factors'],
                    priority=self._calculate_review_priority(decision),
                    metadata={
                        'confidence_score': action.confidence,
                        'similarity_score': action.similarity_score,
                        'match_method': action.match_method,
                        'customer_found': action.customer_found,
                        'routing_timestamp': time.time()
                    }
                )
                
                review_items.append(review_item)
                increment_review_item_created()

            record_database_query_time(time.perf_counter() - start_time)
            
            self.log_info("Review queue entries created", count=len(review_items))

        except Exception as exc:
            raise DatabaseError(
                f"Failed to create review queue entries: {exc}",
                operation="batch_insert",
                table_name="review_queue",
                original_exception=exc
            ) from exc

        return review_items

    # ------------------------------------------------------------------
    def _calculate_review_priority(self, decision: Dict) -> str:
        """
        Calculate priority for review items based on risk factors (string)
        """
        risk_level = decision['risk_level']
        review_reasons = decision['review_reasons']
        confidence = decision['confidence_score']
        
        # Critical priority
        if (risk_level == ActionRiskLevel.CRITICAL or
            ReviewReason.UNKNOWN_ACTION in review_reasons or
            ReviewReason.CUSTOMER_NOT_FOUND in review_reasons):
            return 'critical'
        
        # High priority
        if (risk_level == ActionRiskLevel.HIGH or
            ReviewReason.HIGH_RISK_ACTION in review_reasons or
            ReviewReason.LARGE_AMOUNT in review_reasons or
            ReviewReason.REGULATORY_REQUIREMENT in review_reasons):
            return 'high'
        
        # Medium priority
        if (risk_level == ActionRiskLevel.MEDIUM or
            confidence < 0.7 or
            ReviewReason.CUSTOMER_INACTIVE in review_reasons):
            return 'medium'
        
        # Low priority
        return 'low'

    def _calculate_review_priority_numeric(self, decision: Dict) -> int:
        """
        Calculate numeric priority for database (1=highest, 5=lowest)
        """
        priority_str = self._calculate_review_priority(decision)
        priority_map = {
            'critical': 1,
            'high': 2,
            'medium': 3,
            'low': 4
        }
        return priority_map.get(priority_str, 3)

    # ------------------------------------------------------------------
    def _separate_actions_by_decision(
        self, 
        matched_actions: List[MatchedAction], 
        routing_decisions: List[Dict],
        review_items: List[ReviewItem]
    ) -> Tuple[List[MatchedAction], List[ReviewItem]]:
        """
        Separate actions into auto-approved and review-required lists
        """
        auto_approved = []
        review_required = []
        
        for i, decision in enumerate(routing_decisions):
            action = matched_actions[i]
            
            if decision['should_review']:
                # Find corresponding review item
                review_item = None
                for item in review_items:
                    if (item.national_id == action.national_id and 
                        item.original_action == action.original_action):
                        review_item = item
                        break
                
                if review_item:
                    review_required.append(review_item)
            else:
                auto_approved.append(action)

        return auto_approved, review_required

    # ------------------------------------------------------------------
    def _handle_conflicting_actions(
        self, 
        auto_approved: List[MatchedAction]
    ) -> Tuple[List[MatchedAction], List[ReviewItem]]:
        """
        Detect and handle conflicting actions for the same customer
        """
        # Group actions by customer
        customer_actions = {}
        for action in auto_approved:
            customer_id = action.customer_id or action.national_id
            if customer_id not in customer_actions:
                customer_actions[customer_id] = []
            customer_actions[customer_id].append(action)
        
        # Detect conflicts
        final_auto_approved = []
        additional_reviews = []
        
        for customer_id, actions in customer_actions.items():
            if len(actions) == 1:
                # No conflicts for this customer
                final_auto_approved.extend(actions)
                continue
            
            # Check for conflicting actions
            action_types = [action.matched_action for action in actions]
            conflicts = self._detect_action_conflicts(action_types)
            
            if conflicts:
                # Send all actions for this customer to review
                self.log_warning(
                    "Conflicting actions detected",
                    customer_id=customer_id,
                    actions=action_types,
                    conflicts=conflicts
                )
                
                for action in actions:
                    review_item = ReviewItem(
                        review_id=None,  # Will be created later
                        job_id=action.customer_id,  # Temporary
                        national_id=action.national_id,
                        original_action=action.original_action,
                        matched_action=action.matched_action,
                        confidence=action.confidence,
                        similarity_score=action.similarity_score,
                        page_number=action.page_number,
                        context=action.context,
                        customer_id=action.customer_id,
                        customer_name=action.customer_name,
                        customer_status=action.customer_status,
                        review_reasons=[ReviewReason.CONFLICTING_ACTIONS.value],
                        risk_level=ActionRiskLevel.MEDIUM.value,
                        risk_factors=[f"Conflicts with: {', '.join(conflicts)}"],
                        priority='high',
                        metadata={'conflict_detection': True}
                    )
                    additional_reviews.append(review_item)
            else:
                # No conflicts - approve all actions for this customer
                final_auto_approved.extend(actions)
        
        return final_auto_approved, additional_reviews

    # ------------------------------------------------------------------
    def _detect_action_conflicts(self, action_types: List[str]) -> List[str]:
        """
        Detect conflicting action combinations for your 2 actions
        """
        conflicts = []
        action_set = set(action_types)
        
        # Define conflicting action pairs for your specific actions
        conflict_rules = [
            ('freeze_funds', 'release_funds'),  # Can't freeze and release at same time
        ]
        
        for action1, action2 in conflict_rules:
            if action1 in action_set and action2 in action_set:
                conflicts.append(f"{action1} vs {action2}")
        
        return conflicts

    # ------------------------------------------------------------------
    def _log_routing_stats(
        self, 
        matched_actions: List[MatchedAction],
        auto_approved: List[MatchedAction],
        review_required: List[ReviewItem],
        routing_decisions: List[Dict]
    ) -> None:
        """
        Log detailed routing statistics
        """
        total_actions = len(matched_actions)
        auto_count = len(auto_approved)
        review_count = len(review_required)
        
        # Review reason distribution
        reason_distribution = {}
        for decision in routing_decisions:
            if decision['should_review']:
                for reason in decision['review_reasons']:
                    reason_name = reason.value if hasattr(reason, 'value') else str(reason)
                    reason_distribution[reason_name] = reason_distribution.get(reason_name, 0) + 1
        
        # Risk level distribution
        risk_distribution = {}
        for decision in routing_decisions:
            risk_level = decision['risk_level'].value
            risk_distribution[risk_level] = risk_distribution.get(risk_level, 0) + 1
        
        # Priority distribution for review items
        priority_distribution = {}
        for item in review_required:
            priority = item.priority
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Confidence statistics
        confidences = [action.confidence for action in matched_actions]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Customer status distribution
        customer_status_dist = {}
        for action in matched_actions:
            status = action.customer_status or 'unknown'
            customer_status_dist[status] = customer_status_dist.get(status, 0) + 1

        self.log_info(
            "Review routing statistics",
            routing_summary={
                'total_actions': total_actions,
                'auto_approved': auto_count,
                'review_required': review_count,
                'auto_approval_rate': f"{auto_count/total_actions*100:.1f}%" if total_actions > 0 else "0%",
                'review_rate': f"{review_count/total_actions*100:.1f}%" if total_actions > 0 else "0%"
            },
            review_reasons=reason_distribution,
            risk_levels=risk_distribution,
            priority_levels=priority_distribution,
            confidence_stats={
                'average': f"{avg_confidence:.3f}",
                'above_auto_threshold': len([c for c in confidences if c >= self.auto_approve_threshold]),
                'below_review_threshold': len([c for c in confidences if c < self.review_threshold])
            },
            customer_status_distribution=customer_status_dist
        )

# Export the agent class
__all__ = ['ReviewRouterAgent']

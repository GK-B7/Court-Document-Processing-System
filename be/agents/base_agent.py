"""
BaseAgent for the comprehensive LangGraph workflow.

All concrete agents should inherit from BaseAgent and implement the
`process` coroutine which receives a `DocumentState` and returns an
updated `DocumentState`.

The BaseAgent provides:
1.  Standardised logging
2.  Timing + metrics hooks
3.  Unified exception handling (maps any uncaught error to AgentError)
4.  Helper `__call__` so the class instance can be passed directly as a
    LangGraph node.
"""

from __future__ import annotations

import time
import logging
from abc import ABC, abstractmethod
from typing import Any

from config import settings
from agents.state import DocumentState, ProcessingStatus
from exceptions import AgentError
from metrics import (
    record_document_processing_time,
    increment_vector_search,
    increment_llm_api_call,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)


class BaseAgent(ABC):
    """
    Abstract base-class for every agent in the pipeline.
    """

    # Concrete agents MUST set these two attributes
    NAME: str = "base_agent"
    DESCRIPTION: str = "Abstract base agent. Should be overridden."

    async def __call__(self, state: DocumentState) -> DocumentState:  # pragma: no cover
        """
        Makes an instance directly callable so we can register it as a
        LangGraph node, e.g.:

        workflow.add_node("preprocess", PreprocessingAgent())
        """
        start_time = time.perf_counter()

        # Keep track of which agent is currently running
        state.current_agent = self.NAME

        try:
            updated_state = await self.process(state)

            # ── House-keeping ────────────────────────────────────────────
            updated_state.current_agent = None
            # Mark PREPROCESSING as completed only if it was still pending
            if updated_state.status == ProcessingStatus.PROCESSING.value:
                updated_state.progress = min(updated_state.progress + 0.1, 1.0)

            return updated_state

        except Exception as exc:  # noqa: BLE001
            logger.exception(f"{self.NAME}: unhandled exception")
            # Wrap in our own AgentError to bubble up through workflow
            raise AgentError(
                f"{self.NAME} failed – {exc}",
                agent_name=self.NAME,
                job_id=getattr(state, "job_id", None),
                state_data=state.dict() if hasattr(state, "dict") else {},
                original_exception=exc,
            ) from exc

        finally:
            duration = time.perf_counter() - start_time
            record_document_processing_time(duration)

    # ------------------------------------------------------------------
    # Concrete implementations must provide this coroutine
    # ------------------------------------------------------------------
    @abstractmethod
    async def process(self, state: DocumentState) -> DocumentState:  # pragma: no cover
        """
        Perform the agent's work and return the updated state.

        NOTE: Must be implemented by child classes.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Convenience helpers for all agents
    # ------------------------------------------------------------------
    def log_debug(self, msg: str, **kwargs: Any) -> None:
        logger.debug(f"[{self.NAME}] {msg} – {kwargs if kwargs else ''}")

    def log_info(self, msg: str, **kwargs: Any) -> None:
        logger.info(f"[{self.NAME}] {msg} – {kwargs if kwargs else ''}")

    def log_warning(self, msg: str, **kwargs: Any) -> None:
        logger.warning(f"[{self.NAME}] {msg} – {kwargs if kwargs else ''}")

    def log_error(self, msg: str, **kwargs: Any) -> None:
        logger.error(f"[{self.NAME}] {msg} – {kwargs if kwargs else ''}")

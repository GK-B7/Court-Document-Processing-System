"""
ExtractionAgent
───────────────
1. Takes preprocessed pages_text from DocumentState
2. For each page, sends text to GPT-4o with structured prompt
3. Extracts National ID & Action pairs with confidence scores
4. Handles multiple IDs/actions per page
5. Populates state.extracted_pairs with structured data
6. Updates progress and logs metrics

GPT-4o Prompt Strategy:
• Use structured JSON output format
• Request confidence scores for each extraction
• Handle edge cases like partial IDs or ambiguous actions
• Provide context about financial document processing
"""

from __future__ import annotations

import time
import json
import logging
from typing import List, Dict, Any

from openai import OpenAI
from openai.types.chat import ChatCompletion

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus, IDActionPair
from exceptions import LLMError, AgentError
from metrics import (
    increment_llm_api_call,
    record_llm_response_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI Client Setup
# ─────────────────────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Extraction Prompt Template
# ─────────────────────────────────────────────────────────────────────────────
EXTRACTION_PROMPT = """
You are an expert document processor specializing in financial documents. Your task is to extract National ID numbers and associated actions from document text.

INSTRUCTIONS:
1. Look for National ID numbers (typically 10-20 digit numbers)
2. Identify actions to be taken on customer accounts (freeze, release, close, suspend, etc.)
3. Match each National ID with its corresponding action
4. Provide confidence scores (0.0 to 1.0) for each extraction
5. Return results in the specified JSON format

CONTEXT:
- This is page {page_num} of a financial document
- Text was extracted via {source} with {confidence:.2f} confidence
- Look for patterns like: "ID: 1234567890 - Freeze account" or "Customer 0987654321 should have funds released"

DOCUMENT TEXT:
{text}

OUTPUT FORMAT (JSON only, no other text):
{{
    "extractions": [
        {{
            "national_id": "1234567890",
            "action": "freeze funds",
            "confidence": 0.95,
            "context": "brief context where this was found",
            "start_position": 123,
            "end_position": 156
        }}
    ],
    "page_confidence": 0.90,
    "notes": "any relevant observations"
}}

IMPORTANT:
- If no valid ID/action pairs found, return empty extractions array
- Be conservative with confidence scores
- Include surrounding context for verification
- Flag ambiguous cases with lower confidence
"""


class ExtractionAgent(BaseAgent):
    NAME = "extraction_agent"
    DESCRIPTION = "Extract National ID & Action pairs using GPT-4o"

    def __init__(self):
        """Initialize extraction agent with retry settings"""
        self.max_retries = 3
        self.retry_delay = 2.0
        self.token_limit = settings.OPENAI_MAX_TOKENS

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:
        """
        Process each page through GPT-4o for ID/Action extraction
        """
        start_time = time.perf_counter()
        
        if not state.pages_text:
            raise AgentError(
                "No pages_text found in state - preprocessing may have failed",
                agent_name=self.NAME,
                job_id=state.job_id
            )

        extracted_pairs: List[IDActionPair] = []
        total_pages = len(state.pages_text)
        
        self.log_info("Starting extraction", pages=total_pages)

        try:
            for page_data in state.pages_text:
                page_num = page_data["page_num"]
                page_text = page_data["text"]
                source = page_data["source"]
                page_confidence = page_data["confidence"]

                # Skip pages with very little text
                if len(page_text.strip()) < 10:
                    self.log_debug("Skipping page with minimal text", page=page_num)
                    continue

                # Extract from this page
                page_extractions = await self._extract_from_page(
                    page_text=page_text,
                    page_num=page_num,
                    source=source,
                    page_confidence=page_confidence
                )

                extracted_pairs.extend(page_extractions)
                
                # Update progress incrementally
                progress_increment = 0.25 / total_pages  # 25% of total progress for extraction
                state.progress = min(state.progress + progress_increment, 0.4)

        except Exception as exc:
            raise AgentError(
                f"Extraction failed: {exc}",
                agent_name=self.NAME,
                job_id=state.job_id,
                original_exception=exc
            ) from exc

        # Update state with extracted data
        state.extracted_pairs = extracted_pairs
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.4  # 40% complete after extraction

        # Log results and metrics
        duration = time.perf_counter() - start_time
        record_llm_response_time(duration)
        
        self.log_info(
            "Extraction completed",
            pairs_found=len(extracted_pairs),
            pages_processed=total_pages,
            duration=f"{duration:.2f}s"
        )

        return state

    # ------------------------------------------------------------------
    async def _extract_from_page(
        self,
        page_text: str,
        page_num: int,
        source: str, 
        page_confidence: float
    ) -> List[IDActionPair]:
        """
        Extract ID/Action pairs from a single page using GPT-4o
        """
        start_time = time.perf_counter()
        
        # Prepare the prompt
        prompt = EXTRACTION_PROMPT.format(
            page_num=page_num,
            source=source,
            confidence=page_confidence,
            text=page_text[:3000]  # Limit text length to avoid token limits
        )

        # Call OpenAI API with retries
        response_data = await self._call_openai_with_retry(prompt, page_num)
        
        # Parse response into IDActionPair objects
        extractions = self._parse_extraction_response(
            response_data, 
            page_num, 
            page_confidence
        )

        # Record metrics
        increment_llm_api_call()
        record_llm_response_time(time.perf_counter() - start_time)
        
        self.log_debug(
            "Page extraction completed",
            page=page_num,
            extractions=len(extractions)
        )

        return extractions

    # ------------------------------------------------------------------
    async def _call_openai_with_retry(self, prompt: str, page_num: int) -> Dict[str, Any]:
        """
        Call OpenAI API with retry logic and error handling
        """
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response: ChatCompletion = openai_client.chat.completions.create(
                    model=settings.OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise document extraction specialist. Always return valid JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    temperature=settings.OPENAI_TEMPERATURE,
                    max_tokens=self.token_limit,
                    timeout=60.0
                )

                # Extract content from response
                content = response.choices[0].message.content
                if not content:
                    raise LLMError("Empty response from OpenAI", model_name=settings.OPENAI_MODEL)

                # Parse JSON response
                try:
                    return json.loads(content)
                except json.JSONDecodeError as json_err:
                    # Attempt to extract JSON from response if it's wrapped in text
                    import re
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group())
                    raise LLMError(
                        f"Invalid JSON response: {content[:200]}...",
                        model_name=settings.OPENAI_MODEL,
                        api_call_type="extraction"
                    ) from json_err

            except Exception as exc:
                last_exception = exc
                self.log_warning(
                    f"OpenAI API attempt {attempt + 1} failed",
                    page=page_num,
                    error=str(exc)
                )
                
                if attempt < self.max_retries - 1:
                    # Wait before retry
                    import asyncio
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
                else:
                    break

        # All retries failed
        raise LLMError(
            f"OpenAI API failed after {self.max_retries} attempts: {last_exception}",
            model_name=settings.OPENAI_MODEL,
            api_call_type="extraction",
            original_exception=last_exception
        )

    # ------------------------------------------------------------------
    def _parse_extraction_response(
        self, 
        response_data: Dict[str, Any], 
        page_num: int,
        page_confidence: float
    ) -> List[IDActionPair]:
        """
        Parse OpenAI response into IDActionPair objects
        """
        extractions = []
        
        try:
            raw_extractions = response_data.get("extractions", [])
            page_conf_from_llm = response_data.get("page_confidence", page_confidence)
            
            for extraction in raw_extractions:
                # Validate required fields
                national_id = extraction.get("national_id", "").strip()
                action = extraction.get("action", "").strip()
                confidence = float(extraction.get("confidence", 0.0))
                
                if not national_id or not action:
                    self.log_warning("Skipping extraction with missing ID or action", extraction=extraction)
                    continue

                # Create IDActionPair object
                pair = IDActionPair(
                    national_id=national_id,
                    action=action.lower().replace(" ", "_"),  # Normalize action format
                    confidence=min(confidence, page_conf_from_llm),  # Cap by page confidence
                    page_number=page_num,
                    context=extraction.get("context", ""),
                    start_position=extraction.get("start_position"),
                    end_position=extraction.get("end_position"),
                    extraction_method="llm",
                    raw_text=extraction.get("context", "")
                )
                
                extractions.append(pair)

            self.log_debug(
                "Parsed extractions",
                page=page_num,
                valid_extractions=len(extractions),
                raw_count=len(raw_extractions)
            )

        except Exception as exc:
            self.log_error(
                "Failed to parse extraction response",
                page=page_num,
                response=response_data,
                error=str(exc)
            )
            # Don't raise - return empty list to continue processing other pages

        return extractions

    # ------------------------------------------------------------------
    def _estimate_token_count(self, text: str) -> int:
        """
        Rough estimation of token count for text
        """
        # Approximate: 1 token ≈ 4 characters for English text
        return len(text) // 4

    def _truncate_text_to_tokens(self, text: str, max_tokens: int) -> str:
        """
        Truncate text to fit within token limit
        """
        max_chars = max_tokens * 4  # Rough conversion
        if len(text) <= max_chars:
            return text
            
        # Truncate and add note
        truncated = text[:max_chars-100]
        return truncated + "\n\n[TEXT TRUNCATED DUE TO LENGTH]"

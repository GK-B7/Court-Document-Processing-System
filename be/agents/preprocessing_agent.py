"""
PreprocessingAgent
──────────────────
1. Accepts a PDF file path from DocumentState.
2. Extracts text page-by-page using pdfplumber.
3. Falls back to OCR (PyMuPDF + Tesseract) if a page has < MIN_CHARS.
4. Populates `state.pages_text` with:
   {
       "page_num": int,
       "text": str,
       "source": "text" | "ocr",
       "confidence": float   # 0-1 simple heuristic
   }
5. Updates `state.progress`.

Heuristics:
• If pdfplumber returns ≥ MIN_CHARS we assume high confidence (1.0)
• OCR confidence is estimated by proportion of recognised characters.
  (Tesseract conf not used because pytesseract returns string conf lines)
"""

from __future__ import annotations

import io
import time
import logging
from typing import List, Dict

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from config import settings
from agents.base_agent import BaseAgent
from agents.state import DocumentState, ProcessingStatus
from exceptions import PDFError
from metrics import (
    increment_ocr_operation,
    record_ocr_processing_time,
    record_document_processing_time,
)

logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
MIN_CHARS = 25  # below this triggers OCR


class PreprocessingAgent(BaseAgent):
    NAME = "preprocessing_agent"
    DESCRIPTION = "Extract text from PDF & perform OCR fallback."

    # ------------------------------------------------------------------
    async def process(self, state: DocumentState) -> DocumentState:  # noqa: D401
        """
        Process the PDF and populate `pages_text`.
        """
        t0 = time.perf_counter()

        pdf_path = state.file_path
        pages_text: List[Dict] = []

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                self.log_info("Opened PDF", pages=total_pages)

                for idx, page in enumerate(pdf.pages, start=1):
                    text = (page.extract_text() or "").strip()
                    if len(text) >= MIN_CHARS:
                        pages_text.append(
                            {
                                "page_num": idx,
                                "text": text,
                                "source": "text",
                                "confidence": 1.0,
                            }
                        )
                        continue  # next page

                    # ── OCR fallback ────────────────────────────────────
                    ocr_text, confidence = self._ocr_page(pdf_path, idx - 1)
                    pages_text.append(
                        {
                            "page_num": idx,
                            "text": ocr_text,
                            "source": "ocr",
                            "confidence": confidence,
                        }
                    )

        except Exception as exc:  # noqa: BLE001
            raise PDFError(
                f"Preprocessing failed for '{pdf_path}': {exc}",
                file_path=pdf_path,
                operation="text_extraction",
                original_exception=exc,
            ) from exc

        # Update state
        state.pages_text = pages_text
        state.status = ProcessingStatus.PROCESSING.value
        state.progress = 0.15  # 15 % done after preprocessing

        # Timing metric
        record_document_processing_time(time.perf_counter() - t0)
        self.log_info("Finished preprocessing", pages=len(pages_text))
        return state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ocr_page(self, pdf_path: str, zero_index: int) -> tuple[str, float]:
        """
        Render a single PDF page to image and run Tesseract OCR.

        Returns (text, confidence)
        """
        start = time.perf_counter()

        # Open page via PyMuPDF
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(zero_index)
            pix = page.get_pixmap(dpi=settings.PDF_DPI if hasattr(settings, "PDF_DPI") else 300)
            img_bytes = pix.tobytes("png")

        # OCR with pytesseract
        increment_ocr_operation()
        try:
            img = Image.open(io.BytesIO(img_bytes))
            ocr_text = (
                pytesseract.image_to_string(
                    img,
                    lang=settings.OCR_LANGUAGE,
                    config="--psm 6"  # Assume uniform block of text
                ).strip()
            )
        except Exception as exc:  # noqa: BLE001
            raise PDFError(
                f"OCR failed on page {zero_index + 1}: {exc}",
                file_path=pdf_path,
                page_number=zero_index + 1,
                operation="ocr",
                original_exception=exc,
            ) from exc

        # Simple confidence heuristic: proportion of ASCII letters
        letters = [c for c in ocr_text if c.isalnum()]
        confidence = min(1.0, max(0.2, len(letters) / max(len(ocr_text), 1)))

        # Metrics
        record_ocr_processing_time(time.perf_counter() - start)
        self.log_debug("OCR page extracted", page=zero_index + 1, confidence=confidence)

        return ocr_text, confidence

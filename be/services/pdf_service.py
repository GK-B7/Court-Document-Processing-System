"""
PDF Processing Service
─────────────────────
Centralized service for PDF text extraction, OCR operations, and document processing.
Provides clean interfaces for PDF manipulation with comprehensive error handling.

Features:
• Text extraction using pdfplumber
• OCR fallback using PyMuPDF + Tesseract
• Image preprocessing for better OCR results
• Page-by-page processing with metadata
• Format validation and error handling
• Performance optimization and caching
"""

from __future__ import annotations

import io
import os
import logging
import tempfile
import time
from typing import List, Dict, Optional, Tuple, Any, BinaryIO
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import cv2
import numpy as np

from config import settings
from exceptions import PDFError, FileError
from metrics import (
    increment_ocr_operation,
    record_ocr_processing_time,
    record_document_processing_time,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# OCR Configuration
DEFAULT_OCR_CONFIG = {
    'lang': getattr(settings, 'OCR_LANGUAGE', 'eng'),
    'psm': getattr(settings, 'OCR_PSM', 6),  # Page segmentation mode
    'oem': getattr(settings, 'OCR_OEM', 3),  # OCR Engine mode
    'timeout': getattr(settings, 'OCR_TIMEOUT', 120),
    'min_confidence': getattr(settings, 'OCR_MIN_CONFIDENCE', 60)
}

# PDF Processing Configuration
PDF_CONFIG = {
    'dpi': getattr(settings, 'PDF_DPI', 300),
    'image_quality': getattr(settings, 'IMAGE_QUALITY', 95),
    'max_pages': getattr(settings, 'MAX_PAGES_PER_DOCUMENT', 100),
    'min_text_length': 25,  # Minimum text length to avoid OCR
    'enable_preprocessing': getattr(settings, 'OCR_PREPROCESS_IMAGE', True),
    'enhance_contrast': getattr(settings, 'OCR_ENHANCE_CONTRAST', True)
}

# ─────────────────────────────────────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Represents content extracted from a single PDF page"""
    page_number: int
    text: str
    extraction_method: str  # "text" or "ocr"
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate page content data"""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.text = self.text.strip()
        
        if not self.metadata:
            self.metadata = {}

@dataclass
class DocumentInfo:
    """Contains document metadata and information"""
    filename: str
    file_size: int
    page_count: int
    file_format: str
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    author: Optional[str] = None
    title: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    encrypted: bool = False
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if not self.metadata:
            self.metadata = {}

@dataclass
class ProcessingResult:
    """Complete result of PDF processing"""
    document_info: DocumentInfo
    pages: List[PageContent]
    processing_stats: Dict[str, Any]
    total_processing_time: float
    success: bool
    error_message: Optional[str] = None

# ─────────────────────────────────────────────────────────────────────────────
# PDF Service Class
# ─────────────────────────────────────────────────────────────────────────────

class PDFService:
    """
    Comprehensive PDF processing service with text extraction and OCR capabilities
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PDF service with configuration
        
        Args:
            config: Optional configuration overrides
        """
        self.config = {**PDF_CONFIG, **(config or {})}
        self.ocr_config = {**DEFAULT_OCR_CONFIG, **(config.get('ocr', {}) if config else {})}
        
        # Validate Tesseract installation
        self._validate_tesseract()
        
        # Initialize statistics
        self.stats = {
            'documents_processed': 0,
            'pages_processed': 0,
            'ocr_operations': 0,
            'total_processing_time': 0.0,
        }
    
    def _validate_tesseract(self) -> None:
        """Validate Tesseract OCR installation"""
        try:
            # Set Tesseract command path if specified
            if hasattr(settings, 'TESSERACT_CMD') and settings.TESSERACT_CMD:
                pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
            
            # Test Tesseract availability
            pytesseract.get_tesseract_version()
            logger.info("Tesseract OCR validated successfully")
            
        except Exception as exc:
            logger.warning(f"Tesseract validation failed: {exc}")
            # Don't raise error - OCR will be disabled if Tesseract is unavailable
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Processing Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    async def process_pdf(self, file_path: str) -> ProcessingResult:
        """
        Process PDF file and extract text from all pages
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Complete processing result with all extracted content
            
        Raises:
            PDFError: If PDF processing fails
            FileError: If file access fails
        """
        start_time = time.perf_counter()
        
        logger.info(f"Starting PDF processing: {file_path}")
        
        try:
            # Validate file
            self._validate_pdf_file(file_path)
            
            # Extract document information
            doc_info = await self._extract_document_info(file_path)
            
            # Check page count limit
            if doc_info.page_count > self.config['max_pages']:
                raise PDFError(
                    f"Document has {doc_info.page_count} pages, exceeds limit of {self.config['max_pages']}",
                    file_path=file_path
                )
            
            # Process all pages
            pages = await self._process_all_pages(file_path, doc_info.page_count)
            
            # Calculate statistics
            processing_time = time.perf_counter() - start_time
            stats = self._calculate_processing_stats(pages, processing_time)
            
            # Update service statistics
            self._update_service_stats(stats)
            
            # Record metrics
            record_document_processing_time(processing_time)
            
            logger.info(
                f"PDF processing completed successfully",
                file_path=file_path,
                pages=len(pages),
                duration=f"{processing_time:.2f}s"
            )
            
            return ProcessingResult(
                document_info=doc_info,
                pages=pages,
                processing_stats=stats,
                total_processing_time=processing_time,
                success=True
            )
            
        except Exception as exc:
            processing_time = time.perf_counter() - start_time
            
            logger.error(f"PDF processing failed: {exc}", file_path=file_path)
            
            # Return failed result with partial information
            return ProcessingResult(
                document_info=DocumentInfo(
                    filename=os.path.basename(file_path),
                    file_size=0,
                    page_count=0,
                    file_format="pdf"
                ),
                pages=[],
                processing_stats={},
                total_processing_time=processing_time,
                success=False,
                error_message=str(exc)
            )
    
    async def extract_text_from_page(self, file_path: str, page_number: int) -> PageContent:
        """
        Extract text from a specific PDF page
        
        Args:
            file_path: Path to PDF file
            page_number: Page number (1-indexed)
            
        Returns:
            Page content with extracted text
            
        Raises:
            PDFError: If page extraction fails
        """
        logger.debug(f"Extracting text from page {page_number}")
        
        try:
            with pdfplumber.open(file_path) as pdf:
                if page_number < 1 or page_number > len(pdf.pages):
                    raise PDFError(
                        f"Page {page_number} out of range (1-{len(pdf.pages)})",
                        file_path=file_path,
                        page_number=page_number
                    )
                
                page = pdf.pages[page_number - 1]  # Convert to 0-indexed
                return await self._process_single_page(file_path, page, page_number)
                
        except Exception as exc:
            raise PDFError(
                f"Failed to extract text from page {page_number}: {exc}",
                file_path=file_path,
                page_number=page_number,
                original_exception=exc
            ) from exc
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal Processing Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _validate_pdf_file(self, file_path: str) -> None:
        """Validate PDF file exists and is accessible"""
        if not os.path.exists(file_path):
            raise FileError(f"PDF file not found: {file_path}", file_path=file_path)
        
        if not os.path.isfile(file_path):
            raise FileError(f"Path is not a file: {file_path}", file_path=file_path)
        
        # Check file size
        file_size = os.path.getsize(file_path)
        max_size = getattr(settings, 'MAX_FILE_SIZE', 10 * 1024 * 1024)  # 10MB default
        
        if file_size > max_size:
            raise FileError(
                f"File too large: {file_size} bytes (max: {max_size})",
                file_path=file_path,
                file_size=file_size,
                max_size=max_size
            )
        
        # Try to open with pdfplumber to validate format
        try:
            with pdfplumber.open(file_path) as pdf:
                if len(pdf.pages) == 0:
                    raise PDFError("PDF contains no pages", file_path=file_path)
        except Exception as exc:
            raise PDFError(
                f"Invalid or corrupted PDF file: {exc}",
                file_path=file_path,
                original_exception=exc
            ) from exc
    
    async def _extract_document_info(self, file_path: str) -> DocumentInfo:
        """Extract document metadata and information"""
        try:
            file_stats = os.stat(file_path)
            
            with pdfplumber.open(file_path) as pdf:
                metadata = pdf.metadata or {}
                
                # Extract basic information
                doc_info = DocumentInfo(
                    filename=os.path.basename(file_path),
                    file_size=file_stats.st_size,
                    page_count=len(pdf.pages),
                    file_format="pdf",
                    creation_date=metadata.get('CreationDate'),
                    modification_date=metadata.get('ModDate'),
                    author=metadata.get('Author'),
                    title=metadata.get('Title'),
                    subject=metadata.get('Subject'),
                    keywords=metadata.get('Keywords', '').split(',') if metadata.get('Keywords') else None,
                    encrypted=pdf.is_encrypted,
                    metadata=metadata
                )
                
                logger.debug(f"Extracted document info: {doc_info.page_count} pages")
                return doc_info
                
        except Exception as exc:
            raise PDFError(
                f"Failed to extract document information: {exc}",
                file_path=file_path,
                original_exception=exc
            ) from exc
    
    async def _process_all_pages(self, file_path: str, page_count: int) -> List[PageContent]:
        """Process all pages in the PDF"""
        pages = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, 1):
                    page_content = await self._process_single_page(file_path, page, i)
                    pages.append(page_content)
                    
                    logger.debug(f"Processed page {i}/{page_count}")
            
            return pages
            
        except Exception as exc:
            raise PDFError(
                f"Failed to process PDF pages: {exc}",
                file_path=file_path,
                original_exception=exc
            ) from exc
    
    async def _process_single_page(
        self, 
        file_path: str, 
        page: Any, 
        page_number: int
    ) -> PageContent:
        """Process a single PDF page with text extraction and OCR fallback"""
        start_time = time.perf_counter()
        
        try:
            # First, try text extraction
            text = (page.extract_text() or "").strip()
            
            if len(text) >= self.config['min_text_length']:
                # Sufficient text extracted directly
                processing_time = time.perf_counter() - start_time
                
                return PageContent(
                    page_number=page_number,
                    text=text,
                    extraction_method="text",
                    confidence=1.0,  # High confidence for direct text extraction
                    processing_time=processing_time,
                    metadata={
                        'direct_extraction': True,
                        'text_length': len(text),
                        'bbox': page.bbox if hasattr(page, 'bbox') else None
                    }
                )
            else:
                # Insufficient text - use OCR
                return await self._ocr_page(file_path, page_number - 1, start_time)
                
        except Exception as exc:
            # If text extraction fails, try OCR
            logger.warning(f"Text extraction failed for page {page_number}, trying OCR: {exc}")
            try:
                return await self._ocr_page(file_path, page_number - 1, start_time)
            except Exception as ocr_exc:
                # Both methods failed
                processing_time = time.perf_counter() - start_time
                
                return PageContent(
                    page_number=page_number,
                    text="",
                    extraction_method="failed",
                    confidence=0.0,
                    processing_time=processing_time,
                    metadata={
                        'text_extraction_error': str(exc),
                        'ocr_error': str(ocr_exc),
                        'failed': True
                    }
                )
    
    async def _ocr_page(
        self, 
        file_path: str, 
        zero_indexed_page: int, 
        start_time: Optional[float] = None
    ) -> PageContent:
        """Perform OCR on a specific page"""
        if start_time is None:
            start_time = time.perf_counter()
        
        ocr_start = time.perf_counter()
        page_number = zero_indexed_page + 1
        
        try:
            increment_ocr_operation()
            
            # Render page to image
            image_bytes = await self._render_page_to_image(file_path, zero_indexed_page)
            
            # Preprocess image if enabled
            if self.config['enable_preprocessing']:
                image_bytes = await self._preprocess_image(image_bytes)
            
            # Perform OCR
            ocr_text, confidence = await self._perform_ocr(image_bytes)
            
            # Record OCR timing
            ocr_time = time.perf_counter() - ocr_start
            total_time = time.perf_counter() - start_time
            record_ocr_processing_time(ocr_time)
            
            logger.debug(
                f"OCR completed for page {page_number}",
                confidence=confidence,
                text_length=len(ocr_text),
                ocr_time=f"{ocr_time:.2f}s"
            )
            
            return PageContent(
                page_number=page_number,
                text=ocr_text,
                extraction_method="ocr",
                confidence=confidence,
                processing_time=total_time,
                metadata={
                    'ocr_time': ocr_time,
                    'image_preprocessing': self.config['enable_preprocessing'],
                    'ocr_config': self.ocr_config,
                    'text_length': len(ocr_text)
                }
            )
            
        except Exception as exc:
            total_time = time.perf_counter() - start_time
            
            raise PDFError(
                f"OCR failed for page {page_number}: {exc}",
                file_path=file_path,
                page_number=page_number,
                operation="ocr",
                original_exception=exc
            ) from exc
    
    async def _render_page_to_image(self, file_path: str, page_index: int) -> bytes:
        """Render PDF page to image bytes"""
        try:
            with fitz.open(file_path) as doc:
                page = doc.load_page(page_index)
                
                # Render page with specified DPI
                pix = page.get_pixmap(dpi=self.config['dpi'])
                image_bytes = pix.tobytes("png")
                
                logger.debug(f"Rendered page {page_index + 1} to image ({len(image_bytes)} bytes)")
                return image_bytes
                
        except Exception as exc:
            raise PDFError(
                f"Failed to render page {page_index + 1} to image: {exc}",
                file_path=file_path,
                page_number=page_index + 1,
                operation="image_rendering",
                original_exception=exc
            ) from exc
    
    async def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """Preprocess image to improve OCR accuracy"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale if not already
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast if enabled
            if self.config['enhance_contrast']:
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.5)  # Increase contrast by 50%
            
            # Apply noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to OpenCV format for additional processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            
            # Apply morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY))
            
            # Save to bytes
            output_buffer = io.BytesIO()
            processed_image.save(output_buffer, format='PNG', quality=self.config['image_quality'])
            
            processed_bytes = output_buffer.getvalue()
            logger.debug(f"Image preprocessing completed ({len(processed_bytes)} bytes)")
            
            return processed_bytes
            
        except Exception as exc:
            logger.warning(f"Image preprocessing failed, using original: {exc}")
            return image_bytes  # Return original if preprocessing fails
    
    async def _perform_ocr(self, image_bytes: bytes) -> Tuple[str, float]:
        """Perform OCR on image bytes"""
        try:
            # Load image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Build Tesseract configuration
            config = f"--psm {self.ocr_config['psm']} --oem {self.ocr_config['oem']}"
            
            # Perform OCR
            ocr_text = pytesseract.image_to_string(
                image,
                lang=self.ocr_config['lang'],
                config=config,
                timeout=self.ocr_config['timeout']
            ).strip()
            
            # Calculate confidence score
            confidence = self._calculate_ocr_confidence(ocr_text, image_bytes)
            
            return ocr_text, confidence
            
        except pytesseract.TesseractError as exc:
            raise PDFError(
                f"Tesseract OCR error: {exc}",
                operation="ocr",
                original_exception=exc
            ) from exc
        except Exception as exc:
            raise PDFError(
                f"OCR processing error: {exc}",
                operation="ocr",
                original_exception=exc
            ) from exc
    
    def _calculate_ocr_confidence(self, text: str, image_bytes: bytes) -> float:
        """Calculate confidence score for OCR results"""
        if not text:
            return 0.0
        
        try:
            # Simple heuristic based on text characteristics
            total_chars = len(text)
            if total_chars == 0:
                return 0.0
            
            # Count alphanumeric characters
            alnum_chars = sum(1 for c in text if c.isalnum())
            alnum_ratio = alnum_chars / total_chars
            
            # Count whitespace characters
            space_chars = sum(1 for c in text if c.isspace())
            space_ratio = space_chars / total_chars
            
            # Calculate confidence based on character distribution
            # Good OCR should have reasonable ratios of alphanumeric and space characters
            confidence = min(1.0, alnum_ratio + (space_ratio * 0.5))
            
            # Apply minimum confidence floor
            confidence = max(0.2, confidence)
            
            return confidence
            
        except Exception:
            # Fallback to moderate confidence if calculation fails
            return 0.6
    
    # ─────────────────────────────────────────────────────────────────────────
    # Statistics and Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_processing_stats(
        self, 
        pages: List[PageContent], 
        total_time: float
    ) -> Dict[str, Any]:
        """Calculate comprehensive processing statistics"""
        if not pages:
            return {}
        
        text_extractions = [p for p in pages if p.extraction_method == "text"]
        ocr_extractions = [p for p in pages if p.extraction_method == "ocr"]
        failed_extractions = [p for p in pages if p.extraction_method == "failed"]
        
        total_text_length = sum(len(p.text) for p in pages)
        avg_confidence = sum(p.confidence for p in pages) / len(pages)
        avg_processing_time = sum(p.processing_time for p in pages) / len(pages)
        
        return {
            'total_pages': len(pages),
            'text_extractions': len(text_extractions),
            'ocr_extractions': len(ocr_extractions),
            'failed_extractions': len(failed_extractions),
            'total_text_length': total_text_length,
            'average_confidence': avg_confidence,
            'average_processing_time_per_page': avg_processing_time,
            'total_processing_time': total_time,
            'pages_per_second': len(pages) / total_time if total_time > 0 else 0,
            'success_rate': (len(pages) - len(failed_extractions)) / len(pages),
            'ocr_usage_rate': len(ocr_extractions) / len(pages),
            'extraction_methods': {
                'text': len(text_extractions),
                'ocr': len(ocr_extractions),
                'failed': len(failed_extractions)
            }
        }
    
    def _update_service_stats(self, processing_stats: Dict[str, Any]) -> None:
        """Update service-level statistics"""
        self.stats['documents_processed'] += 1
        self.stats['pages_processed'] += processing_stats.get('total_pages', 0)
        self.stats['ocr_operations'] += processing_stats.get('ocr_extractions', 0)
        self.stats['total_processing_time'] += processing_stats.get('total_processing_time', 0)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get current service statistics"""
        return {
            **self.stats,
            'average_processing_time_per_document': (
                self.stats['total_processing_time'] / self.stats['documents_processed']
                if self.stats['documents_processed'] > 0 else 0
            ),
            'average_pages_per_document': (
                self.stats['pages_processed'] / self.stats['documents_processed']
                if self.stats['documents_processed'] > 0 else 0
            ),
            'ocr_usage_percentage': (
                (self.stats['ocr_operations'] / self.stats['pages_processed']) * 100
                if self.stats['pages_processed'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset service statistics"""
        self.stats = {
            'documents_processed': 0,
            'pages_processed': 0,
            'ocr_operations': 0,
            'total_processing_time': 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global Service Instance and Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

# Global PDF service instance
pdf_service = PDFService()

async def extract_text_from_pdf(file_path: str) -> ProcessingResult:
    """
    Convenience function to extract text from PDF
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Complete processing result
    """
    return await pdf_service.process_pdf(file_path)

async def extract_text_from_page(file_path: str, page_number: int) -> PageContent:
    """
    Convenience function to extract text from specific page
    
    Args:
        file_path: Path to PDF file
        page_number: Page number (1-indexed)
        
    Returns:
        Page content with extracted text
    """
    return await pdf_service.extract_text_from_page(file_path, page_number)

def validate_pdf_file(file_path: str) -> bool:
    """
    Validate if file is a valid PDF
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid PDF, False otherwise
    """
    try:
        pdf_service._validate_pdf_file(file_path)
        return True
    except (PDFError, FileError):
        return False

def get_pdf_service_stats() -> Dict[str, Any]:
    """Get PDF service statistics"""
    return pdf_service.get_service_stats()

# Export main classes and functions
__all__ = [
    'PDFService',
    'PageContent',
    'DocumentInfo',
    'ProcessingResult',
    'pdf_service',
    'extract_text_from_pdf',
    'extract_text_from_page',
    'validate_pdf_file',
    'get_pdf_service_stats'
]

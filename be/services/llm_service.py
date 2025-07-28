"""
LLM Service for Document Processing
──────────────────────────────────
Centralized service for Large Language Model operations including text processing,
extraction, embedding generation, and chat completions using OpenAI GPT models.

Features:
• GPT-4o integration for structured text extraction
• Embedding generation using OpenAI embedding models
• Token counting and cost estimation
• Rate limiting and quota management
• Retry logic with exponential backoff
• Response caching for efficiency
• Comprehensive error handling and logging
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import re
from collections import defaultdict

from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion
from openai.types import CreateEmbeddingResponse
import tiktoken

from config import settings
from exceptions import LLMError, ValidationError
from metrics import (
    increment_llm_api_call,
    record_llm_response_time,
)

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(settings.LOG_LEVEL)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration and Constants
# ─────────────────────────────────────────────────────────────────────────────

# OpenAI Configuration
OPENAI_CONFIG = {
    'api_key': settings.OPENAI_API_KEY,
    'model': getattr(settings, 'OPENAI_MODEL', 'gpt-4o'),
    'embedding_model': getattr(settings, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
    'max_tokens': getattr(settings, 'OPENAI_MAX_TOKENS', 4000),
    'temperature': getattr(settings, 'OPENAI_TEMPERATURE', 0.1),
    'timeout': getattr(settings, 'OPENAI_TIMEOUT', 60),
    'max_retries': 3,
    'retry_delay': 2.0
}

# Rate limiting configuration
RATE_LIMITS = {
    'requests_per_minute': getattr(settings, 'OPENAI_MAX_REQUESTS_PER_MINUTE', 60),
    'tokens_per_minute': getattr(settings, 'OPENAI_MAX_TOKENS_PER_MINUTE', 150000),
    'concurrent_requests': 10
}

# Token limits for different models
MODEL_TOKEN_LIMITS = {
    'gpt-4o': 128000,
    'gpt-4': 8192,
    'gpt-3.5-turbo': 4096,
    'text-embedding-3-small': 8191,
    'text-embedding-3-large': 8191
}

# ─────────────────────────────────────────────────────────────────────────────
# Enums and Data Classes
# ─────────────────────────────────────────────────────────────────────────────

class LLMOperation(Enum):
    """Types of LLM operations"""
    CHAT_COMPLETION = "chat_completion"
    EMBEDDING = "embedding"
    TEXT_EXTRACTION = "text_extraction"
    CLASSIFICATION = "classification"

@dataclass
class TokenUsage:
    """Token usage statistics"""
    prompt_tokens: int
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.prompt_tokens + self.completion_tokens

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model: str
    operation: LLMOperation
    token_usage: TokenUsage
    response_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'content': self.content,
            'model': self.model,
            'operation': self.operation.value,
            'token_usage': {
                'prompt_tokens': self.token_usage.prompt_tokens,
                'completion_tokens': self.token_usage.completion_tokens,
                'total_tokens': self.token_usage.total_tokens
            },
            'response_time': self.response_time,
            'success': self.success,
            'error_message': self.error_message,
            'metadata': self.metadata
        }

@dataclass
class ExtractionRequest:
    """Request for structured data extraction"""
    text: str
    extraction_type: str  # "id_action_pairs", "entities", etc.
    context: Optional[str] = None
    page_number: Optional[int] = None
    additional_instructions: Optional[str] = None
    confidence_threshold: float = 0.7

@dataclass
class ExtractionResult:
    """Result of structured data extraction"""
    extracted_data: List[Dict[str, Any]]
    confidence: float
    extraction_type: str
    source_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ─────────────────────────────────────────────────────────────────────────────
# Rate Limiting and Caching
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self):
        self.request_times = []
        self.token_usage = []
        self.lock = asyncio.Lock()
    
    async def check_rate_limit(self, estimated_tokens: int = 0) -> None:
        """Check if request can be made within rate limits"""
        async with self.lock:
            current_time = time.time()
            
            # Remove old entries (beyond 1 minute)
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if current_time - t < 60]
            
            # Check request rate limit
            if len(self.request_times) >= RATE_LIMITS['requests_per_minute']:
                wait_time = 60 - (current_time - self.request_times[0])
                if wait_time > 0:
                    logger.warning(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Check token rate limit
            total_tokens = sum(tokens for _, tokens in self.token_usage) + estimated_tokens
            if total_tokens >= RATE_LIMITS['tokens_per_minute']:
                # Wait for oldest tokens to expire
                if self.token_usage:
                    wait_time = 60 - (current_time - self.token_usage[0][0])
                    if wait_time > 0:
                        logger.warning(f"Token rate limit reached, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(current_time)
            if estimated_tokens > 0:
                self.token_usage.append((current_time, estimated_tokens))

class ResponseCache:
    """Simple in-memory response cache"""
    
    def __init__(self, max_size: int = 1000):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.max_size = max_size
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, operation: str, **kwargs) -> str:
        """Generate cache key from operation and parameters"""
        # Create deterministic key from operation and sorted kwargs
        key_data = f"{operation}:{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, operation: str, ttl: int = 3600, **kwargs) -> Optional[Any]:
        """Get cached response if available and not expired"""
        key = self._generate_key(operation, **kwargs)
        
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < ttl:
                self.access_times[key] = time.time()
                return value
            else:
                # Expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
        
        return None
    
    def set(self, operation: str, value: Any, **kwargs) -> None:
        """Cache response value"""
        key = self._generate_key(operation, **kwargs)
        
        # Evict oldest entries if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        current_time = time.time()
        self.cache[key] = (value, current_time)
        self.access_times[key] = current_time

# ─────────────────────────────────────────────────────────────────────────────
# LLM Service Class
# ─────────────────────────────────────────────────────────────────────────────

class LLMService:
    """
    Comprehensive service for Large Language Model operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLM service with configuration
        
        Args:
            config: Optional configuration overrides
        """
        self.config = {**OPENAI_CONFIG, **(config or {})}
        
        # Initialize OpenAI clients
        self.client = OpenAI(api_key=self.config['api_key'])
        self.async_client = AsyncOpenAI(api_key=self.config['api_key'])
        
        # Initialize rate limiter and cache
        self.rate_limiter = RateLimiter()
        self.cache = ResponseCache()
        
        # Initialize token encoder for counting
        try:
            self.token_encoder = tiktoken.encoding_for_model(self.config['model'])
        except KeyError:
            # Fallback to cl100k_base encoding
            self.token_encoder = tiktoken.get_encoding("cl100k_base")
        
        # Statistics tracking
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost_usd': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'operations': defaultdict(int)
        }
        
        logger.info("LLM Service initialized successfully")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main API Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    async def extract_structured_data(self, request: ExtractionRequest) -> LLMResponse:
        """
        Extract structured data from text using GPT models
        
        Args:
            request: Extraction request with text and parameters
            
        Returns:
            LLM response with extracted structured data
        """
        start_time = time.perf_counter()
        
        try:
            # Check cache first
            cache_key = self._generate_extraction_cache_key(request)
            cached_response = self.cache.get("extract_structured_data", text=request.text, 
                                           extraction_type=request.extraction_type)
            
            if cached_response:
                self.stats['cache_hits'] += 1
                logger.debug("Cache hit for extraction request")
                return cached_response
            
            self.stats['cache_misses'] += 1
            
            # Build prompt based on extraction type
            prompt = self._build_extraction_prompt(request)
            
            # Estimate tokens and check rate limits
            estimated_tokens = self.count_tokens(prompt)
            await self.rate_limiter.check_rate_limit(estimated_tokens)
            
            # Make API call
            response = await self._make_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a precise document extraction specialist. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                operation=LLMOperation.TEXT_EXTRACTION
            )
            
            # Parse structured response
            extracted_data = self._parse_extraction_response(response.content, request.extraction_type)
            
            # Create structured response
            llm_response = LLMResponse(
                content=json.dumps(extracted_data, indent=2),
                model=self.config['model'],
                operation=LLMOperation.TEXT_EXTRACTION,
                token_usage=response.token_usage,
                response_time=time.perf_counter() - start_time,
                success=True,
                metadata={
                    'extraction_type': request.extraction_type,
                    'extracted_items_count': len(extracted_data.get('extractions', [])),
                    'confidence': extracted_data.get('confidence', 0.0),
                    'page_number': request.page_number
                }
            )
            
            # Cache successful response
            self.cache.set("extract_structured_data", llm_response, 
                         text=request.text, extraction_type=request.extraction_type)
            
            # Update statistics
            self._update_stats(llm_response)
            
            return llm_response
            
        except Exception as exc:
            response_time = time.perf_counter() - start_time
            
            error_response = LLMResponse(
                content="",
                model=self.config['model'],
                operation=LLMOperation.TEXT_EXTRACTION,
                token_usage=TokenUsage(prompt_tokens=0),
                response_time=response_time,
                success=False,
                error_message=str(exc)
            )
            
            self._update_stats(error_response)
            
            raise LLMError(
                f"Structured data extraction failed: {exc}",
                model_name=self.config['model'],
                api_call_type="extraction",
                original_exception=exc
            ) from exc
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        start_time = time.perf_counter()
        
        try:
            # Check cache for batch
            cache_key = f"embeddings_{hashlib.md5(json.dumps(texts, sort_keys=True).encode()).hexdigest()}"
            cached_embeddings = self.cache.get("generate_embeddings", cache_key=cache_key)
            
            if cached_embeddings:
                self.stats['cache_hits'] += 1
                return cached_embeddings
            
            self.stats['cache_misses'] += 1
            
            # Estimate tokens for all texts
            total_tokens = sum(self.count_tokens(text) for text in texts)
            await self.rate_limiter.check_rate_limit(total_tokens)
            
            # Make embedding API call
            increment_llm_api_call()
            
            response: CreateEmbeddingResponse = await self.async_client.embeddings.create(
                model=self.config['embedding_model'],
                input=texts,
                timeout=self.config['timeout']
            )
            
            # Extract embeddings
            embeddings = [data.embedding for data in response.data]
            
            # Record metrics
            response_time = time.perf_counter() - start_time
            record_llm_response_time(response_time)
            
            # Update statistics
            self.stats['total_requests'] += 1
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += response.usage.total_tokens
            self.stats['operations'][LLMOperation.EMBEDDING.value] += 1
            
            # Cache result
            self.cache.set("generate_embeddings", embeddings, cache_key=cache_key)
            
            logger.debug(f"Generated embeddings for {len(texts)} texts in {response_time:.2f}s")
            
            return embeddings
            
        except Exception as exc:
            self.stats['total_requests'] += 1
            self.stats['failed_requests'] += 1
            
            raise LLMError(
                f"Embedding generation failed: {exc}",
                model_name=self.config['embedding_model'],
                api_call_type="embedding",
                original_exception=exc
            ) from exc
    
    async def chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate chat completion using GPT models
        
        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the API call
            
        Returns:
            LLM response with completion
        """
        try:
            # Add system prompt if provided
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Make API call
            return await self._make_chat_completion(messages, LLMOperation.CHAT_COMPLETION, **kwargs)
            
        except Exception as exc:
            raise LLMError(
                f"Chat completion failed: {exc}",
                model_name=self.config['model'],
                api_call_type="completion",
                original_exception=exc
            ) from exc
    
    # ─────────────────────────────────────────────────────────────────────────
    # Internal Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    async def _make_chat_completion(
        self, 
        messages: List[Dict[str, str]], 
        operation: LLMOperation,
        **kwargs
    ) -> LLMResponse:
        """Make chat completion API call with retry logic"""
        start_time = time.perf_counter()
        
        # Estimate tokens
        prompt_tokens = sum(self.count_tokens(msg['content']) for msg in messages)
        await self.rate_limiter.check_rate_limit(prompt_tokens)
        
        last_exception = None
        
        for attempt in range(self.config['max_retries']):
            try:
                increment_llm_api_call()
                
                # Make API call
                response: ChatCompletion = await self.async_client.chat.completions.create(
                    model=self.config['model'],
                    messages=messages,
                    temperature=kwargs.get('temperature', self.config['temperature']),
                    max_tokens=kwargs.get('max_tokens', self.config['max_tokens']),
                    timeout=self.config['timeout']
                )
                
                # Extract response content
                content = response.choices[0].message.content or ""
                
                # Create token usage object
                token_usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens
                )
                
                # Record timing
                response_time = time.perf_counter() - start_time
                record_llm_response_time(response_time)
                
                # Create response object
                llm_response = LLMResponse(
                    content=content,
                    model=self.config['model'],
                    operation=operation,
                    token_usage=token_usage,
                    response_time=response_time,
                    success=True,
                    metadata={
                        'finish_reason': response.choices[0].finish_reason,
                        'attempt': attempt + 1
                    }
                )
                
                return llm_response
                
            except Exception as exc:
                last_exception = exc
                logger.warning(f"API call attempt {attempt + 1} failed: {exc}")
                
                if attempt < self.config['max_retries'] - 1:
                    # Wait before retry with exponential backoff
                    wait_time = self.config['retry_delay'] * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    break
        
        # All retries failed
        response_time = time.perf_counter() - start_time
        
        raise LLMError(
            f"API call failed after {self.config['max_retries']} attempts: {last_exception}",
            model_name=self.config['model'],
            api_call_type=operation.value,
            original_exception=last_exception
        )
    
    def _build_extraction_prompt(self, request: ExtractionRequest) -> str:
        """Build extraction prompt based on request type"""
        if request.extraction_type == "id_action_pairs":
            return self._build_id_action_extraction_prompt(request)
        else:
            return self._build_generic_extraction_prompt(request)
    
    def _build_id_action_extraction_prompt(self, request: ExtractionRequest) -> str:
        """Build prompt for ID/Action pair extraction"""
        base_prompt = f"""
You are an expert document processor specializing in financial documents. Your task is to extract National ID numbers and associated actions from document text.

INSTRUCTIONS:
1. Look for National ID numbers (typically 10-20 digit numbers)
2. Identify actions to be taken on customer accounts (freeze, release, close, suspend, etc.)
3. Match each National ID with its corresponding action
4. Provide confidence scores (0.0 to 1.0) for each extraction
5. Return results in the specified JSON format

CONTEXT:
- This is page {request.page_number or 'unknown'} of a financial document
- Look for patterns like: "ID: 1234567890 - Freeze account" or "Customer 0987654321 should have funds released"

DOCUMENT TEXT:
{request.text}

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
        
        if request.additional_instructions:
            base_prompt += f"\n\nADDITIONAL INSTRUCTIONS:\n{request.additional_instructions}"
        
        return base_prompt
    
    def _build_generic_extraction_prompt(self, request: ExtractionRequest) -> str:
        """Build generic extraction prompt"""
        return f"""
Extract structured data of type "{request.extraction_type}" from the following text.

Text:
{request.text}

Context: {request.context or 'None provided'}

Return the extracted data as a JSON object with the following structure:
{{
    "extracted_data": [...],
    "confidence": 0.0-1.0,
    "extraction_type": "{request.extraction_type}",
    "metadata": {{}}
}}

Additional instructions: {request.additional_instructions or 'None'}
"""
    
    def _parse_extraction_response(self, response_content: str, extraction_type: str) -> Dict[str, Any]:
        """Parse LLM response into structured format"""
        try:
            # First try to parse as JSON directly
            try:
                return json.loads(response_content)
            except json.JSONDecodeError:
                # Try to extract JSON from response text
                json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found in response")
        
        except Exception as exc:
            logger.error(f"Failed to parse extraction response: {exc}")
            
            # Return empty result structure
            if extraction_type == "id_action_pairs":
                return {
                    "extractions": [],
                    "page_confidence": 0.0,
                    "notes": f"Failed to parse response: {exc}",
                    "raw_response": response_content
                }
            else:
                return {
                    "extracted_data": [],
                    "confidence": 0.0,
                    "extraction_type": extraction_type,
                    "metadata": {"parse_error": str(exc), "raw_response": response_content}
                }
    
    def _generate_extraction_cache_key(self, request: ExtractionRequest) -> str:
        """Generate cache key for extraction request"""
        key_data = {
            'text': request.text[:500],  # Use first 500 chars to avoid huge keys
            'extraction_type': request.extraction_type,
            'context': request.context,
            'additional_instructions': request.additional_instructions
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _update_stats(self, response: LLMResponse) -> None:
        """Update service statistics"""
        self.stats['total_requests'] += 1
        
        if response.success:
            self.stats['successful_requests'] += 1
            self.stats['total_tokens_used'] += response.token_usage.total_tokens
            
            # Estimate cost (rough approximation)
            cost_per_token = 0.00002  # Approximate cost per token for GPT-4
            self.stats['total_cost_usd'] += response.token_usage.total_tokens * cost_per_token
        else:
            self.stats['failed_requests'] += 1
        
        self.stats['operations'][response.operation.value] += 1
        
        # Update average response time
        total_responses = self.stats['successful_requests'] + self.stats['failed_requests']
        current_avg = self.stats['average_response_time']
        self.stats['average_response_time'] = (
            (current_avg * (total_responses - 1) + response.response_time) / total_responses
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utility Methods
    # ─────────────────────────────────────────────────────────────────────────
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using the model's tokenizer"""
        try:
            return len(self.token_encoder.encode(text))
        except Exception:
            # Fallback to rough estimation
            return len(text.split()) * 1.3  # Rough approximation
    
    def estimate_cost(self, tokens: int, operation: LLMOperation) -> float:
        """Estimate cost for token usage"""
        # Rough cost estimates (update with current pricing)
        cost_per_token = {
            LLMOperation.CHAT_COMPLETION: 0.00002,
            LLMOperation.TEXT_EXTRACTION: 0.00002,
            LLMOperation.EMBEDDING: 0.000001,
            LLMOperation.CLASSIFICATION: 0.00002
        }
        
        return tokens * cost_per_token.get(operation, 0.00002)
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get current service statistics"""
        return {
            **self.stats,
            'success_rate': (
                self.stats['successful_requests'] / self.stats['total_requests']
                if self.stats['total_requests'] > 0 else 0
            ),
            'cache_hit_rate': (
                self.stats['cache_hits'] / (self.stats['cache_hits'] + self.stats['cache_misses'])
                if (self.stats['cache_hits'] + self.stats['cache_misses']) > 0 else 0
            ),
            'average_tokens_per_request': (
                self.stats['total_tokens_used'] / self.stats['successful_requests']
                if self.stats['successful_requests'] > 0 else 0
            )
        }
    
    def reset_stats(self) -> None:
        """Reset service statistics"""
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_tokens_used': 0,
            'total_cost_usd': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_response_time': 0.0,
            'operations': defaultdict(int)
        }


# ─────────────────────────────────────────────────────────────────────────────
# Global Service Instance and Utility Functions
# ─────────────────────────────────────────────────────────────────────────────

# Global LLM service instance
llm_service = LLMService()

async def extract_id_action_pairs(
    text: str, 
    page_number: Optional[int] = None,
    context: Optional[str] = None
) -> LLMResponse:
    """
    Convenience function to extract ID/Action pairs from text
    
    Args:
        text: Text to extract from
        page_number: Optional page number for context
        context: Optional additional context
        
    Returns:
        LLM response with extracted ID/Action pairs
    """
    request = ExtractionRequest(
        text=text,
        extraction_type="id_action_pairs",
        page_number=page_number,
        context=context
    )
    
    return await llm_service.extract_structured_data(request)

async def generate_text_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Convenience function to generate embeddings
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    return await llm_service.generate_embeddings(texts)

async def chat_with_gpt(
    messages: List[Dict[str, str]], 
    system_prompt: Optional[str] = None
) -> LLMResponse:
    """
    Convenience function for chat completions
    
    Args:
        messages: List of message dictionaries
        system_prompt: Optional system prompt
        
    Returns:
        LLM response with completion
    """
    return await llm_service.chat_completion(messages, system_prompt)

def count_text_tokens(text: str) -> int:
    """
    Convenience function to count tokens in text
    
    Args:
        text: Text to count tokens for
        
    Returns:
        Number of tokens
    """
    return llm_service.count_tokens(text)

def get_llm_service_stats() -> Dict[str, Any]:
    """Get LLM service statistics"""
    return llm_service.get_service_stats()

# Export main classes and functions
__all__ = [
    'LLMService',
    'LLMResponse',
    'ExtractionRequest',
    'ExtractionResult',
    'TokenUsage',
    'LLMOperation',
    'llm_service',
    'extract_id_action_pairs',
    'generate_text_embeddings',
    'chat_with_gpt',
    'count_text_tokens',
    'get_llm_service_stats'
]

"""
Configuration management for the Document Processing API
Uses pydantic-settings for environment variable management
"""

import os
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables
    """
    
    # Application settings
    APP_NAME: str = Field(default="Document Processing API")
    APP_VERSION: str = Field(default="1.0.0")
    DEBUG: bool = Field(default=True)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="./logs/app.log")
    
    # Database settings
    DATABASE_HOST: str = Field(default="localhost")
    DATABASE_PORT: int = Field(default=5432)
    DATABASE_NAME: str = Field(default="doc_processing")
    DATABASE_USER: str = Field(default="postgres")
    DATABASE_PASSWORD: str = Field(default="password")
    
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o")
    OPENAI_MAX_TOKENS: int = Field(default=4000)
    OPENAI_TEMPERATURE: float = Field(default=0.1)
    OPENAI_TIMEOUT: int = Field(default=60)
    OPENAI_MAX_REQUESTS_PER_MINUTE: int = Field(default=60)
    OPENAI_MAX_TOKENS_PER_MINUTE: int = Field(default=150000)
    
    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db")
    CHROMA_COLLECTION_NAME: str = Field(default="actions")
    SIMILARITY_THRESHOLD: float = Field(default=0.8)
    MAX_SEARCH_RESULTS: int = Field(default=5)
    
    # File upload settings
    UPLOAD_DIRECTORY: str = Field(default="./uploads")
    MAX_FILE_SIZE: int = Field(default=10485760)
    ALLOWED_FILE_TYPES: Union[List[str], str] = Field(default=".pdf,.docx,.doc,.txt")
    TEMP_DIRECTORY: str = Field(default="./temp")
    CLEANUP_TEMP_FILES: bool = Field(default=True)
    
    # OCR settings
    OCR_LANGUAGE: str = Field(default="eng")
    
    # Processing settings
    CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    REVIEW_THRESHOLD: float = Field(default=0.6)
    AUTO_APPROVE_THRESHOLD: float = Field(default=0.8)
    MAX_RETRIES: int = Field(default=3)
    RETRY_DELAY_SECONDS: int = Field(default=5)
    
    # Job processing settings
    MAX_CONCURRENT_JOBS: int = Field(default=5)
    JOB_TIMEOUT_MINUTES: int = Field(default=30)
    AGENT_TIMEOUT_SECONDS: int = Field(default=300)
    ENABLE_AGENT_LOGGING: bool = Field(default=True)
    AGENT_LOG_LEVEL: str = Field(default="INFO")
    
    # Customer validation settings
    NATIONAL_ID_PATTERN: str = Field(default=r'^\d{10,20}$')
    
    # CORS settings
    CORS_ORIGINS: Union[List[str], str] = Field(default="http://localhost:3000")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: Union[List[str], str] = Field(default='["*"]')
    CORS_ALLOW_HEADERS: Union[List[str], str] = Field(default='["*"]')
    
    # Business logic settings
    SUPPORTED_LANGUAGES: Union[List[str], str] = Field(default='["en", "es", "fr", "de"]')
    SUPPORTED_ACTIONS: Union[List[str], str] = Field(default='["freeze_funds", "release_funds"]')
    DEFAULT_ACTION_TIMEOUT: int = Field(default=300)
    
# Create global settings instance
try:
    settings = Settings()
except Exception as e:
    print(f"Settings error: {e}")
    # Create minimal settings for development
    settings = Settings(
        OPENAI_API_KEY="temp-key",
        SECRET_KEY="temp-secret"
    )
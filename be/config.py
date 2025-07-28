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
    DATABASE_MIN_CONNECTIONS: int = Field(default=5)
    DATABASE_MAX_CONNECTIONS: int = Field(default=20)
    DATABASE_POOL_TIMEOUT: int = Field(default=30)
    
    # OpenAI settings
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4o")
    OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")
    OPENAI_MAX_TOKENS: int = Field(default=4000)
    OPENAI_TEMPERATURE: float = Field(default=0.1)
    OPENAI_TIMEOUT: int = Field(default=60)
    OPENAI_MAX_REQUESTS_PER_MINUTE: int = Field(default=60)
    OPENAI_MAX_TOKENS_PER_MINUTE: int = Field(default=150000)
    
    # ChromaDB settings
    CHROMA_PERSIST_DIRECTORY: str = Field(default="./chroma_db")
    CHROMA_COLLECTION_NAME: str = Field(default="actions")
    CHROMA_HOST: str = Field(default="localhost")
    CHROMA_PORT: int = Field(default=8001)
    SIMILARITY_THRESHOLD: float = Field(default=0.8)
    MAX_SEARCH_RESULTS: int = Field(default=5)
    
    # File upload settings
    UPLOAD_DIRECTORY: str = Field(default="./uploads")
    MAX_FILE_SIZE: int = Field(default=10485760)
    ALLOWED_FILE_TYPES: Union[List[str], str] = Field(default=".pdf,.docx,.doc,.txt")
    TEMP_DIRECTORY: str = Field(default="./temp")
    CLEANUP_TEMP_FILES: bool = Field(default=True)
    
    # OCR settings
    TESSERACT_CMD: Optional[str] = Field(default=None)
    OCR_LANGUAGE: str = Field(default="eng")
    OCR_PSM: int = Field(default=6)
    OCR_OEM: int = Field(default=3)
    OCR_TIMEOUT: int = Field(default=120)
    OCR_MIN_CONFIDENCE: int = Field(default=60)
    OCR_PREPROCESS_IMAGE: bool = Field(default=True)
    OCR_ENHANCE_CONTRAST: bool = Field(default=True)
    
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
    
    # PDF processing settings
    PDF_DPI: int = Field(default=300)
    IMAGE_QUALITY: int = Field(default=95)
    MAX_PAGES_PER_DOCUMENT: int = Field(default=100)
    
    # Customer validation settings
    VALIDATE_CUSTOMER_STATUS: bool = Field(default=True)
    REQUIRE_ACTIVE_CUSTOMER: bool = Field(default=True)
    NATIONAL_ID_PATTERN: str = Field(default=r'^\d{10,20}$')
    
    # Background job settings
    QUEUE_MAX_SIZE: int = Field(default=100)
    WORKER_COUNT: int = Field(default=5)
    JOB_CLEANUP_INTERVAL: int = Field(default=3600)
    COLLECT_METRICS: bool = Field(default=True)
    METRICS_INTERVAL: int = Field(default=60)
    HEALTH_CHECK_INTERVAL: int = Field(default=30)
    
    # Redis settings
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_DB: int = Field(default=0)
    REDIS_PASSWORD: str = Field(default="")
    REDIS_TIMEOUT: int = Field(default=5)
    REDIS_MAX_CONNECTIONS: int = Field(default=10)
    CACHE_TTL: int = Field(default=3600)
    ENABLE_RESPONSE_CACHING: bool = Field(default=False)
    ENABLE_QUERY_CACHING: bool = Field(default=True)
    
    # Security settings
    SECRET_KEY: str = Field(default="development-secret-key")
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)
    
    # CORS settings
    CORS_ORIGINS: Union[List[str], str] = Field(default="http://localhost:3000")
    CORS_ALLOW_CREDENTIALS: bool = Field(default=True)
    CORS_ALLOW_METHODS: Union[List[str], str] = Field(default='["*"]')
    CORS_ALLOW_HEADERS: Union[List[str], str] = Field(default='["*"]')
    
    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_WINDOW: int = Field(default=60)
    ENABLE_RATE_LIMITING: bool = Field(default=True)
    MAX_REQUEST_SIZE: int = Field(default=50485760)
    ENABLE_INPUT_SANITIZATION: bool = Field(default=True)
    ALLOW_DANGEROUS_HTML: bool = Field(default=False)
    
    # Logging settings
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_ROTATION: str = Field(default="daily")
    LOG_RETENTION_DAYS: int = Field(default=30)
    LOG_MAX_BYTES: int = Field(default=10485760)
    LOG_BACKUP_COUNT: int = Field(default=5)
    
    # Performance monitoring
    ENABLE_PERFORMANCE_MONITORING: bool = Field(default=True)
    SLOW_QUERY_THRESHOLD: float = Field(default=1.0)
    MEMORY_USAGE_THRESHOLD: int = Field(default=80)
    CPU_USAGE_THRESHOLD: int = Field(default=80)
    
    # Error tracking
    ENABLE_ERROR_TRACKING: bool = Field(default=True)
    ERROR_SAMPLE_RATE: float = Field(default=1.0)
    SENTRY_DSN: str = Field(default="")
    
    # Development settings
    ENABLE_RELOAD: bool = Field(default=True)
    ENABLE_PROFILING: bool = Field(default=False)
    ENABLE_DEBUG_TOOLBAR: bool = Field(default=False)
    TEST_DATABASE_NAME: str = Field(default="doc_processing_test")
    ENABLE_TEST_MODE: bool = Field(default=False)
    MOCK_EXTERNAL_APIS: bool = Field(default=False)
    ENABLE_SWAGGER_UI: bool = Field(default=True)
    ENABLE_REDOC: bool = Field(default=True)
    SWAGGER_UI_OAUTH2_REDIRECT_URL: str = Field(default="")
    
    # Email settings
    SMTP_HOST: str = Field(default="smtp.gmail.com")
    SMTP_PORT: int = Field(default=587)
    SMTP_USER: str = Field(default="")
    SMTP_PASSWORD: str = Field(default="")
    SMTP_USE_TLS: bool = Field(default=True)
    FROM_EMAIL: str = Field(default="noreply@yourcompany.com")
    
    # Webhook settings
    WEBHOOK_URL: str = Field(default="")
    WEBHOOK_SECRET: str = Field(default="")
    ENABLE_WEBHOOKS: bool = Field(default=False)
    SLACK_WEBHOOK_URL: str = Field(default="")
    DISCORD_WEBHOOK_URL: str = Field(default="")
    TEAMS_WEBHOOK_URL: str = Field(default="")
    
    # Feature flags
    ENABLE_HUMAN_REVIEW: bool = Field(default=True)
    ENABLE_AUTO_RETRY: bool = Field(default=True)
    ENABLE_BATCH_PROCESSING: bool = Field(default=True)
    ENABLE_ASYNC_PROCESSING: bool = Field(default=True)
    ENABLE_VECTOR_SEARCH: bool = Field(default=True)
    ENABLE_OCR_FALLBACK: bool = Field(default=True)
    ENABLE_CONFIDENCE_SCORING: bool = Field(default=True)
    ENABLE_AUDIT_LOGGING: bool = Field(default=True)
    ENABLE_ADVANCED_NLP: bool = Field(default=False)
    ENABLE_MULTI_LANGUAGE: bool = Field(default=False)
    ENABLE_IMAGE_PROCESSING: bool = Field(default=False)
    ENABLE_WORKFLOW_OPTIMIZATION: bool = Field(default=False)
    
    # Regional settings
    TIMEZONE: str = Field(default="UTC")
    LOCALE: str = Field(default="en_US.UTF-8")
    CURRENCY: str = Field(default="USD")
    DATE_FORMAT: str = Field(default="%Y-%m-%d")
    TIME_FORMAT: str = Field(default="%H:%M:%S")
    DATETIME_FORMAT: str = Field(default="%Y-%m-%d %H:%M:%S")
    
    # Business logic settings
    MIN_DOCUMENT_PAGES: int = Field(default=1)
    MAX_DOCUMENT_PAGES: int = Field(default=50)
    SUPPORTED_LANGUAGES: Union[List[str], str] = Field(default='["en", "es", "fr", "de"]')
    SUPPORTED_ACTIONS: Union[List[str], str] = Field(default='["freeze_funds", "release_funds"]')
    DEFAULT_ACTION_TIMEOUT: int = Field(default=300)
    
    # Health check settings
    HEALTH_CHECK_ENDPOINT: str = Field(default="/health")
    HEALTH_CHECK_TIMEOUT: int = Field(default=5)
    DATABASE_HEALTH_CHECK: bool = Field(default=True)
    EXTERNAL_SERVICE_HEALTH_CHECK: bool = Field(default=True)
    DISK_SPACE_THRESHOLD: int = Field(default=90)
    MEMORY_THRESHOLD: int = Field(default=85)
    
    # Backup settings
    ENABLE_AUTO_BACKUP: bool = Field(default=False)
    BACKUP_INTERVAL_HOURS: int = Field(default=24)
    BACKUP_RETENTION_DAYS: int = Field(default=7)
    BACKUP_DIRECTORY: str = Field(default="./backups")
    BACKUP_UPLOADED_FILES: bool = Field(default=True)
    BACKUP_PROCESSING_LOGS: bool = Field(default=True)
    BACKUP_CONFIGURATION: bool = Field(default=True)
    
    # Analytics settings
    ENABLE_ANALYTICS: bool = Field(default=True)
    ANALYTICS_RETENTION_DAYS: int = Field(default=90)
    ANONYMOUS_ANALYTICS: bool = Field(default=True)
    GENERATE_DAILY_REPORTS: bool = Field(default=True)
    GENERATE_WEEKLY_REPORTS: bool = Field(default=True)
    REPORT_EMAIL_RECIPIENTS: str = Field(default="admin@yourcompany.com")
    REPORT_DIRECTORY: str = Field(default="./reports")
    
    # Audit settings
    AUDIT_LOG_LEVEL: str = Field(default="INFO")
    AUDIT_LOG_FILE: str = Field(default="./logs/audit.log")
    AUDIT_ALL_REQUESTS: bool = Field(default=True)
    AUDIT_SENSITIVE_DATA: bool = Field(default=False)
    DATA_RETENTION_DAYS: int = Field(default=365)
    PURGE_OLD_DATA_AUTOMATICALLY: bool = Field(default=True)
    ANONYMIZE_OLD_DATA: bool = Field(default=True)
    
    # Privacy settings
    ENABLE_DATA_ENCRYPTION: bool = Field(default=True)
    HASH_SENSITIVE_DATA: bool = Field(default=True)
    GDPR_COMPLIANT: bool = Field(default=True)
    CCPA_COMPLIANT: bool = Field(default=True)
    
    # Validators for list fields
    @field_validator('ALLOWED_FILE_TYPES')
    @classmethod
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [ext.strip() for ext in v.strip('[]').replace('"', '').split(',')]
            else:
                return [ext.strip() for ext in v.split(',')]
        return v
    
    @field_validator('CORS_ORIGINS')
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [origin.strip() for origin in v.strip('[]').replace('"', '').split(',')]
            else:
                return [origin.strip() for origin in v.split(',')]
        return v
    
    @field_validator('CORS_ALLOW_METHODS')
    @classmethod
    def parse_cors_allow_methods(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [method.strip() for method in v.strip('[]').replace('"', '').split(',')]
            else:
                return [method.strip() for method in v.split(',')]
        return v
    
    @field_validator('CORS_ALLOW_HEADERS')
    @classmethod
    def parse_cors_allow_headers(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [header.strip() for header in v.strip('[]').replace('"', '').split(',')]
            else:
                return [header.strip() for header in v.split(',')]
        return v
    
    @field_validator('SUPPORTED_LANGUAGES')
    @classmethod
    def parse_supported_languages(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [lang.strip() for lang in v.strip('[]').replace('"', '').split(',')]
            else:
                return [lang.strip() for lang in v.split(',')]
        return v
    
    @field_validator('SUPPORTED_ACTIONS')
    @classmethod
    def parse_supported_actions(cls, v):
        if isinstance(v, str):
            if v.startswith('[') and v.endswith(']'):
                import json
                try:
                    return json.loads(v)
                except:
                    return [action.strip() for action in v.strip('[]').replace('"', '').split(',')]
            else:
                return [action.strip() for action in v.split(',')]
        return v

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "allow"  # Changed to "allow" to accept extra fields
    }

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

# Optional validation (don't fail in development)
def validate_settings():
    """Validate that critical settings are present"""
    if not settings.DEBUG:  # Only validate in production
        required_settings = ["OPENAI_API_KEY", "SECRET_KEY"]
        
        missing_settings = []
        for setting in required_settings:
            value = getattr(settings, setting, None)
            if not value or value in ["", "temp-key", "temp-secret"]:
                missing_settings.append(setting)
        
        if missing_settings:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_settings)}")

# Don't fail on startup for development
try:
    validate_settings()
except Exception as e:
    if not getattr(settings, 'DEBUG', True):
        raise
    else:
        print(f"Settings validation warning (development mode): {e}")

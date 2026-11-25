"""
Application settings and configuration management using Pydantic Settings.
Compliant with SRS Section 4, 13.2, INT-01 through INT-14, and Addendum v1.3.
"""

from typing import List, Literal, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # General App Configuration
    APP_NAME: str = "Multimodal Research Co-pilot"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    API_PREFIX: str = "/api/v1"

    # Gateway Configuration (FreeLLMAPI or OpenAI-compatible) (INT-01, INT-02)
    GATEWAY_BASE_URL: str = Field(
        default="http://localhost:3001/v1", description="OpenAI-compatible gateway URL"
    )
    GATEWAY_API_KEY: str = Field(
        default="freellmapi-token", description="Bearer token for the gateway"
    )

    # Model Modes (INT-04)
    MODEL_MODE: Literal["auto", "pinned"] = Field(
        default="auto",
        description="'auto' routes dynamically through FreeLLMAPI; 'pinned' forces explicit models",
    )
    PINNED_DEFAULT_MODEL: str = "gpt-4o-mini"
    PINNED_VERIFIER_MODEL: str = "gpt-4o-mini"
    PINNED_VISION_MODEL: str = "gpt-4o-mini"

    # Embedding backend (ADR-4, INT-05)
    # "local"   → deterministic n-gram hashing; zero API cost, works fully offline
    # "gateway" → calls GATEWAY_BASE_URL/embeddings with EMBEDDING_API_MODEL; best semantic quality
    EMBEDDING_BACKEND: Literal["local", "gateway"] = "local"
    EMBEDDING_API_MODEL: str = "text-embedding-3-small"
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 64
    EMBEDDING_FALLBACK_TO_LOCAL: bool = True

    # Storage Paths
    DATA_DIR: str = "./data"
    UPLOADS_DIR: str = "./data/uploads"
    CROPS_DIR: str = "./data/crops"
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/copilot.db"

    # Notes storage (Addendum v1.3)
    NOTES_DIR: str = "./data/notes"

    # Cited works cache (Addendum v1.3)
    CITED_WORKS_DIR: str = "./data/cited_works"
    MAX_CITED_WORK_SIZE_MB: int = 20
    CITED_WORK_FETCH_TIMEOUT: int = 30

    # Formula extraction (Addendum v1.3)
    FORMULA_CONFIDENCE_THRESHOLD: float = 0.7

    # Privacy Settings (INT-09, NFR-PRIV-01)
    PRIVACY_DEFAULT: Literal["allow_remote", "local_only"] = "allow_remote"

    # Resiliency & Backoff (INT-06)
    MAX_RETRIES: int = 4
    BASE_BACKOFF_SECONDS: float = 1.0
    MAX_BACKOFF_SECONDS: float = 16.0
    REQUEST_TIMEOUT_SECONDS: float = 60.0

    # Structured Output Retries (INT-07)
    SCHEMA_VALIDATION_RETRIES: int = 2

    # Ingestion Cache (INT-12)
    CACHE_INGESTION_RESPONSES: bool = True

    # External APIs (SRS 8.4, FR-REF-03, 05)
    GROBID_URL: str = "http://localhost:8070"
    CROSSREF_MAILTO: str = "research-copilot@example.edu"
    OPENALEX_MAILTO: str = "research-copilot@example.edu"

    # File limits (FR-ING-01)
    MAX_FILE_SIZE_MB: int = 50
    MAX_PAGES: int = 100


settings = Settings()

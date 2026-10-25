"""
app/config.py
─────────────
Centralised application configuration backed by pydantic-settings.

All values are read from environment variables (and an optional .env file).
No secret is ever hard-coded here.  Unknown fields in .env are silently
ignored, so the same .env.example can be used as a template across teams.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_env_file() -> str | None:
    """Return .env path unless in test environment to preserve test isolation."""
    return None if os.environ.get("APP_ENV") == "test" else ".env"

# ─── Enumerations ─────────────────────────────────────────────────────────────

class AppEnvironment(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TEST = "test"


class LLMProvider(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    MOCK = "mock"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# ─── Settings model ───────────────────────────────────────────────────────────

class Settings(BaseSettings):
    """
    VADER application settings.

    Environment variables take precedence over defaults.
    A `.env` file in the backend directory is loaded automatically when present.
    """

    model_config = SettingsConfigDict(
        env_file=_get_env_file(),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",         # silently ignore unrecognised env vars
        populate_by_name=True,
    )

    def __init__(self, **values):
        if os.environ.get("APP_ENV") == "test" and "_env_file" not in values:
            super().__init__(_env_file=None, **values)
        else:
            super().__init__(**values)

    # ── Application ──────────────────────────────────────────────────────────
    app_env: AppEnvironment = Field(
        default=AppEnvironment.DEVELOPMENT,
        description="Runtime environment.",
    )
    app_debug: bool = Field(default=False)
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_log_level: LogLevel = Field(default=LogLevel.INFO)

    # ── CORS ─────────────────────────────────────────────────────────────────
    cors_allowed_origins: str = Field(
        default="http://localhost:5173,http://127.0.0.1:5173",
        description=(
            "Comma-separated list of origins the React dev server runs on. "
            "Use '*' ONLY in development and never in production."
        ),
    )

    @property
    def cors_origins_list(self) -> list[str]:
        """Return parsed CORS origins as a list, stripping whitespace."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    # ── LLM ──────────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key.")
    gemini_api_key: str = Field(default="", description="Google Gemini API key.")

    groq_model_id: str = Field(default="", description="Groq model ID override.")
    gemini_model_id: str = Field(default="", description="Gemini model ID override.")

    llm_provider: LLMProvider = Field(
        default=LLMProvider.MOCK,
        description="Active LLM provider.  Use 'mock' for local testing.",
    )
    llm_model_id: str = Field(
        default="",
        description=(
            "Provider-specific model identifier.  Leave blank to use the "
            "provider adapter's own default, which is looked up at runtime."
        ),
    )
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=1, le=32768)
    llm_timeout_seconds: int = Field(default=60, ge=1, le=300)
    llm_max_retries: int = Field(default=3, ge=0, le=10)

    # ── Debate orchestrator ───────────────────────────────────────────────────
    debate_max_rounds: int = Field(default=3, ge=1, le=10)
    debate_similarity_threshold: float = Field(
        default=0.92, ge=0.0, le=1.0,
        description=(
            "Cosine-similarity convergence signal threshold.  "
            "NOT a correctness metric — used only as an optional early-stop hint."
        ),
    )

    # ── Rate limiting ─────────────────────────────────────────────────────────
    rate_limit_requests_per_minute: int = Field(default=20, ge=1)
    rate_limit_enabled: bool = Field(default=True)

    # ── Search / retrieval ────────────────────────────────────────────────────
    search_provider: str = Field(
        default="ddg",
        description="Active search provider: 'ddg' or 'mock'.",
    )
    search_max_results: int = Field(default=5, ge=1, le=20)
    search_timeout_seconds: int = Field(default=10, ge=1, le=60)

    # ── Input limits (request validation) ────────────────────────────────────
    max_query_length: int = Field(
        default=4000,
        description="Maximum number of characters allowed in a user query.",
    )

    # ─── Validators ──────────────────────────────────────────────────────────

    @field_validator("groq_api_key", "gemini_api_key", mode="before")
    @classmethod
    def _strip_key(cls, v: str) -> str:
        """Strip accidental whitespace around API keys."""
        return v.strip() if isinstance(v, str) else v

    def requires_groq_key(self) -> bool:
        return self.llm_provider == LLMProvider.GROQ

    def requires_gemini_key(self) -> bool:
        return self.llm_provider == LLMProvider.GEMINI

    def active_key_present(self) -> bool:
        """Return True if the key for the active provider is set."""
        if self.llm_provider == LLMProvider.MOCK:
            return True
        if self.llm_provider == LLMProvider.GROQ:
            return bool(self.groq_api_key)
        if self.llm_provider == LLMProvider.GEMINI:
            return bool(self.gemini_api_key)
        return False

    def configure_logging(self) -> None:
        """Apply the configured log level to the root logger."""
        logging.basicConfig(
            level=self.app_log_level.value.upper(),
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )


# ─── Singleton accessor ───────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    In tests, call `get_settings.cache_clear()` after patching env vars to
    force a fresh read.
    """
    return Settings()

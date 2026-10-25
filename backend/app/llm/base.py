"""
app/llm/base.py
───────────────
Abstract LLM client interface for VADER.

All provider-specific logic lives in concrete subclasses (groq_adapter.py,
gemini_adapter.py, mock_adapter.py).  The rest of the application depends
ONLY on this interface, making it trivial to swap providers or inject mocks.

Design principles:
  - Async-first: every method is a coroutine.
  - Validated output: structured outputs are validated with Pydantic before
    being returned to the caller.
  - No unbounded retries: retry policy is configured at construction time
    and enforced by the tenacity decorator in concrete subclasses.
  - No private chain-of-thought: adapters expose only the final response text.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Data classes shared by all adapters ─────────────────────────────────────

@dataclass(frozen=True)
class LLMMessage:
    """
    A single message in the conversation history sent to the LLM.

    role must be one of: "system", "user", "assistant".
    """

    role: str          # "system" | "user" | "assistant"
    content: str

    def __post_init__(self) -> None:
        allowed = {"system", "user", "assistant"}
        if self.role not in allowed:
            raise ValueError(f"LLMMessage.role must be one of {allowed}, got {self.role!r}")
        if not self.content.strip():
            raise ValueError("LLMMessage.content must not be blank.")


@dataclass
class LLMResponse:
    """
    Structured response returned by every LLM adapter.

    chain_of_thought is intentionally NOT included — adapters must not
    leak private reasoning steps to the caller.
    """

    content: str                          # The model's public response text
    model_id: str                         # Actual model identifier used
    provider: str                         # Provider name ("groq", "gemini", "mock")
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: str = "stop"           # "stop" | "length" | "error"


@dataclass
class LLMConfig:
    """
    Runtime configuration passed to an LLM adapter.

    Adapters validate that model_id is supported before making API calls.
    An empty model_id signals that the adapter should use its own default.
    """

    model_id: str = ""
    temperature: float = 0.3
    max_tokens: int = 2048
    timeout_seconds: int = 60
    max_retries: int = 3
    system_prompt: str | None = None
    extra: dict = field(default_factory=dict)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class LLMError(Exception):
    """Base exception for all LLM adapter errors."""


class LLMAuthenticationError(LLMError):
    """Raised when the API key is missing or invalid."""


class LLMRateLimitError(LLMError):
    """Raised when the provider returns a 429 / rate-limit response."""


class LLMTimeoutError(LLMError):
    """Raised when the request exceeds the configured timeout."""


class LLMValidationError(LLMError):
    """Raised when the model's output does not conform to expected schema."""


class LLMUnsupportedModelError(LLMError):
    """Raised when model_id is not supported by the active provider."""


# ─── Secret sanitization helper ───────────────────────────────────────────────

def sanitize_error_message(text: str, additional_secrets: list[str] | None = None) -> str:
    """
    Remove API keys, tokens, and credentials from exception messages.
    Prevents secrets from leaking into application logs or client responses.
    """
    if not text:
        return ""

    import re

    redacted = text

    # Redact known environment keys if settings are loaded
    try:
        from app.config import get_settings

        settings = get_settings()
        for key in (settings.groq_api_key, settings.gemini_api_key):
            if key and len(key) >= 6:
                redacted = redacted.replace(key, "[REDACTED]")
    except Exception:
        pass

    if additional_secrets:
        for secret in additional_secrets:
            if secret and len(secret) >= 4:
                redacted = redacted.replace(secret, "[REDACTED]")

    # Redact query parameters / common credential patterns
    redacted = re.sub(
        r"(key|api_key|apikey|token|secret|authorization)=([^&\s]+)",
        r"\1=[REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    # Redact Bearer tokens
    redacted = re.sub(
        r"Bearer\s+([A-Za-z0-9_\-\.]+)",
        "Bearer [REDACTED]",
        redacted,
        flags=re.IGNORECASE,
    )
    # Redact Groq API keys (gsk_...)
    redacted = re.sub(r"gsk_[A-Za-z0-9_\-]+", "[REDACTED]", redacted)
    # Redact Google API keys (AIza...)
    redacted = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED]", redacted)
    # Redact standard sk- keys (sk-...)
    redacted = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[REDACTED]", redacted)

    return redacted


# ─── Abstract base class ──────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    Abstract base class that every LLM adapter must implement.

    Usage example
    -------------
    client = GroqAdapter(config)
    response = await client.complete([
        LLMMessage("system", "You are a helpful assistant."),
        LLMMessage("user", "What is the capital of France?"),
    ])
    print(response.content)
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ── Abstract methods ──────────────────────────────────────────────────────

    @abstractmethod
    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        """
        Send a list of messages to the LLM and return a single response.

        Parameters
        ----------
        messages:
            Ordered conversation history.  The last message must have
            role == "user".

        Returns
        -------
        LLMResponse
            The model's response, including metadata.

        Raises
        ------
        LLMAuthenticationError  — invalid or missing API key
        LLMRateLimitError       — provider rate limit hit (after retries)
        LLMTimeoutError         — request timed out
        LLMValidationError      — response structure is unexpected
        LLMError                — any other provider error
        """
        ...

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier string (e.g. "groq")."""
        ...

    @abstractmethod
    def default_model_id(self) -> str:
        """
        Return the provider's recommended default model ID.

        This is called when LLMConfig.model_id is empty.
        The value MUST be verified against the provider's official docs —
        never invented.
        """
        ...

    # ── Concrete helpers ──────────────────────────────────────────────────────

    @property
    def effective_model_id(self) -> str:
        """Return the configured model ID, falling back to the default."""
        return self._config.model_id or self.default_model_id()

    def _validate_messages(self, messages: list[LLMMessage]) -> None:
        """
        Shared pre-call validation.

        Raises ValueError if the message list is empty or the last message
        is not from the user.
        """
        if not messages:
            raise ValueError("messages must not be empty.")
        if messages[-1].role != "user":
            raise ValueError(
                "The last message in the conversation must have role='user'."
            )

    def _log_call(self, messages: list[LLMMessage]) -> None:
        """Log a summarised representation of the outgoing call (no content)."""
        self._logger.debug(
            "LLM call | provider=%s model=%s messages=%d",
            self.provider_name(),
            self.effective_model_id,
            len(messages),
        )

    def _log_response(self, response: LLMResponse) -> None:
        """Log response metadata (no content logged to avoid leaking data)."""
        self._logger.debug(
            "LLM response | provider=%s model=%s tokens=%d+%d latency=%.0fms finish=%s",
            response.provider,
            response.model_id,
            response.prompt_tokens,
            response.completion_tokens,
            response.latency_ms,
            response.finish_reason,
        )


# ─── Factory helper ───────────────────────────────────────────────────────────

def build_llm_client(provider: str, config: LLMConfig) -> BaseLLMClient:
    """
    Instantiate the correct LLM adapter for the given provider name.

    Import is deferred so that provider SDKs are only loaded when needed.
    This keeps the test suite fast — the mock adapter has zero heavy imports.

    Parameters
    ----------
    provider:
        One of "groq", "gemini", "mock".
    config:
        LLMConfig instance with runtime parameters.

    Raises
    ------
    ValueError
        If provider is not recognised.
    """
    provider = provider.lower().strip()

    if provider == "mock":
        from app.llm.mock_adapter import MockLLMClient
        return MockLLMClient(config)

    if provider == "groq":
        from app.llm.groq_adapter import GroqLLMClient
        return GroqLLMClient(config)

    if provider == "gemini":
        from app.llm.gemini_adapter import GeminiLLMClient
        return GeminiLLMClient(config)

    raise ValueError(
        f"Unknown LLM provider: {provider!r}. "
        "Supported values: 'groq', 'gemini', 'mock'."
    )

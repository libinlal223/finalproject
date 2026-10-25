"""
app/llm/groq_adapter.py
────────────────────────
Groq LLM adapter for VADER.

Model IDs are NOT hard-coded.  The default is read from the Groq SDK's own
documentation pattern and can be overridden via LLM_MODEL_ID in .env.

Supported Groq models (verify current list at https://console.groq.com/docs/models):
  - llama-3.3-70b-versatile   (recommended default — fast, high quality)
  - llama-3.1-8b-instant      (fastest, lower quality)
  - mixtral-8x7b-32768        (large context)
  - gemma2-9b-it              (Google Gemma via Groq)

The adapter raises structured LLMError subclasses so the caller never has to
import the groq SDK directly.

Retry policy: up to LLMConfig.max_retries attempts with exponential back-off,
only on rate-limit (429) and transient server errors (5xx).
"""

from __future__ import annotations

import logging
import time

from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.llm.base import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
    sanitize_error_message,
)

logger = logging.getLogger(__name__)

# Default model — verified against https://console.groq.com/docs/models
_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Known Groq model IDs.  Used for upfront validation only; the API itself is
# the authoritative source.  An empty LLM_MODEL_ID bypasses this check.
_KNOWN_GROQ_MODELS: set[str] = {
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "gemma-7b-it",
}


class GroqLLMClient(BaseLLMClient):
    """
    Async Groq LLM adapter.

    The groq Python SDK is imported lazily inside methods so that
    importing this module does NOT fail when the groq package is absent
    in test environments using only the mock adapter.
    """

    def provider_name(self) -> str:
        return "groq"

    def default_model_id(self) -> str:
        return _GROQ_DEFAULT_MODEL

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        self._validate_messages(messages)

        model_id = self.effective_model_id
        if model_id and model_id not in _KNOWN_GROQ_MODELS:
            logger.warning(
                "Groq model '%s' is not in the known model list.  "
                "Proceeding — the API will reject it if unsupported.",
                model_id,
            )

        self._log_call(messages)

        try:
            return await self._call_with_retry(messages, model_id)
        except RetryError as exc:
            if exc.last_attempt and exc.last_attempt.exception():
                raise exc.last_attempt.exception() from exc
            raise LLMRateLimitError(
                f"Groq rate limit persisted after {self._config.max_retries} retries."
            ) from exc

    async def _call_with_retry(
        self, messages: list[LLMMessage], model_id: str
    ) -> LLMResponse:
        """Perform the API call with tenacity retry for transient errors."""
        retry_wait = self._config.extra.get("retry_wait")
        if retry_wait is None:
            retry_wait = wait_exponential(multiplier=1, min=2, max=30)

        @retry(
            retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
            stop=stop_after_attempt(self._config.max_retries + 1),
            wait=retry_wait,
            reraise=True,
        )
        async def _attempt() -> LLMResponse:
            return await self._single_call(messages, model_id)

        return await _attempt()

    async def _single_call(
        self, messages: list[LLMMessage], model_id: str
    ) -> LLMResponse:
        """Make one API call to Groq."""
        try:
            from groq import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                AsyncGroq,
                AuthenticationError,
                PermissionDeniedError,
                RateLimitError,
            )
        except ImportError as exc:
            raise LLMError(
                "groq package is not installed.  Run: pip install groq"
            ) from exc

        from app.config import get_settings
        settings = get_settings()

        if not settings.groq_api_key:
            raise LLMAuthenticationError(
                "GROQ_API_KEY environment variable is not set."
            )

        groq_messages = [
            {"role": m.role, "content": m.content} for m in messages
        ]
        if self._config.system_prompt:
            groq_messages.insert(
                0, {"role": "system", "content": self._config.system_prompt}
            )

        t_start = time.monotonic()
        try:
            client = AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=float(self._config.timeout_seconds),
                max_retries=0,  # we handle retries via tenacity
            )
            completion = await client.chat.completions.create(
                model=model_id,
                messages=groq_messages,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
            )
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq request timed out: {sanitize_error_message(str(exc))}"
            ) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise LLMAuthenticationError(
                f"Groq authentication failed: {sanitize_error_message(str(exc))}"
            ) from exc
        except RateLimitError as exc:
            raise LLMRateLimitError(
                f"Groq rate limit: {sanitize_error_message(str(exc))}"
            ) from exc
        except APIStatusError as exc:
            sanitized = sanitize_error_message(str(exc))
            if exc.status_code in (401, 403):
                raise LLMAuthenticationError(
                    f"Groq authentication failed: {sanitized}"
                ) from exc
            if exc.status_code == 429:
                raise LLMRateLimitError(
                    f"Groq rate limit: {sanitized}"
                ) from exc
            raise LLMError(f"Groq API error {exc.status_code}: {sanitized}") from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Groq connection error: {sanitize_error_message(str(exc))}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("Groq request timed out.") from exc
        except Exception as exc:
            raise LLMError(
                f"Groq unexpected error: {sanitize_error_message(str(exc))}"
            ) from exc

        latency_ms = (time.monotonic() - t_start) * 1000

        if not getattr(completion, "choices", None):
            raise LLMValidationError("Groq returned empty completion choices.")

        choice = completion.choices[0]
        usage = completion.usage or type("_", (), {"prompt_tokens": 0, "completion_tokens": 0})()

        response = LLMResponse(
            content=choice.message.content or "",
            model_id=completion.model,
            provider=self.provider_name(),
            prompt_tokens=getattr(usage, "prompt_tokens", 0),
            completion_tokens=getattr(usage, "completion_tokens", 0),
            latency_ms=latency_ms,
            finish_reason=choice.finish_reason or "stop",
        )
        self._log_response(response)
        return response

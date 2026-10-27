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


def _parse_groq_rate_limit(exc: Exception) -> LLMRateLimitError:
    """Extract retry_after, token limits, and diagnostic details from a Groq 429 error."""
    import re
    msg = sanitize_error_message(str(exc))
    retry_after: float | None = None
    limit_tokens: int | None = None
    used_tokens: int | None = None
    requested_tokens: int | None = None
    limit_type: str | None = None

    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers:
        raw_retry = headers.get("retry-after")
        if raw_retry:
            try:
                retry_after = float(raw_retry)
            except (ValueError, TypeError):
                pass
        if retry_after is None:
            raw_reset_tokens = headers.get("x-ratelimit-reset-tokens")
            if raw_reset_tokens:
                m_reset = re.search(r"([0-9]+(?:\.[0-9]+)?)", raw_reset_tokens)
                if m_reset:
                    try:
                        retry_after = float(m_reset.group(1))
                    except (ValueError, TypeError):
                        pass

        raw_limit = headers.get("x-ratelimit-limit-tokens")
        if raw_limit and raw_limit.isdigit():
            limit_tokens = int(raw_limit)

    if retry_after is None:
        m_msg = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s?", msg, re.IGNORECASE)
        if m_msg:
            try:
                retry_after = float(m_msg.group(1))
            except (ValueError, TypeError):
                pass

    m_stats = re.search(
        r"Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+)",
        msg,
        re.IGNORECASE,
    )
    if m_stats:
        limit_tokens = int(m_stats.group(1))
        used_tokens = int(m_stats.group(2))
        requested_tokens = int(m_stats.group(3))

    if "tokens per minute" in msg.lower() or "(tpm)" in msg.lower() or "tokens" in msg.lower():
        limit_type = "tokens"
    elif "requests per minute" in msg.lower() or "(rpm)" in msg.lower() or "requests" in msg.lower():
        limit_type = "requests"

    return LLMRateLimitError(
        f"Groq rate limit: {msg}",
        retry_after=retry_after,
        limit_tokens=limit_tokens,
        used_tokens=used_tokens,
        requested_tokens=requested_tokens,
        limit_type=limit_type,
    )


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

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
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
            return await self._call_with_retry(
                messages,
                model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                json_mode=json_mode,
            )
        except RetryError as exc:
            if exc.last_attempt and exc.last_attempt.exception():
                raise exc.last_attempt.exception() from exc
            raise LLMRateLimitError(
                f"Groq rate limit persisted after {self._config.max_retries} retries."
            ) from exc

    async def _call_with_retry(
        self,
        messages: list[LLMMessage],
        model_id: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        """Perform the API call with tenacity retry for transient errors and adaptive rate-limit backoff."""
        override_retry_wait = self._config.extra.get("retry_wait")

        def _compute_wait(retry_state) -> float:
            if override_retry_wait is not None:
                return override_retry_wait(retry_state) if callable(override_retry_wait) else override_retry_wait
            if retry_state.outcome and retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
                    # Respect provider reset time with a 0.5s safety buffer, capped at 45s
                    return min(max(exc.retry_after + 0.5, 1.0), 45.0)
            return wait_exponential(multiplier=1, min=2, max=30)(retry_state)

        def _should_retry(retry_state) -> bool:
            if not (retry_state.outcome and retry_state.outcome.failed):
                return False
            exc = retry_state.outcome.exception()
            if isinstance(exc, LLMTimeoutError):
                return True
            if isinstance(exc, LLMRateLimitError):
                # Abort futile retries if single request tokens alone exceed total limit
                if (
                    exc.requested_tokens is not None
                    and exc.limit_tokens is not None
                    and exc.requested_tokens > exc.limit_tokens
                ):
                    logger.warning(
                        "Groq rate limit: requested tokens (%d) exceed limit (%d); aborting futile retries.",
                        exc.requested_tokens,
                        exc.limit_tokens,
                    )
                    return False
                # Abort futile retries if retry_after exceeds maximum wait budget (45s)
                if exc.retry_after is not None and exc.retry_after > 45.0:
                    logger.warning(
                        "Groq rate limit: retry_after (%.1fs) exceeds maximum wait budget (45s); aborting retries.",
                        exc.retry_after,
                    )
                    return False
                return True
            return False

        @retry(
            retry=_should_retry,
            stop=stop_after_attempt(self._config.max_retries + 1),
            wait=_compute_wait,
            reraise=True,
        )
        async def _attempt() -> LLMResponse:
            if json_mode is not None:
                return await self._single_call(
                    messages,
                    model_id,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    json_mode=json_mode,
                )
            return await self._single_call(
                messages,
                model_id,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        return await _attempt()

    async def _single_call(
        self,
        messages: list[LLMMessage],
        model_id: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool | None = None,
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

        eff_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        eff_temperature = temperature if temperature is not None else self._config.temperature
        eff_json_mode = (
            json_mode
            if json_mode is not None
            else (
                getattr(self._config, "json_mode", False)
                or self._config.extra.get("json_mode")
            )
        )

        t_start = time.monotonic()
        try:
            client = AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=float(self._config.timeout_seconds),
                max_retries=0,  # we handle retries via tenacity
            )
            create_kwargs: dict = {
                "model": model_id,
                "messages": groq_messages,
                "temperature": eff_temperature,
                "max_tokens": eff_max_tokens,
            }
            if eff_json_mode:
                create_kwargs["response_format"] = {"type": "json_object"}

            completion = await client.chat.completions.create(**create_kwargs)
        except APITimeoutError as exc:
            raise LLMTimeoutError(
                f"Groq request timed out: {sanitize_error_message(str(exc))}"
            ) from exc
        except (AuthenticationError, PermissionDeniedError) as exc:
            raise LLMAuthenticationError(
                f"Groq authentication failed: {sanitize_error_message(str(exc))}"
            ) from exc
        except RateLimitError as exc:
            raise _parse_groq_rate_limit(exc) from exc
        except APIStatusError as exc:
            sanitized = sanitize_error_message(str(exc))
            if exc.status_code in (401, 403):
                raise LLMAuthenticationError(
                    f"Groq authentication failed: {sanitized}"
                ) from exc
            if exc.status_code == 429:
                raise _parse_groq_rate_limit(exc) from exc
            raise LLMError(f"Groq API error {exc.status_code}: {sanitized}") from exc
        except APIConnectionError as exc:
            raise LLMError(
                f"Groq connection error: {sanitize_error_message(str(exc))}"
            ) from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("Groq request timed out.") from exc
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
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
        return response

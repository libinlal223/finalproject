"""
app/llm/gemini_adapter.py
──────────────────────────
Google Gemini LLM adapter for VADER.

Model IDs are NOT hard-coded.  The default is read from official docs and
can be overridden via LLM_MODEL_ID in .env.

Supported Gemini models (verify current list at
https://ai.google.dev/gemini-api/docs/models/gemini):
  - gemini-1.5-flash    (recommended default — fast, efficient)
  - gemini-1.5-pro      (higher quality, slower)
  - gemini-2.0-flash    (latest fast model)

Retry policy mirrors the Groq adapter: tenacity with exponential back-off
on rate-limit and transient errors only.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

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

# Default model — verified against https://ai.google.dev/gemini-api/docs/models
_GEMINI_DEFAULT_MODEL = "gemini-3.5-flash-lite"

_KNOWN_GEMINI_MODELS: set[str] = {
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
}


class GeminiLLMClient(BaseLLMClient):
    """
    Async Google Gemini LLM adapter.

    The google-generativeai SDK is imported lazily so that test environments
    using only the mock adapter never need it installed.
    """

    def provider_name(self) -> str:
        return "gemini"

    def default_model_id(self) -> str:
        return _GEMINI_DEFAULT_MODEL

    async def complete(
        self,
        messages: list[LLMMessage],
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        self._validate_messages(messages)

        model_id = self.effective_model_id
        if model_id and model_id not in _KNOWN_GEMINI_MODELS:
            logger.warning(
                "Gemini model '%s' is not in the known model list.  "
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
                f"Gemini rate limit persisted after {self._config.max_retries} retries."
            ) from exc

    async def _call_with_retry(
        self,
        messages: list[LLMMessage],
        model_id: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool | None = None,
    ) -> LLMResponse:
        override_retry_wait = self._config.extra.get("retry_wait")

        def _compute_wait(retry_state) -> float:
            if override_retry_wait is not None:
                return override_retry_wait(retry_state) if callable(override_retry_wait) else override_retry_wait
            if retry_state.outcome and retry_state.outcome.failed:
                exc = retry_state.outcome.exception()
                if isinstance(exc, LLMRateLimitError) and exc.retry_after is not None:
                    return min(max(exc.retry_after + 0.5, 1.0), 45.0)
            return wait_exponential(multiplier=1, min=2, max=30)(retry_state)

        def _should_retry(retry_state) -> bool:
            if not (retry_state.outcome and retry_state.outcome.failed):
                return False
            exc = retry_state.outcome.exception()
            if isinstance(exc, LLMTimeoutError):
                return True
            if isinstance(exc, LLMRateLimitError):
                if exc.retry_after is not None and exc.retry_after > 45.0:
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
        """Make one API call to Google Gemini."""
        import re

        try:
            import google.generativeai as genai
            from google.api_core.exceptions import (
                DeadlineExceeded,
                GoogleAPICallError,
                PermissionDenied,
                ResourceExhausted,
                Unauthenticated,
            )
        except ImportError as exc:
            raise LLMError(
                "google-generativeai package is not installed.  "
                "Run: pip install google-generativeai"
            ) from exc

        from app.config import get_settings
        settings = get_settings()

        if not settings.gemini_api_key:
            raise LLMAuthenticationError(
                "GEMINI_API_KEY environment variable is not set."
            )

        genai.configure(api_key=settings.gemini_api_key)

        # Build Gemini message history (system prompt is handled separately)
        history = []
        system_content = self._config.system_prompt or ""
        for msg in messages[:-1]:
            if msg.role == "system":
                system_content = msg.content
                continue
            gemini_role = "model" if msg.role == "assistant" else "user"
            history.append({"role": gemini_role, "parts": [msg.content]})

        last_user_message = messages[-1].content

        eff_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
        eff_temperature = temperature if temperature is not None else self._config.temperature

        gen_config = None
        gen_kwargs: dict[str, Any] = {
            "temperature": eff_temperature,
            "max_output_tokens": eff_max_tokens,
        }
        eff_json_mode = (
            json_mode
            if json_mode is not None
            else (
                getattr(self._config, "json_mode", False)
                or self._config.extra.get("json_mode")
                or self._config.extra.get("response_mime_type") == "application/json"
            )
        )
        if eff_json_mode:
            gen_kwargs["response_mime_type"] = "application/json"

        if hasattr(genai, "types") and hasattr(genai.types, "GenerationConfig"):
            gen_config = genai.types.GenerationConfig(**gen_kwargs)
        elif hasattr(genai, "GenerationConfig"):
            gen_config = genai.GenerationConfig(**gen_kwargs)

        model_kwargs = {
            "model_name": model_id,
            "system_instruction": system_content or None,
        }
        if gen_config is not None:
            model_kwargs["generation_config"] = gen_config

        t_start = time.monotonic()
        try:
            model = genai.GenerativeModel(**model_kwargs)
            chat = model.start_chat(history=history)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()

            response = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: chat.send_message(last_user_message),
                ),
                timeout=float(self._config.timeout_seconds),
            )
        except ResourceExhausted as exc:
            sanitized = sanitize_error_message(str(exc))
            retry_after = None
            m = re.search(r"try again in\s+([0-9]+(?:\.[0-9]+)?)\s*s?", sanitized, re.IGNORECASE)
            if not m:
                m = re.search(r"retry after\s+([0-9]+(?:\.[0-9]+)?)\s*s?", sanitized, re.IGNORECASE)
            if m:
                try:
                    retry_after = float(m.group(1))
                except (ValueError, TypeError):
                    pass
            raise LLMRateLimitError(
                f"Gemini rate limit: {sanitized}",
                retry_after=retry_after,
                limit_type="tokens",
            ) from exc
        except (Unauthenticated, PermissionDenied) as exc:
            raise LLMAuthenticationError(
                f"Gemini authentication failed: {sanitize_error_message(str(exc))}"
            ) from exc
        except DeadlineExceeded as exc:
            raise LLMTimeoutError(
                f"Gemini request timed out: {sanitize_error_message(str(exc))}"
            ) from exc
        except GoogleAPICallError as exc:
            sanitized = sanitize_error_message(str(exc))
            code = getattr(exc, "code", None)
            if code == 429 or "quota" in sanitized.lower() or "resource exhausted" in sanitized.lower():
                raise LLMRateLimitError(f"Gemini rate limit: {sanitized}") from exc
            if code in (401, 403):
                raise LLMAuthenticationError(f"Gemini authentication failed: {sanitized}") from exc
            raise LLMError(f"Gemini API error: {sanitized}") from exc
        except TimeoutError as exc:
            raise LLMTimeoutError("Gemini request timed out.") from exc
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise LLMError(
                f"Gemini unexpected error: {sanitize_error_message(str(exc))}"
            ) from exc

        latency_ms = (time.monotonic() - t_start) * 1000

        # Check prompt feedback for blocks
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if prompt_feedback:
            block_reason = getattr(prompt_feedback, "block_reason", None)
            if block_reason is not None:
                br_val = getattr(block_reason, "value", block_reason)
                br_name = getattr(block_reason, "name", "")
                if isinstance(br_val, int) and br_val > 0:
                    raise LLMError(f"Gemini prompt blocked by safety filters: {br_name or br_val}")
                elif isinstance(block_reason, str) and block_reason.strip() and block_reason.lower() not in ("unspecified", "none", "0"):
                    raise LLMError(f"Gemini prompt blocked by safety filters: {block_reason}")
                elif isinstance(br_name, str) and br_name.upper() in ("SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT"):
                    raise LLMError(f"Gemini prompt blocked by safety filters: {br_name}")

        # Safely extract text; response.text raises ValueError if candidate blocked
        content_text = ""
        try:
            content_text = response.text or ""
        except (ValueError, AttributeError):
            if hasattr(response, "candidates") and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                    parts = [
                        getattr(p, "text", "")
                        for p in candidate.content.parts
                        if getattr(p, "text", None)
                    ]
                    content_text = "".join(parts)

        # Extract finish reason if available
        finish_reason = "stop"
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            fr = getattr(candidate, "finish_reason", None)
            if fr is not None:
                fr_name = getattr(fr, "name", str(fr)).lower()
                finish_reason = "length" if ("max" in fr_name or "length" in fr_name) else fr_name

        if not content_text.strip():
            if finish_reason in ("safety", "blocklist", "recitation", "other"):
                raise LLMError(f"Gemini candidate generation blocked by safety filters (finish_reason={finish_reason}).")
            raise LLMValidationError("Gemini returned empty completion response.")

        # Gemini SDK exposes token counts in usage_metadata
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) or 0

        result = LLMResponse(
            content=content_text,
            model_id=model_id,
            provider=self.provider_name(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            finish_reason=finish_reason,
        )
        self._log_response(result)
        return result

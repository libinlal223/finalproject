"""
tests/test_llm_adapters.py
──────────────────────────
Comprehensive offline automated tests for VADER LLM adapters:
- MockLLMClient
- GroqLLMClient
- GeminiLLMClient
- BaseLLMClient & factory build_llm_client
- Secret redaction & leak prevention

All tests are deterministic, mock SDK interactions, and make zero network calls.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_none

from app.config import get_settings
from app.llm.base import (
    LLMAuthenticationError,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
    build_llm_client,
    sanitize_error_message,
)
from app.llm.gemini_adapter import GeminiLLMClient
from app.llm.groq_adapter import GroqLLMClient
from app.llm.mock_adapter import MockLLMClient

# ─── 1. Mock Adapter Tests ───────────────────────────────────────────────────

class TestMockAdapter:
    """Covers Task 2 item 1 and item 2: Mock adapter success and simulated errors."""

    async def test_mock_success_single_message(self):
        config = LLMConfig(extra={"simulate_latency_ms": 1.0})
        client = MockLLMClient(config)

        messages = [LLMMessage("user", "Hello")]
        response = await client.complete(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Mock LLM response."
        assert response.model_id == "mock-model-v1"
        assert response.provider == "mock"
        assert response.prompt_tokens == 50
        assert response.completion_tokens == 50
        assert response.latency_ms >= 0.0
        assert response.finish_reason == "stop"

    async def test_mock_success_multi_turn(self):
        config = LLMConfig(extra={"simulate_latency_ms": 1.0})
        client = MockLLMClient(config)

        messages = [
            LLMMessage("system", "You are an assistant."),
            LLMMessage("user", "Ping"),
            LLMMessage("assistant", "Pong"),
            LLMMessage("user", "What is next?"),
        ]
        response = await client.complete(messages)
        assert response.content == "Mock LLM response."

    async def test_mock_custom_response_and_scenario_mapping(self):
        config = LLMConfig(
            extra={
                "simulate_latency_ms": 1.0,
                "mock_response": "Default answer",
                "mock_responses": {
                    "Specific question": "Specific answer",
                },
            }
        )
        client = MockLLMClient(config)

        # Matched scenario
        res1 = await client.complete([LLMMessage("user", "Specific question")])
        assert res1.content == "Specific answer"

        # Fallback to mock_response
        res2 = await client.complete([LLMMessage("user", "Other question")])
        assert res2.content == "Default answer"

    async def test_mock_token_and_finish_reason_metadata(self):
        config = LLMConfig(
            extra={
                "simulate_latency_ms": 1.0,
                "prompt_tokens": 120,
                "completion_tokens": 45,
                "finish_reason": "length",
            }
        )
        client = MockLLMClient(config)
        response = await client.complete([LLMMessage("user", "Test metadata")])

        assert response.prompt_tokens == 120
        assert response.completion_tokens == 45
        assert response.finish_reason == "length"

    async def test_mock_simulated_auth_error(self):
        config = LLMConfig(extra={"raise_error": "auth"})
        client = MockLLMClient(config)

        with pytest.raises(LLMAuthenticationError, match="API key is missing or invalid"):
            await client.complete([LLMMessage("user", "Test")])

    async def test_mock_simulated_rate_limit_error(self):
        config = LLMConfig(extra={"raise_error": "rate_limit"})
        client = MockLLMClient(config)

        with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
            await client.complete([LLMMessage("user", "Test")])

    async def test_mock_simulated_timeout_error(self):
        config = LLMConfig(extra={"raise_error": "timeout"})
        client = MockLLMClient(config)

        with pytest.raises(LLMTimeoutError, match="request timed out"):
            await client.complete([LLMMessage("user", "Test")])

    async def test_mock_simulated_validation_error(self):
        config = LLMConfig(extra={"raise_error": "validation"})
        client = MockLLMClient(config)

        with pytest.raises(LLMValidationError, match="validation failed"):
            await client.complete([LLMMessage("user", "Test")])

    async def test_mock_simulated_generic_server_error(self):
        config = LLMConfig(extra={"raise_error": "internal_crash"})
        client = MockLLMClient(config)

        with pytest.raises(LLMError, match="simulated error 'internal_crash'"):
            await client.complete([LLMMessage("user", "Test")])


# ─── 2. Groq Adapter Tests ───────────────────────────────────────────────────

class TestGroqAdapter:
    """Covers Groq success parsing, error translation, and retry behavior."""

    @pytest.fixture(autouse=True)
    def setup_groq_key(self, monkeypatch):
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_groq_key_1234567890abcdef")
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    async def test_groq_success_parsing_default_model(self):
        config = LLMConfig()
        client = GroqLLMClient(config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Groq generated response"
        mock_choice.finish_reason = "stop"

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 24
        mock_usage.completion_tokens = 56

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama-3.3-70b-versatile"
        mock_completion.usage = mock_usage

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_groq_cls.return_value = mock_groq_instance

            response = await client.complete([LLMMessage("user", "Explain quantum computing")])

            assert response.content == "Groq generated response"
            assert response.model_id == "llama-3.3-70b-versatile"
            assert response.provider == "groq"
            assert response.prompt_tokens == 24
            assert response.completion_tokens == 56
            assert response.finish_reason == "stop"
            assert response.latency_ms >= 0.0

    async def test_groq_custom_model_and_system_prompt(self):
        config = LLMConfig(
            model_id="llama-3.1-8b-instant",
            system_prompt="You are a CSE tutor.",
            temperature=0.7,
            max_tokens=1024,
        )
        client = GroqLLMClient(config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Tutor response"
        mock_choice.finish_reason = "stop"

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama-3.1-8b-instant"
        mock_completion.usage = None

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_create = AsyncMock(return_value=mock_completion)
            mock_groq_instance.chat.completions.create = mock_create
            mock_groq_cls.return_value = mock_groq_instance

            response = await client.complete([LLMMessage("user", "Help me study")])

            assert response.content == "Tutor response"
            assert response.prompt_tokens == 0
            assert response.completion_tokens == 0

            # Verify system prompt was prepended in the call
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["model"] == "llama-3.1-8b-instant"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 1024
            assert call_kwargs["messages"][0] == {"role": "system", "content": "You are a CSE tutor."}
            assert call_kwargs["messages"][1] == {"role": "user", "content": "Help me study"}

    async def test_groq_empty_choices_raises_validation_error(self):
        config = LLMConfig()
        client = GroqLLMClient(config)

        mock_completion = MagicMock()
        mock_completion.choices = []

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(return_value=mock_completion)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMValidationError, match="Groq returned empty completion choices"):
                await client.complete([LLMMessage("user", "Hello")])

    async def test_groq_missing_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        get_settings.cache_clear()

        client = GroqLLMClient(LLMConfig())
        with pytest.raises(LLMAuthenticationError, match="GROQ_API_KEY environment variable is not set"):
            await client.complete([LLMMessage("user", "Hello")])

    async def test_groq_error_401_auth_error(self):
        import httpx
        from groq import APIStatusError

        config = LLMConfig(max_retries=1, extra={"retry_wait": wait_none()})
        client = GroqLLMClient(config)

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp = httpx.Response(status_code=401, request=req)
        err = APIStatusError(message="Invalid API Key", response=resp, body=None)

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(side_effect=err)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMAuthenticationError, match="Groq authentication failed"):
                await client.complete([LLMMessage("user", "Hello")])

            # Auth errors must NOT retry: exactly 1 call
            assert mock_groq_instance.chat.completions.create.call_count == 1

    async def test_groq_error_403_auth_error(self):
        import httpx
        from groq import APIStatusError

        config = LLMConfig(max_retries=1, extra={"retry_wait": wait_none()})
        client = GroqLLMClient(config)

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp = httpx.Response(status_code=403, request=req)
        err = APIStatusError(message="Forbidden access", response=resp, body=None)

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(side_effect=err)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMAuthenticationError, match="Groq authentication failed"):
                await client.complete([LLMMessage("user", "Hello")])

            assert mock_groq_instance.chat.completions.create.call_count == 1

    async def test_groq_error_500_server_error_no_retry(self):
        import httpx
        from groq import APIStatusError

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GroqLLMClient(config)

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp = httpx.Response(status_code=500, request=req)
        err = APIStatusError(message="Internal Server Error", response=resp, body=None)

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(side_effect=err)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMError, match="Groq API error 500"):
                await client.complete([LLMMessage("user", "Hello")])

            # Permanent server error should not retry
            assert mock_groq_instance.chat.completions.create.call_count == 1

    async def test_groq_api_timeout_error_retries_and_raises(self):
        import httpx
        from groq import APITimeoutError

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GroqLLMClient(config)

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        err = APITimeoutError(request=req)

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(side_effect=err)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMTimeoutError, match="Groq request timed out"):
                await client.complete([LLMMessage("user", "Hello")])

            # max_retries=2 means initial call + 2 retries = 3 calls
            assert mock_groq_instance.chat.completions.create.call_count == 3

    async def test_groq_rate_limit_recovers_on_retry(self):
        import httpx
        from groq import APIStatusError

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GroqLLMClient(config)

        req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        resp429 = httpx.Response(status_code=429, request=req)
        err429 = APIStatusError(message="Rate limit hit", response=resp429, body=None)

        mock_choice = MagicMock()
        mock_choice.message.content = "Recovered after 429"
        mock_choice.finish_reason = "stop"
        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        mock_completion.model = "llama-3.3-70b-versatile"
        mock_completion.usage = None

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            # Attempt 1: 429 error, Attempt 2: success
            mock_groq_instance.chat.completions.create = AsyncMock(
                side_effect=[err429, mock_completion]
            )
            mock_groq_cls.return_value = mock_groq_instance

            response = await client.complete([LLMMessage("user", "Hello")])

            assert response.content == "Recovered after 429"
            assert mock_groq_instance.chat.completions.create.call_count == 2


# ─── 3. Gemini Adapter Tests ─────────────────────────────────────────────────

class TestGeminiAdapter:
    """Covers Gemini success parsing, error translation, and retry behavior."""

    @pytest.fixture(autouse=True)
    def setup_gemini_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey1234567890abcdef")
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    async def test_gemini_success_parsing_default_model(self):
        config = LLMConfig()
        client = GeminiLLMClient(config)

        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "STOP"

        mock_usage = MagicMock()
        mock_usage.prompt_token_count = 32
        mock_usage.candidates_token_count = 78

        mock_response = MagicMock()
        mock_response.text = "Gemini answer"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = mock_usage

        with patch("google.generativeai.configure") as mock_configure, \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            response = await client.complete([LLMMessage("user", "Explain recursion")])

            mock_configure.assert_called_once_with(api_key="AIzaSyTestGeminiKey1234567890abcdef")
            assert response.content == "Gemini answer"
            assert response.model_id == "gemini-1.5-flash"
            assert response.provider == "gemini"
            assert response.prompt_tokens == 32
            assert response.completion_tokens == 78
            assert response.finish_reason == "stop"

    async def test_gemini_history_mapping_and_system_prompt(self):
        config = LLMConfig(
            model_id="gemini-1.5-pro",
            system_prompt="Act as a Python reviewer.",
        )
        client = GeminiLLMClient(config)

        mock_response = MagicMock()
        mock_response.text = "Code is good"
        mock_response.candidates = []
        mock_response.usage_metadata = None

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            messages = [
                LLMMessage("system", "Override system prompt in messages"),
                LLMMessage("user", "First question"),
                LLMMessage("assistant", "First answer"),
                LLMMessage("user", "Follow-up question"),
            ]
            response = await client.complete(messages)

            assert response.content == "Code is good"
            assert response.prompt_tokens == 0
            assert response.completion_tokens == 0

            # Verify start_chat was called with correctly mapped history
            history_arg = mock_model_instance.start_chat.call_args.kwargs["history"]
            assert history_arg == [
                {"role": "user", "parts": ["First question"]},
                {"role": "model", "parts": ["First answer"]},
            ]
            # Verify system instruction was passed
            model_kwargs = mock_gen_model.call_args.kwargs
            assert model_kwargs["model_name"] == "gemini-1.5-pro"
            assert model_kwargs["system_instruction"] == "Override system prompt in messages"

    async def test_gemini_blocked_candidate_fallback(self):
        """When response.text raises ValueError due to safety block, extract parts safely."""
        config = LLMConfig()
        client = GeminiLLMClient(config)

        # Simulates google-generativeai response when response.text raises ValueError
        class BlockedResponse:
            @property
            def text(self):
                raise ValueError("The Candidate was blocked due to SAFETY")

        mock_response = BlockedResponse()
        mock_part = MagicMock()
        mock_part.text = "Safe partial content"
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        mock_candidate.finish_reason.name = "SAFETY"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            response = await client.complete([LLMMessage("user", "Edge question")])
            assert response.content == "Safe partial content"
            assert response.finish_reason == "safety"

    async def test_gemini_missing_api_key_raises_auth_error(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        get_settings.cache_clear()

        client = GeminiLLMClient(LLMConfig())
        with pytest.raises(LLMAuthenticationError, match="GEMINI_API_KEY environment variable is not set"):
            await client.complete([LLMMessage("user", "Hello")])

    async def test_gemini_unauthenticated_error_no_retry(self):
        from google.api_core.exceptions import Unauthenticated

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GeminiLLMClient(config)

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = Unauthenticated("API key invalid")
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            with pytest.raises(LLMAuthenticationError, match="Gemini authentication failed"):
                await client.complete([LLMMessage("user", "Hello")])

            # Auth error must not retry
            assert mock_chat.send_message.call_count == 1

    async def test_gemini_permission_denied_error_no_retry(self):
        from google.api_core.exceptions import PermissionDenied

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GeminiLLMClient(config)

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = PermissionDenied("Method not allowed")
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            with pytest.raises(LLMAuthenticationError, match="Gemini authentication failed"):
                await client.complete([LLMMessage("user", "Hello")])

            assert mock_chat.send_message.call_count == 1

    async def test_gemini_deadline_exceeded_retries(self):
        from google.api_core.exceptions import DeadlineExceeded

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GeminiLLMClient(config)

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = DeadlineExceeded("Deadline exceeded: 504")
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            with pytest.raises(LLMTimeoutError, match="Gemini request timed out"):
                await client.complete([LLMMessage("user", "Hello")])

            # max_retries=2 -> 3 attempts
            assert mock_chat.send_message.call_count == 3

    async def test_gemini_resource_exhausted_retries_and_recovers(self):
        from google.api_core.exceptions import ResourceExhausted

        config = LLMConfig(max_retries=2, extra={"retry_wait": wait_none()})
        client = GeminiLLMClient(config)

        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "STOP"
        mock_response = MagicMock()
        mock_response.text = "Recovered response"
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = None

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = [
                ResourceExhausted("Rate limit 429"),
                mock_response,
            ]
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            response = await client.complete([LLMMessage("user", "Hello")])

            assert response.content == "Recovered response"
            assert mock_chat.send_message.call_count == 2


# ─── 4. Client Factory & Common Message Validation ───────────────────────────

class TestClientFactoryAndValidation:
    """Covers factory selection, unsupported providers, and input validations."""

    def test_factory_builds_mock(self):
        client = build_llm_client("mock", LLMConfig())
        assert isinstance(client, MockLLMClient)
        assert client.provider_name() == "mock"
        assert client.default_model_id() == "mock-model-v1"

    def test_factory_builds_groq(self):
        client = build_llm_client("groq", LLMConfig())
        assert isinstance(client, GroqLLMClient)
        assert client.provider_name() == "groq"
        assert client.default_model_id() == "llama-3.3-70b-versatile"

    def test_factory_builds_gemini(self):
        client = build_llm_client("gemini", LLMConfig())
        assert isinstance(client, GeminiLLMClient)
        assert client.provider_name() == "gemini"
        assert client.default_model_id() == "gemini-1.5-flash"

    def test_factory_case_insensitive_and_whitespace(self):
        c1 = build_llm_client("  MOCK  ", LLMConfig())
        assert isinstance(c1, MockLLMClient)

        c2 = build_llm_client("GrOq", LLMConfig())
        assert isinstance(c2, GroqLLMClient)

        c3 = build_llm_client("GeMiNi\n", LLMConfig())
        assert isinstance(c3, GeminiLLMClient)

    def test_factory_unknown_provider_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown LLM provider: 'anthropic'"):
            build_llm_client("anthropic", LLMConfig())

        with pytest.raises(ValueError, match="Unknown LLM provider: ''"):
            build_llm_client("", LLMConfig())

    async def test_validate_messages_empty_raises(self):
        client = MockLLMClient(LLMConfig())
        with pytest.raises(ValueError, match="messages must not be empty"):
            await client.complete([])

    async def test_validate_messages_last_not_user_raises(self):
        client = MockLLMClient(LLMConfig())
        messages = [
            LLMMessage("user", "Hello"),
            LLMMessage("assistant", "Hi there"),
        ]
        with pytest.raises(ValueError, match="last message.*must have role='user'"):
            await client.complete(messages)

    def test_llm_message_validations(self):
        # Valid roles
        m1 = LLMMessage("system", "instructions")
        m2 = LLMMessage("user", "question")
        m3 = LLMMessage("assistant", "answer")
        assert m1.role == "system"
        assert m2.role == "user"
        assert m3.role == "assistant"

        # Invalid roles
        with pytest.raises(ValueError, match="role must be one of"):
            LLMMessage("model", "text")

        with pytest.raises(ValueError, match="role must be one of"):
            LLMMessage("admin", "text")

        # Blank content
        with pytest.raises(ValueError, match="content must not be blank"):
            LLMMessage("user", "")

        with pytest.raises(ValueError, match="content must not be blank"):
            LLMMessage("user", "   \n\t  ")


# ─── 5. Secret Redaction & Leak Prevention ───────────────────────────────────

class TestSecretRedaction:
    """Covers Task 2 item 10: Ensuring secrets do not leak into logs or errors."""

    def test_sanitize_groq_api_key(self):
        raw = "Groq request failed with key gsk_1234567890abcdef1234567890 in header"
        sanitized = sanitize_error_message(raw)
        assert "gsk_1234567890abcdef1234567890" not in sanitized
        assert "[REDACTED]" in sanitized

    def test_sanitize_gemini_api_key(self):
        raw = "GoogleAPICallError: 403. URL: https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyD-abc123xyz_test45678901234"
        sanitized = sanitize_error_message(raw)
        assert "AIzaSyD-abc123xyz_test45678901234" not in sanitized
        assert "key=[REDACTED]" in sanitized

    def test_sanitize_query_param_key(self):
        raw = "Failure at https://api.example.com/v1?api_key=secret_value_123&other=true"
        sanitized = sanitize_error_message(raw)
        assert "secret_value_123" not in sanitized
        assert "api_key=[REDACTED]" in sanitized

    def test_sanitize_bearer_token(self):
        raw = "HTTP 401 Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        sanitized = sanitize_error_message(raw)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in sanitized
        assert "Bearer [REDACTED]" in sanitized

    def test_sanitize_explicit_additional_secrets(self):
        secret = "super_custom_secret_password_99"
        raw = f"DB connection failed with password {secret}"
        sanitized = sanitize_error_message(raw, additional_secrets=[secret])
        assert secret not in sanitized
        assert "[REDACTED]" in sanitized

    async def test_groq_error_does_not_leak_key_in_exception(self, monkeypatch):
        import httpx
        from groq import APIStatusError

        test_key = "gsk_supersecret_groq_key_99999999"
        monkeypatch.setenv("GROQ_API_KEY", test_key)
        get_settings.cache_clear()

        config = LLMConfig(max_retries=0)
        client = GroqLLMClient(config)

        req = httpx.Request("POST", f"https://api.groq.com/openai/v1/chat/completions?key={test_key}")
        resp = httpx.Response(
            status_code=401,
            request=req,
            text=f"Unauthorized: Invalid key {test_key}",
        )
        err = APIStatusError(message=f"Unauthorized with key {test_key}", response=resp, body=None)

        with patch("groq.AsyncGroq") as mock_groq_cls:
            mock_groq_instance = MagicMock()
            mock_groq_instance.chat.completions.create = AsyncMock(side_effect=err)
            mock_groq_cls.return_value = mock_groq_instance

            with pytest.raises(LLMAuthenticationError) as exc_info:
                await client.complete([LLMMessage("user", "Hello")])

            # Ensure secret does NOT appear anywhere in the raised exception string
            assert test_key not in str(exc_info.value)

    async def test_gemini_error_does_not_leak_key_in_exception(self, monkeypatch):
        from google.api_core.exceptions import Unauthenticated

        test_key = "AIzaSySecretGeminiKey1234567890123"
        monkeypatch.setenv("GEMINI_API_KEY", test_key)
        get_settings.cache_clear()

        config = LLMConfig(max_retries=0)
        client = GeminiLLMClient(config)

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_model_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = Unauthenticated(
                f"Invalid API key: {test_key} at endpoint https://.../?key={test_key}"
            )
            mock_model_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_model_instance

            with pytest.raises(LLMAuthenticationError) as exc_info:
                await client.complete([LLMMessage("user", "Hello")])

            assert test_key not in str(exc_info.value)

    def test_log_call_and_response_do_not_leak_content(self, caplog):
        caplog.set_level(logging.DEBUG)
        client = MockLLMClient(LLMConfig())

        secret_prompt = "My super confidential medical query"
        messages = [LLMMessage("user", secret_prompt)]
        client._log_call(messages)

        # Log records must not contain the prompt content
        for record in caplog.records:
            assert secret_prompt not in record.message
            assert "messages=1" in record.message

        caplog.clear()

        secret_answer = "Confidential diagnosis details"
        response = LLMResponse(
            content=secret_answer,
            model_id="mock-model-v1",
            provider="mock",
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=15.0,
            finish_reason="stop",
        )
        client._log_response(response)

        # Log records must not contain the response content
        for record in caplog.records:
            assert secret_answer not in record.message
            assert "tokens=10+20" in record.message

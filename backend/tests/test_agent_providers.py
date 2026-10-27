"""
tests/test_agent_providers.py
──────────────────────────────
Tests for VADER Stage 2:
- Target configuration resolution across agents (Planner, Executor A/B, Critic, Judge).
- Per-agent provider and model environment variable overrides.
- DebateOrchestrator role-specific client resolution.
- Gemini adapter runtime behavior: token overrides, safety block detection, rate limit handling, failure propagation.
- Zero network calls; all external APIs mocked.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.critic import CriticAgent
from app.agents.executor_a import ExecutorAAgent
from app.agents.executor_b import ExecutorBAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.config import Settings, get_settings
from app.llm.base import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
)
from app.llm.gemini_adapter import GeminiLLMClient
from app.llm.groq_adapter import GroqLLMClient
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import AgentRole, TaskPlan
from app.orchestrator.orchestrator import DebateOrchestrator


class TestTargetConfiguration:
    """Verifies default and overridden provider/model assignments per agent."""

    def test_mock_provider_defaults_all_agents_to_mock(self, monkeypatch):
        """When LLM_PROVIDER=mock, all agents must resolve to mock client."""
        monkeypatch.setenv("LLM_PROVIDER", "mock")
        get_settings.cache_clear()
        settings = get_settings()

        for role in (AgentRole.PLANNER, AgentRole.EXECUTOR_A, AgentRole.EXECUTOR_B, AgentRole.CRITIC, AgentRole.JUDGE):
            assert settings.get_agent_provider(role) == "mock"

    def test_target_configuration_non_mock(self, monkeypatch):
        """
        Target configuration:
          Planner: Groq, openai/gpt-oss-120b
          Executor A: Groq, openai/gpt-oss-120b
          Executor B: Gemini, gemini-3.5-flash-lite
          Critic: Gemini, gemini-3.5-flash-lite
          Judge: Groq, openai/gpt-oss-120b
        """
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.get_agent_provider(AgentRole.PLANNER) == "groq"
        assert settings.get_agent_model_id(AgentRole.PLANNER) == "openai/gpt-oss-120b"

        assert settings.get_agent_provider(AgentRole.EXECUTOR_A) == "groq"
        assert settings.get_agent_model_id(AgentRole.EXECUTOR_A) == "openai/gpt-oss-120b"

        assert settings.get_agent_provider(AgentRole.EXECUTOR_B) == "gemini"
        assert settings.get_agent_model_id(AgentRole.EXECUTOR_B) == "gemini-3.5-flash-lite"

        assert settings.get_agent_provider(AgentRole.CRITIC) == "gemini"
        assert settings.get_agent_model_id(AgentRole.CRITIC) == "gemini-3.5-flash-lite"

        assert settings.get_agent_provider(AgentRole.JUDGE) == "groq"
        assert settings.get_agent_model_id(AgentRole.JUDGE) == "openai/gpt-oss-120b"

    def test_per_agent_env_overrides(self, monkeypatch):
        """Custom environment variables can override individual agent assignments."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("EXECUTOR_B_PROVIDER", "mock")
        monkeypatch.setenv("EXECUTOR_B_MODEL_ID", "custom-mock-b")
        monkeypatch.setenv("CRITIC_PROVIDER", "groq")
        monkeypatch.setenv("CRITIC_MODEL_ID", "llama-3.3-70b-versatile")
        get_settings.cache_clear()
        settings = get_settings()

        assert settings.get_agent_provider(AgentRole.EXECUTOR_B) == "mock"
        assert settings.get_agent_model_id(AgentRole.EXECUTOR_B) == "custom-mock-b"

        assert settings.get_agent_provider(AgentRole.CRITIC) == "groq"
        assert settings.get_agent_model_id(AgentRole.CRITIC) == "llama-3.3-70b-versatile"

        # Non-overridden roles keep target defaults
        assert settings.get_agent_provider(AgentRole.PLANNER) == "groq"
        assert settings.get_agent_provider(AgentRole.JUDGE) == "groq"


class TestDebateOrchestratorResolution:
    """Verifies DebateOrchestrator instantiates role-specific clients."""

    def test_orchestrator_injected_shared_client_used_for_all_agents(self):
        """When an explicit llm_client is provided, all agents share it."""
        mock_client = MockLLMClient(LLMConfig())
        orchestrator = DebateOrchestrator(llm_client=mock_client)

        assert orchestrator.planner.client is mock_client
        assert orchestrator.executor_a.client is mock_client
        assert orchestrator.executor_b.client is mock_client
        assert orchestrator.critic.client is mock_client
        assert orchestrator.judge.client is mock_client

    def test_orchestrator_resolves_hybrid_clients(self, monkeypatch):
        """When llm_client is None in non-mock mode, orchestrator sets up hybrid agents."""
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "gsk_test_groq_key_1234567890abcdef")
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey1234567890abcdef")
        get_settings.cache_clear()

        orchestrator = DebateOrchestrator(llm_client=None)

        assert isinstance(orchestrator.planner.client, GroqLLMClient)
        assert orchestrator.planner.client.provider_name() == "groq"
        assert orchestrator.planner.client.effective_model_id == "openai/gpt-oss-120b"

        assert isinstance(orchestrator.executor_a.client, GroqLLMClient)
        assert orchestrator.executor_a.client.provider_name() == "groq"
        assert orchestrator.executor_a.client.effective_model_id == "openai/gpt-oss-120b"

        assert isinstance(orchestrator.executor_b.client, GeminiLLMClient)
        assert orchestrator.executor_b.client.provider_name() == "gemini"
        assert orchestrator.executor_b.client.effective_model_id == "gemini-3.5-flash-lite"

        assert isinstance(orchestrator.critic.client, GeminiLLMClient)
        assert orchestrator.critic.client.provider_name() == "gemini"
        assert orchestrator.critic.client.effective_model_id == "gemini-3.5-flash-lite"

        assert isinstance(orchestrator.judge.client, GroqLLMClient)
        assert orchestrator.judge.client.provider_name() == "groq"
        assert orchestrator.judge.client.effective_model_id == "openai/gpt-oss-120b"


class TestGeminiAdapterBehavior:
    """Verifies Gemini adapter features: generation config, safety blocks, failure propagation."""

    @pytest.fixture(autouse=True)
    def setup_gemini_key(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "AIzaSyTestGeminiKey1234567890abcdef")
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    async def test_gemini_token_budget_and_temperature_override(self):
        """max_tokens and temperature overrides must pass to GenerationConfig."""
        client = GeminiLLMClient(LLMConfig(model_id="gemini-3.5-flash-lite", temperature=0.7, max_tokens=2048))

        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "STOP"
        mock_candidate.content.parts = [MagicMock(text="Structured response")]

        mock_response = MagicMock()
        mock_response.text = '{"status": "ok"}'
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock(prompt_token_count=10, candidates_token_count=20)
        mock_response.prompt_feedback = None

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_instance

            # Call with overrides: max_tokens=512, temperature=0.1
            resp = await client.complete(
                [LLMMessage("user", "Hello")],
                max_tokens=512,
                temperature=0.1,
            )

            assert resp.content == '{"status": "ok"}'
            assert resp.model_id == "gemini-3.5-flash-lite"
            assert resp.prompt_tokens == 10
            assert resp.completion_tokens == 20

            # Verify generation_config was passed with the overrides
            gen_config = mock_gen_model.call_args.kwargs.get("generation_config")
            assert gen_config is not None
            assert getattr(gen_config, "temperature", None) == 0.1 or gen_config.temperature == 0.1
            assert getattr(gen_config, "max_output_tokens", None) == 512 or gen_config.max_output_tokens == 512

    async def test_gemini_candidate_safety_block_raises_llm_error(self):
        """When candidate generation is blocked by safety filters, must raise LLMError."""
        client = GeminiLLMClient(LLMConfig(model_id="gemini-3.5-flash-lite"))

        mock_candidate = MagicMock()
        mock_candidate.finish_reason.name = "SAFETY"
        mock_candidate.content.parts = []

        class SafetyBlockedResponse:
            candidates = [mock_candidate]
            usage_metadata = None
            prompt_feedback = None
            @property
            def text(self):
                raise ValueError("Candidate blocked due to SAFETY")

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = SafetyBlockedResponse()
            mock_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_instance

            with pytest.raises(LLMError, match="blocked by safety filters"):
                await client.complete([LLMMessage("user", "Controversial prompt")])

    async def test_gemini_prompt_safety_block_raises_llm_error(self):
        """When prompt feedback indicates a block, must raise LLMError."""
        client = GeminiLLMClient(LLMConfig(model_id="gemini-3.5-flash-lite"))

        mock_feedback = MagicMock()
        mock_feedback.block_reason = "PROHIBITED_CONTENT"

        mock_response = MagicMock()
        mock_response.prompt_feedback = mock_feedback

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.return_value = mock_response
            mock_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_instance

            with pytest.raises(LLMError, match="Gemini prompt blocked by safety filters"):
                await client.complete([LLMMessage("user", "Blocked content")])

    async def test_gemini_rate_limit_extracts_retry_after(self):
        """Gemini ResourceExhausted translates to LLMRateLimitError with retry_after."""
        from google.api_core.exceptions import ResourceExhausted

        client = GeminiLLMClient(LLMConfig(model_id="gemini-3.5-flash-lite", max_retries=0))

        with patch("google.generativeai.configure"), \
             patch("google.generativeai.GenerativeModel") as mock_gen_model:

            mock_instance = MagicMock()
            mock_chat = MagicMock()
            mock_chat.send_message.side_effect = ResourceExhausted("Quota exceeded: Please try again in 12.5s.")
            mock_instance.start_chat.return_value = mock_chat
            mock_gen_model.return_value = mock_instance

            with pytest.raises(LLMRateLimitError) as exc_info:
                await client.complete([LLMMessage("user", "High load")])

            err = exc_info.value
            assert err.retry_after == 12.5
            assert "Gemini rate limit" in str(err)

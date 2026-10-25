"""
app/llm/mock_adapter.py
────────────────────────
Deterministic mock LLM client for local development and automated tests.

The mock:
  - Requires no API keys or network access.
  - Returns predictable, configurable responses.
  - Simulates latency, token counts, and finish reasons.
  - Can be configured to raise specific exceptions to test error paths.
"""

from __future__ import annotations

import asyncio
import json
import time

from app.llm.base import (
    BaseLLMClient,
    LLMAuthenticationError,
    LLMError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
)


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM adapter for testing.

    Constructor parameters (via LLMConfig.extra dict)
    --------------------------------------------------
    mock_response : str
        Fixed string returned for every call (default: "Mock LLM response.").
    mock_responses : dict[str, str]
        Keyed by the last user message content; used for scenario-specific
        responses.  Falls back to mock_response if key is not found.
    simulate_latency_ms : float
        Artificial async sleep (default 10 ms).
    raise_error : str | None
        If set, raise the named exception class instead of returning a response.
        Values: "auth", "rate_limit", "timeout", "llm".
    mock_tokens : int
        Simulated token count (default 50 each for prompt and completion).
    """

    _DEFAULT_RESPONSE = "Mock LLM response."
    _DEFAULT_LATENCY_MS = 10.0
    _DEFAULT_TOKENS = 50

    def provider_name(self) -> str:
        return "mock"

    def default_model_id(self) -> str:
        return "mock-model-v1"

    async def complete(self, messages: list[LLMMessage]) -> LLMResponse:
        self._validate_messages(messages)
        self._log_call(messages)

        extra: dict = self._config.extra or {}

        # ── Simulate configurable errors ──────────────────────────────────
        raise_error: str | None = extra.get("raise_error")
        if raise_error:
            match raise_error:
                case "auth":
                    raise LLMAuthenticationError(
                        "Mock: API key is missing or invalid."
                    )
                case "rate_limit":
                    raise LLMRateLimitError(
                        "Mock: rate limit exceeded."
                    )
                case "timeout":
                    raise LLMTimeoutError(
                        "Mock: request timed out."
                    )
                case "validation":
                    raise LLMValidationError(
                        "Mock: response validation failed."
                    )
                case _:
                    raise LLMError(f"Mock: simulated error '{raise_error}'.")

        # ── Simulate async latency ────────────────────────────────────────
        latency_ms: float = extra.get(
            "simulate_latency_ms", self._DEFAULT_LATENCY_MS
        )
        t_start = time.monotonic()
        await asyncio.sleep(latency_ms / 1000.0)
        actual_latency_ms = (time.monotonic() - t_start) * 1000

        # ── Determine response content ────────────────────────────────────
        last_user_content = messages[-1].content
        scenario_map: dict[str, str] = extra.get("mock_responses", {})
        if last_user_content in scenario_map:
            content = scenario_map[last_user_content]
        elif "mock_response" in extra:
            content = extra["mock_response"]
        else:
            # Context-aware mock JSON based on calling agent's exact system prompt header
            sys_text = messages[0].content if messages and messages[0].role == "system" else ""
            if "You are the Planner Agent" in sys_text:
                from app.agents.planner import PlannerAgent
                content = PlannerAgent.mock_output_json()
            elif "You are Executor A" in sys_text:
                from app.agents.executor_a import ExecutorAAgent
                d = ExecutorAAgent.mock_output_dict()
                if "Revise and strengthen" in messages[-1].content:
                    d["answer"] += " (revised with clarified definitions and boundary conditions)"
                content = json.dumps(d)
            elif "You are Executor B" in sys_text:
                from app.agents.executor_b import ExecutorBAgent
                d = ExecutorBAgent.mock_output_dict()
                if "Revise and strengthen" in messages[-1].content:
                    d["answer"] += " (revised with empirical nuance and counter-perspectives)"
                content = json.dumps(d)
            elif "You are the Critic Agent" in sys_text:
                from app.agents.critic import CriticAgent
                content = CriticAgent.mock_output_json()
            elif "You are the Judge Agent" in sys_text:
                from app.agents.judge import JudgeAgent
                content = JudgeAgent.mock_output_json()
            elif "You are the Claim Verification Service" in sys_text:
                content = json.dumps({
                    "results": [
                        {
                            "claim": "Verified empirical claim",
                            "status": "supported",
                            "evidence_refs": [],
                            "explanation": "Supported by retrieved evidence.",
                            "uncertainty": None,
                        }
                    ],
                    "summary": "Mock verification complete.",
                    "evidence_count": 0,
                })
            else:
                content = self._DEFAULT_RESPONSE

        default_tokens: int = extra.get("mock_tokens", self._DEFAULT_TOKENS)
        prompt_tokens: int = extra.get("prompt_tokens", default_tokens)
        completion_tokens: int = extra.get("completion_tokens", default_tokens)
        finish_reason: str = extra.get("finish_reason", "stop")

        response = LLMResponse(
            content=content,
            model_id=self.effective_model_id,
            provider=self.provider_name(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=actual_latency_ms,
            finish_reason=finish_reason,
        )

        self._log_response(response)
        return response

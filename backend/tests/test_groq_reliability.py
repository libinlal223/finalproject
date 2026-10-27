"""
tests/test_groq_reliability.py
───────────────────────────────
Targeted tests for Groq reliability improvements:
1. Parsing of HTTP 429 response details (headers, reset seconds, token metrics, limit types).
2. Futile retry prevention (immediate abort when requested tokens exceed model ceiling or timeout).
3. Adaptive retry wait honoring provider-supplied retry-after intervals.
4. Agent token budget right-sizing across Planner, Executors, Verifier, Critic, and Judge.
5. Critic prompt compactness (pruning redundant claims and compact formatting).
6. Honest failure tracking and uncertainty preservation when agents fail.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import wait_none

from app.agents.critic import CriticAgent
from app.agents.executor_a import ExecutorAAgent
from app.agents.executor_b import ExecutorBAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.llm.base import (
    LLMConfig,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
)
from app.llm.groq_adapter import GroqLLMClient, _parse_groq_rate_limit
from app.models.schemas import (
    AgentRole,
    ClaimStatus,
    ClaimVerificationResult,
    DebateRound,
    DebateStatus,
    EvidenceItem,
    ExecutorOutput,
    TaskPlan,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService


# ─── 1. HTTP 429 Error Parsing Tests ──────────────────────────────────────────

def test_parse_groq_rate_limit_from_message_text():
    """Extracts TPM limit, used tokens, requested tokens, and retry-after from Groq error message."""
    raw_error_message = (
        "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_abc123` "
        "on tokens per minute (TPM): Limit 8000, Used 4371, Requested 4784. "
        "Please try again in 23.5s. Visit https://console.groq.com/docs/rate-limits"
    )
    exc = Exception(raw_error_message)
    parsed = _parse_groq_rate_limit(exc)

    assert isinstance(parsed, LLMRateLimitError)
    assert parsed.limit_tokens == 8000
    assert parsed.used_tokens == 4371
    assert parsed.requested_tokens == 4784
    assert parsed.retry_after == 23.5
    assert parsed.limit_type == "tokens"


def test_parse_groq_rate_limit_from_response_headers():
    """Extracts retry_after and limit headers when present on HTTP response object."""
    mock_response = MagicMock()
    mock_response.headers = {
        "retry-after": "18.2",
        "x-ratelimit-limit-tokens": "8000",
        "x-ratelimit-reset-tokens": "18.2s",
    }
    exc = Exception("429 Too Many Requests")
    exc.response = mock_response

    parsed = _parse_groq_rate_limit(exc)
    assert parsed.retry_after == 18.2
    assert parsed.limit_tokens == 8000


def test_parse_groq_rate_limit_requests_rpm_type():
    """Correctly categorizes RPM (requests per minute) rate limits."""
    raw_error_message = (
        "Rate limit reached for model `llama-3.3-70b-versatile` on requests per minute (RPM): "
        "Limit 30, Used 30, Requested 1. Please try again in 4.2s."
    )
    parsed = _parse_groq_rate_limit(Exception(raw_error_message))
    assert parsed.limit_type == "requests"
    assert parsed.retry_after == 4.2


# ─── 2. Futile Retry Prevention & Adaptive Backoff Tests ─────────────────────

@pytest.mark.asyncio
async def test_futile_retry_aborted_when_tokens_requested_exceed_limit():
    """When a single request's requested tokens alone exceed total limit, abort immediately without retrying."""
    config = LLMConfig(
        max_retries=3,
        extra={"retry_wait": wait_none()},
    )
    client = GroqLLMClient(config)

    # 8,500 requested > 8,000 limit -> impossible to ever succeed
    futile_msg = (
        "Rate limit reached on tokens per minute (TPM): Limit 8000, Used 0, Requested 8500. "
        "Please try again in 5.0s."
    )
    call_count = 0

    async def _mock_single_call(messages, model_id, max_tokens=None, temperature=None):
        nonlocal call_count
        call_count += 1
        raise _parse_groq_rate_limit(Exception(futile_msg))

    client._single_call = _mock_single_call

    with pytest.raises(LLMRateLimitError) as exc_info:
        await client.complete([LLMMessage(role="user", content="Analyze complex data")])

    # Must fail on the FIRST attempt; no futile 3 retries
    assert call_count == 1
    assert exc_info.value.requested_tokens == 8500
    assert exc_info.value.limit_tokens == 8000


@pytest.mark.asyncio
async def test_futile_retry_aborted_when_retry_after_exceeds_budget():
    """When retry_after exceeds reasonable wait budget (45s), abort immediately."""
    config = LLMConfig(
        max_retries=3,
        extra={"retry_wait": wait_none()},
    )
    client = GroqLLMClient(config)

    excessive_wait_msg = "Rate limit reached. Please try again in 75.0s."
    call_count = 0

    async def _mock_single_call(messages, model_id, max_tokens=None, temperature=None):
        nonlocal call_count
        call_count += 1
        raise _parse_groq_rate_limit(Exception(excessive_wait_msg))

    client._single_call = _mock_single_call

    with pytest.raises(LLMRateLimitError) as exc_info:
        await client.complete([LLMMessage(role="user", content="Test")])

    assert call_count == 1
    assert exc_info.value.retry_after == 75.0


@pytest.mark.asyncio
async def test_groq_client_passes_max_tokens_override():
    """Groq client accepts and passes max_tokens and temperature overrides to single_call."""
    client = GroqLLMClient(LLMConfig(max_tokens=2048, temperature=0.3))

    recorded_kwargs = {}

    async def _mock_single_call(messages, model_id, max_tokens=None, temperature=None):
        recorded_kwargs["max_tokens"] = max_tokens
        recorded_kwargs["temperature"] = temperature
        return LLMResponse(
            content="{}",
            model_id=model_id,
            provider="groq",
            prompt_tokens=10,
            completion_tokens=10,
        )

    client._single_call = _mock_single_call

    await client.complete(
        [LLMMessage(role="user", content="Test")],
        max_tokens=512,
        temperature=0.1,
    )

    assert recorded_kwargs["max_tokens"] == 512
    assert recorded_kwargs["temperature"] == 0.1


# ─── 3. Agent Output Token Budget Right-Sizing Tests ─────────────────────────

def test_agent_token_budgets_are_right_sized():
    """Verify that agent default configurations have right-sized token limits to protect Groq TPM."""
    planner_cfg = PlannerAgent.default_config()
    assert planner_cfg.max_tokens <= 1024, "Planner output requires token budget <=1024 to prevent JSON truncation"

    critic_cfg = CriticAgent.default_config()
    assert critic_cfg.max_tokens <= 2560, "Critic output requires expanded token budget <=2560 to prevent JSON truncation"

    verifier = ClaimVerificationService()
    assert verifier._config.max_tokens <= 1024, "Claim verifier output is structured results; <=1024 safe"

    exec_a_cfg = ExecutorAAgent.default_config()
    assert exec_a_cfg.max_tokens <= 1536, "Executor A output <=1536 safe"

    exec_b_cfg = ExecutorBAgent.default_config()
    assert exec_b_cfg.max_tokens <= 1536, "Executor B output <=1536 safe"

    judge_cfg = JudgeAgent.default_config()
    assert judge_cfg.max_tokens <= 2560, "Judge output <=2560 safe"


# ─── 4. Critic Prompt Compactness Tests ───────────────────────────────────────

@pytest.mark.asyncio
async def test_critic_prompt_removes_duplicate_claims_to_verify():
    """Critic prompt does NOT duplicate claims_to_verify JSON array from executors."""
    critic = CriticAgent()

    captured_prompt = ""

    async def _mock_execute_prompt(user_prompt, schema, system_prompt=None):
        nonlocal captured_prompt
        captured_prompt = user_prompt
        return CriticAgent.mock_output_dict(), LLMResponse(
            content=CriticAgent.mock_output_json(),
            model_id="mock-model",
            provider="mock",
        )

    critic._execute_prompt = _mock_execute_prompt

    plan = TaskPlan(
        core_question="Is quantum supremacy achieved?",
        subtasks=["Analyze fidelity", "Evaluate benchmark circuits"],
        constraints=["Exclude unverified preprints"],
        evidence_needed=["Quantum supremacy benchmark"],
    )

    cand_a = ExecutorOutput(
        executor_id="executor_a",
        approach_summary="First-principles physics",
        answer="Quantum supremacy is operational in specialized sampling tasks.",
        key_assumptions=["Cross-entropy benchmarking is valid"],
        claims_to_verify=["Supremacy in random circuit sampling"],
    )
    cand_b = ExecutorOutput(
        executor_id="executor_b",
        approach_summary="Classical simulation rebuttal",
        answer="Classical tensor networks can replicate results in reasonable time.",
        key_assumptions=["Classical cluster scaling holds"],
        claims_to_verify=["Tensor networks simulate in 14 hours"],
    )

    verifications = [
        ClaimVerificationResult(
            claim_id="CLAIM-1",
            claim_text="Supremacy in random circuit sampling",
            status=ClaimStatus.SUPPORTED,
            explanation="Corroborated by peer-reviewed experimental metrics.",
            evidence=[],
            limitations=None,
        )
    ]

    await critic.critique(
        query="Is quantum supremacy achieved?",
        plan=plan,
        executor_a=cand_a,
        executor_b=cand_b,
        verifications=verifications,
    )

    # Both answers MUST be present
    assert cand_a.answer in captured_prompt
    assert cand_b.answer in captured_prompt

    # Structured verification findings MUST be present
    assert "CLAIM-1" in captured_prompt
    assert "SUPPORTED" in captured_prompt

    # Redundant raw claims_to_verify JSON dumps MUST be pruned
    assert "Claims to verify:" not in captured_prompt


# ─── 5. Honest Verification & Failure Tracking Tests ─────────────────────────

@pytest.mark.asyncio
async def test_critic_failure_preserves_honest_error_and_uncertainty():
    """If Critic agent fails, response explicitly populates error_message and marks answer as unverified."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = RuntimeError("Groq HTTP 429: Rate limit exceeded")

    orchestrator = DebateOrchestrator(critic=mock_critic)
    resp = await orchestrator.run_debate("Test query on macroeconomic policy", max_rounds=1)

    # Must preserve fallback answer honestly without pretending critique succeeded
    assert resp.final_answer is not None
    assert resp.error_message is not None
    assert "Critic agent failed" in resp.error_message
    assert any("Unverified candidate" in unc for unc in resp.unresolved_uncertainties)
    assert "Critique step failed" in (resp.verdict_summary or "")


@pytest.mark.asyncio
async def test_judge_failure_preserves_honest_error_message():
    """If Judge fails, error_message is recorded and epistemic uncertainty explicitly logged."""
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.side_effect = RuntimeError("Groq HTTP 429: TPM limit reached")

    orchestrator = DebateOrchestrator(judge=mock_judge)
    resp = await orchestrator.run_debate("Quantum computing threat", max_rounds=1)

    assert resp.final_answer is not None
    assert resp.error_message is not None
    assert "Judge adjudication failed" in resp.error_message
    assert any("Adjudication fallback" in unc for unc in resp.unresolved_uncertainties)

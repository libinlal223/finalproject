"""
tests/test_planner_reliability.py
──────────────────────────────────
Regression tests for Planner reliability and fallback behavior:
1. Planner max_tokens is right-sized to 1024.
2. A Planner LLMValidationError triggers a valid fallback TaskPlan and allows execution to proceed.
3. Planner timeout (LLMTimeoutError) and provider failure (LLMError) trigger recoverable fallback plans.
4. Unrelated programming errors (e.g., TypeError) are NOT swallowed silently.
5. The degraded planning warning is recorded in trace and preserved in unresolved_uncertainties.
6. Existing behavior remains unchanged when Planner succeeds.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.llm.base import (
    LLMConfig,
    LLMError,
    LLMResponse,
    LLMTimeoutError,
    LLMValidationError,
)
from app.models.schemas import (
    AgentRole,
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    DebateStatus,
    JudgeOutput,
    RetrievalResponse,
    RetrievalStatus,
    TaskPlan,
    VerificationReportStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.service import EvidenceRetrievalService


# ─── Fixtures & Helpers ───────────────────────────────────────────────────────

def _make_mock_retrieval() -> EvidenceRetrievalService:
    mock_provider = AsyncMock()
    mock_provider.provider_name = lambda: "mock"
    mock_provider.search.return_value = RetrievalResponse(
        query="test",
        status=RetrievalStatus.SUCCESS,
        items=[],
        total_results=0,
    )
    return EvidenceRetrievalService(provider=mock_provider)


def _make_mock_verifier() -> ClaimVerificationService:
    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=[],
        summary="No claims to verify.",
        evidence_count=0,
        claims_checked=0,
        claims_not_checked=0,
    )
    return mock_verifier


def _make_mock_judge() -> JudgeAgent:
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer="Adjudicated response.",
            resolved_points=["Point 1 resolved."],
            unresolved_uncertainties=["Inherent domain uncertainty."],
            limitations=[],
            evidence_citations=[],
            verdict_summary="Adjudication complete.",
            adjudication_rationale="Balanced synthesis.",
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=5.0),
    )
    return mock_judge


# ─── 1. Token Budget Right-Sizing Test ────────────────────────────────────────

def test_planner_max_tokens_is_1024():
    """Verify that PlannerAgent.default_config() allocates 1024 max_tokens."""
    cfg = PlannerAgent.default_config()
    assert cfg.max_tokens == 1024, f"Expected Planner max_tokens=1024, got {cfg.max_tokens}"


# ─── 2. Planner LLMValidationError Fallback & Continuation ───────────────────

@pytest.mark.asyncio
async def test_planner_llm_validation_error_uses_fallback_plan_and_proceeds():
    """When Planner fails with truncation/validation error, fallback plan is used and debate completes."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.side_effect = LLMValidationError(
        "Agent 'planner' output was truncated due to token limit budget (finish_reason='length')."
    )

    query = "Does quantum computing pose a near-term threat to RSA-2048 and ECC-256 cryptography?"
    orchestrator = DebateOrchestrator(
        planner=mock_planner,
        judge=_make_mock_judge(),
        claim_verifier=_make_mock_verifier(),
        retrieval_service=_make_mock_retrieval(),
    )

    resp = await orchestrator.run_debate(query, max_rounds=1)

    # Must NOT abort with DebateStatus.ERROR
    assert resp.status == DebateStatus.COMPLETED
    assert resp.rounds_completed >= 1
    assert resp.final_answer is not None

    # Fallback plan must be schema-valid and derived directly from query
    assert resp.subtasks == [f"Analyze and address: {query}"]

    # Trace must explicitly record warning and fallback plan
    assert resp.trace[0].agent == AgentRole.PLANNER
    assert "[WARNING: Automated planning failed" in resp.trace[0].output
    assert "Using minimal fallback plan derived directly from user query" in resp.trace[0].output

    # Unresolved uncertainties must disclose degraded planning
    assert any("Degraded planning" in u for u in resp.unresolved_uncertainties)
    assert any("finish_reason='length'" in u for u in resp.unresolved_uncertainties)


# ─── 3. Planner Timeout Error Per Existing Policy ─────────────────────────────

@pytest.mark.asyncio
async def test_planner_timeout_error_per_existing_recoverability_policy():
    """When Planner times out, orchestrator cleanly terminates with ERROR status per existing policy."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.side_effect = LLMTimeoutError("Groq request timed out after 60s.")

    query = "Is India a 3rd world country?"
    orchestrator = DebateOrchestrator(
        planner=mock_planner,
        judge=_make_mock_judge(),
        claim_verifier=_make_mock_verifier(),
        retrieval_service=_make_mock_retrieval(),
    )

    resp = await orchestrator.run_debate(query, max_rounds=1)

    assert resp.status == DebateStatus.ERROR
    assert resp.rounds_completed == 0
    assert resp.final_answer is None
    assert "Planner agent failure" in resp.error_message
    assert any("Planning failed" in entry.output for entry in resp.trace)


# ─── 4. Planner Provider Failure Per Existing Policy ──────────────────────────

@pytest.mark.asyncio
async def test_planner_provider_failure_per_existing_recoverability_policy():
    """When Planner suffers provider authentication or connection failure, terminates with ERROR."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.side_effect = LLMError("Groq connection error: Connection refused to api.groq.com key=sk-secret-999")

    query = "Evaluating renewable baseload potential"
    orchestrator = DebateOrchestrator(
        planner=mock_planner,
        judge=_make_mock_judge(),
        claim_verifier=_make_mock_verifier(),
        retrieval_service=_make_mock_retrieval(),
    )

    resp = await orchestrator.run_debate(query, max_rounds=1)

    assert resp.status == DebateStatus.ERROR
    assert resp.rounds_completed == 0
    assert resp.final_answer is None
    assert "Planner agent failure" in resp.error_message
    assert "sk-secret-999" not in resp.error_message
    assert any("Planning failed" in entry.output for entry in resp.trace)


@pytest.mark.asyncio
async def test_planner_unrelated_programming_error_is_not_swallowed():
    """Unrelated programming errors (e.g. TypeError) must not be treated as recoverable fallback and must report ERROR."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.side_effect = TypeError("Unexpected argument in planner internal routine")

    query = "Test query for programming error bubbling"
    orchestrator = DebateOrchestrator(
        planner=mock_planner,
        judge=_make_mock_judge(),
        claim_verifier=_make_mock_verifier(),
        retrieval_service=_make_mock_retrieval(),
    )

    resp = await orchestrator.run_debate(query, max_rounds=1)

    assert resp.status == DebateStatus.ERROR
    assert resp.rounds_completed == 0
    assert resp.final_answer is None
    assert "Planner agent failure" in resp.error_message
    assert "Unexpected argument in planner internal routine" in resp.error_message


# ─── 6. Planner Success Preserves Normal Behavior ────────────────────────────

@pytest.mark.asyncio
async def test_planner_success_preserves_standard_behavior():
    """When Planner succeeds, normal subtasks and trace are preserved without fallback warnings."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.return_value = (
        TaskPlan(
            core_question="What are the quantum threats to cryptography?",
            subtasks=[
                "Assess Shor's algorithm requirements",
                "Evaluate post-quantum migration standards",
            ],
            constraints=["Rely on NIST guidelines"],
            expected_format="structured_synthesis",
            evidence_needed=["Shor algorithm physical qubits required"],
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=10.0),
    )

    query = "What are the quantum threats to cryptography?"
    orchestrator = DebateOrchestrator(
        planner=mock_planner,
        judge=_make_mock_judge(),
        claim_verifier=_make_mock_verifier(),
        retrieval_service=_make_mock_retrieval(),
    )

    resp = await orchestrator.run_debate(query, max_rounds=1)

    assert resp.status == DebateStatus.COMPLETED
    assert resp.subtasks == [
        "Assess Shor's algorithm requirements",
        "Evaluate post-quantum migration standards",
    ]
    # No warning banner in trace
    assert "[WARNING:" not in resp.trace[0].output
    # No degraded planning uncertainty
    assert not any("Degraded planning" in u for u in resp.unresolved_uncertainties)

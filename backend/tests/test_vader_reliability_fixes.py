"""
tests/test_vader_reliability_fixes.py
──────────────────────────────────────
Regression tests for VADER reliability improvements:
  1. Judge output truncation and bounded valid JSON recovery.
  2. Strict validation preserved on Judge output (never accepts malformed JSON).
  3. Retrieval timeouts and partial retrieval failure isolation.
  4. Zero retrieved sources and NOT_CHECKED claims preservation.
  5. Safe unadjudicated fallback behavior when Judge fails.
  6. Normal successful end-to-end debate with accurate round counts and trace.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.critic import CriticAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.llm.base import (
    LLMConfig,
    LLMError,
    LLMMessage,
    LLMResponse,
    LLMValidationError,
)
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import (
    AgentRole,
    AgentTraceEntry,
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    CriticOutput,
    DebateRound,
    DebateStatus,
    EvidenceItem,
    ExecutorOutput,
    JudgeOutput,
    QueryResponse,
    RetrievalResponse,
    RetrievalStatus,
    TaskPlan,
    VerificationReportStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.service import EvidenceRetrievalService
from app.retrieval.targeted_retrieval import retrieve_for_claims


# ─── 1. Judge JSON Truncation and Valid Recovery ──────────────────────────────

@pytest.mark.asyncio
async def test_judge_truncation_recovers_with_valid_json_on_retry():
    """Judge output truncated on first call retries once with expanded tokens and recovers valid JSON."""
    client = MockLLMClient(LLMConfig())
    judge = JudgeAgent(client=client)

    call_count = 0
    passed_max_tokens = []

    mock_valid_json = JudgeAgent.mock_output_json("Quantum Cryptography Threat")

    async def _mock_complete(messages: list[LLMMessage], **kwargs):
        nonlocal call_count
        call_count += 1
        passed_max_tokens.append(kwargs.get("max_tokens"))
        if call_count == 1:
            # Simulate Groq 400 json_validate_failed / truncation error
            raise LLMValidationError(
                "Agent 'judge' output was truncated due to token limit budget (finish_reason='length')."
            )
        return LLMResponse(
            content=mock_valid_json,
            model_id="mock-model",
            provider="mock",
            finish_reason="stop",
        )

    judge._client.complete = _mock_complete

    plan = TaskPlan(core_question="Is RSA-2048 at risk?", subtasks=["Analyze qubits"])
    exec_a = ExecutorOutput(executor_id="executor_a", approach_summary="Theory", answer="Answer A")
    exec_b = ExecutorOutput(executor_id="executor_b", approach_summary="Practice", answer="Answer B")
    critic = CriticOutput.model_validate(CriticAgent.mock_output_dict())

    output, response = await judge.judge(
        query="Is RSA-2048 at risk?",
        plan=plan,
        executor_a=exec_a,
        executor_b=exec_b,
        critic=critic,
    )

    assert call_count == 2
    assert passed_max_tokens[1] > passed_max_tokens[0]
    assert isinstance(output, JudgeOutput)
    assert output.preferred_candidate == "synthesis"


@pytest.mark.asyncio
async def test_judge_truncation_persisting_raises_validation_error_without_accepting_malformed():
    """If Judge truncation persists through retry, it must raise without accepting malformed output."""
    client = MockLLMClient(LLMConfig())
    judge = JudgeAgent(client=client)

    call_count = 0

    async def _mock_complete(messages: list[LLMMessage], **kwargs):
        nonlocal call_count
        call_count += 1
        raise LLMError("Groq API error 400: json_validate_failed: max completion tokens reached")

    judge._client.complete = _mock_complete

    plan = TaskPlan(core_question="Test query", subtasks=["Analyze"])
    exec_a = ExecutorOutput(executor_id="executor_a", approach_summary="A", answer="Answer A")
    exec_b = ExecutorOutput(executor_id="executor_b", approach_summary="B", answer="Answer B")
    critic = CriticOutput.model_validate(CriticAgent.mock_output_dict())

    with pytest.raises((LLMError, LLMValidationError)) as exc_info:
        await judge.judge(
            query="Test query",
            plan=plan,
            executor_a=exec_a,
            executor_b=exec_b,
            critic=critic,
        )

    assert "json_validate_failed" in str(exc_info.value).lower()
    assert call_count == 2  # Attempted initial + 1 bounded recovery


# ─── 2. Retrieval Timeouts and Partial Isolation ─────────────────────────────

@pytest.mark.asyncio
async def test_targeted_retrieval_one_failure_does_not_abort_other_claims():
    """One claim timing out or failing does not abort searches for other claims."""
    mock_service = AsyncMock(spec=EvidenceRetrievalService)

    async def _mock_retrieve(query: str, max_results: int = 2):
        if "rsa" in query.lower():
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.SUCCESS,
                items=[
                    EvidenceItem(
                        title="RSA Threat Research",
                        url="https://example.com/rsa-threat",
                        snippet="Breaking RSA-2048 requires millions of physical qubits.",
                        domain="example.com",
                    )
                ],
                total_results=1,
                latency_ms=10.0,
            )
        # Other claim times out
        return RetrievalResponse(
            query=query,
            status=RetrievalStatus.TIMEOUT,
            items=[],
            total_results=0,
            error_message="Search request timed out after 10s.",
            latency_ms=10000.0,
        )

    mock_service.retrieve.side_effect = _mock_retrieve

    claims = [
        "Breaking RSA-2048 requires 20 million physical qubits",
        "IBM Eagle achieved quantum utility in 2023",
    ]

    status, all_items, claim_ev_map, claim_st_map = await retrieve_for_claims(
        claims=claims,
        retrieval_service=mock_service,
        concurrency_limit=2,
    )

    # Aggregate status is SUCCESS because at least one claim succeeded
    assert status == RetrievalStatus.SUCCESS
    assert len(all_items) == 1
    assert all_items[0].url == "https://example.com/rsa-threat"

    # Per-claim status reflects the isolated failure
    assert claim_st_map[claims[0]] == RetrievalStatus.SUCCESS
    assert claim_st_map[claims[1]] == RetrievalStatus.TIMEOUT


# ─── 3. Zero Retrieved Sources and NOT_CHECKED Claims ─────────────────────────

@pytest.mark.asyncio
async def test_zero_sources_from_timeouts_marks_all_claims_not_checked():
    """When all claim searches time out, report status is UNAVAILABLE and claims are NOT_CHECKED."""
    verifier = ClaimVerificationService()

    claims = ["Claim 1: Metric A is 50%", "Claim 2: Metric B is 100%"]
    claim_status_map = {
        claims[0]: RetrievalStatus.TIMEOUT,
        claims[1]: RetrievalStatus.TIMEOUT,
    }

    report = await verifier.verify_claims(
        claims=claims,
        evidence=[],
        retrieval_status=RetrievalStatus.TIMEOUT,
        claim_evidence_map={c: [] for c in claims},
        claim_status_map=claim_status_map,
    )

    assert report.report_status == VerificationReportStatus.UNAVAILABLE
    assert report.evidence_count == 0
    assert report.claims_not_checked == 2
    assert all(r.status == ClaimStatus.NOT_CHECKED for r in report.results)
    assert all("timeout" in r.explanation.lower() for r in report.results)


# ─── 4. Safe Fallback Behavior when Judge Fails ──────────────────────────────

@pytest.mark.asyncio
async def test_judge_failure_produces_safe_honest_fallback_with_zero_sources():
    """When Judge fails and zero sources were retrieved, fallback contains explicit warning and disclosures."""
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.side_effect = LLMError("Groq 400: json_validate_failed: max completion tokens reached")

    orchestrator = DebateOrchestrator(judge=mock_judge)

    # Force retrieval to return 0 sources with TIMEOUT
    async def _mock_retrieve(query: str, max_results: int = 2):
        return RetrievalResponse(
            query=query,
            status=RetrievalStatus.TIMEOUT,
            items=[],
            total_results=0,
            error_message="Timeout",
            latency_ms=10.0,
        )

    orchestrator._retrieval_service.retrieve = _mock_retrieve

    resp = await orchestrator.run_debate("Does quantum computing break RSA-2048?", max_rounds=1)

    assert resp.status in (DebateStatus.ERROR, DebateStatus.MAX_ROUNDS)
    assert resp.final_answer is not None
    assert "[UNADJUDICATED FALLBACK CANDIDATE - NOT VERIFIED]" in resp.final_answer
    assert "ZERO external evidence sources were verified" in resp.final_answer
    assert "UNCORROBORATED MODEL GENERATION" in resp.final_answer
    assert "Judge adjudication failed" in resp.verdict_summary

    # Never claims successful adjudication
    assert "fully grounded" not in resp.verdict_summary.lower()
    assert "fully verified" not in resp.verdict_summary.lower()

    # Trace includes Judge failure entry
    judge_traces = [t for t in resp.trace if t.agent == AgentRole.JUDGE]
    assert len(judge_traces) == 1
    assert "Judge failed" in judge_traces[0].output


# ─── 5. Normal Successful End-to-End Debate ──────────────────────────────────

@pytest.mark.asyncio
async def test_normal_successful_end_to_end_debate():
    """A standard debate with working providers produces accurate rounds, trace entries, and Judge synthesis."""
    orchestrator = DebateOrchestrator()

    resp = await orchestrator.run_debate("What are the advantages of asynchronous I/O?", max_rounds=1)

    assert resp.status in (DebateStatus.COMPLETED, DebateStatus.CONVERGED, DebateStatus.MAX_ROUNDS)
    assert resp.rounds_completed >= 1
    assert resp.final_answer is not None
    assert "[UNADJUDICATED FALLBACK CANDIDATE" not in resp.final_answer

    # Verify agent roles in trace
    agents_in_trace = [t.agent for t in resp.trace]
    assert AgentRole.PLANNER in agents_in_trace
    assert AgentRole.EXECUTOR_A in agents_in_trace
    assert AgentRole.EXECUTOR_B in agents_in_trace
    assert AgentRole.CRITIC in agents_in_trace
    assert AgentRole.JUDGE in agents_in_trace

    # Trace entries have valid non-empty output
    for entry in resp.trace:
        assert entry.output.strip()
        assert entry.round_number >= 0

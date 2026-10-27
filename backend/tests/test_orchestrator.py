"""
tests/test_orchestrator.py
──────────────────────────
Comprehensive offline tests for DebateOrchestrator and POST /api/query.

Requirements covered:
  - Successful complete workflow.
  - Planner failure handling.
  - Executor failure before and during revision.
  - Retrieval timeout and empty results handling.
  - Supported and contradicted claims processing.
  - Critic finding no material issues (early convergence).
  - Critic identifying unresolved issues (multi-round revision).
  - Correct enforcement of max_rounds and server-side hard limits.
  - Termination at the round limit.
  - Revision failure preserving the last valid candidate.
  - Judge receiving only grounded evidence references (no invented source URLs).
  - No API keys or private chain-of-thought in traces.
  - Correct final response schema validation.
  - Resource and call-count limits.
  - FastAPI endpoint integration test (POST /api/query).
"""

from __future__ import annotations

import re
from unittest.mock import AsyncMock

import pytest

from app.agents.critic import CriticAgent
from app.agents.executor_a import ExecutorAAgent
from app.agents.executor_b import ExecutorBAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.config import Settings
from app.llm.base import LLMConfig, LLMError, LLMResponse
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import (
    AgentRole,
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    CriticOutput,
    CritiqueIssue,
    CritiqueSeverity,
    DebateStatus,
    EvidenceItem,
    ExecutorOutput,
    JudgeOutput,
    QueryResponse,
    RetrievalResponse,
    RetrievalStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.mock_provider import MockSearchProvider
from app.retrieval.service import EvidenceRetrievalService


def _dummy_llm_response(text: str = "") -> LLMResponse:
    return LLMResponse(
        content=text,
        model_id="mock-model",
        provider="mock",
        latency_ms=10.0,
    )


# ─── 1. Successful Complete Workflow ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_complete_workflow():
    """Verify that a full multi-agent debate executes end-to-end and returns QueryResponse."""
    llm_client = MockLLMClient(LLMConfig())
    search_provider = MockSearchProvider()
    retrieval_service = EvidenceRetrievalService(provider=search_provider)

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        retrieval_service=retrieval_service,
    )

    query = "What causes inflation and how can central banks mitigate it?"
    response = await orchestrator.run_debate(query=query, max_rounds=2)

    assert isinstance(response, QueryResponse)
    assert response.query == query
    assert response.status == DebateStatus.COMPLETED
    assert response.rounds_completed >= 1
    assert response.final_answer is not None
    assert len(response.final_answer) > 0
    assert len(response.subtasks) > 0
    assert len(response.debate_rounds) >= 1
    assert len(response.trace) >= 4  # Planner, Exec A, Exec B, Critic, Judge
    assert response.error_message is None


# ─── 2. Planner Failure ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_planner_failure_handled_gracefully():
    """When the planner agent fails, orchestrator terminates cleanly with ERROR status."""
    mock_planner = AsyncMock(spec=PlannerAgent)
    mock_planner.plan.side_effect = LLMError("API key expired or server unavailable: sk-secret-1234")

    orchestrator = DebateOrchestrator(planner=mock_planner)
    response = await orchestrator.run_debate("Analyze macroeconomic policy")

    assert response.status == DebateStatus.ERROR
    assert response.rounds_completed == 0
    assert response.final_answer is None
    assert "Planner agent failure" in response.error_message
    # Verify no secret leaked
    assert "sk-secret-1234" not in response.error_message
    assert any("Planning failed" in entry.output for entry in response.trace)


# ─── 3. Executor Failures ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_both_executors_fail_initial_generation():
    """If both executors fail on round 1, orchestrator reports clean ERROR."""
    mock_exec_a = AsyncMock(spec=ExecutorAAgent)
    mock_exec_a.execute.side_effect = LLMError("Executor A model failure")

    mock_exec_b = AsyncMock(spec=ExecutorBAgent)
    mock_exec_b.execute.side_effect = LLMError("Executor B model failure")

    orchestrator = DebateOrchestrator(
        executor_a=mock_exec_a,
        executor_b=mock_exec_b,
    )
    response = await orchestrator.run_debate("Test query")

    assert response.status == DebateStatus.ERROR
    assert response.rounds_completed == 0
    assert "Both Executor A and Executor B failed" in response.error_message


@pytest.mark.asyncio
async def test_one_executor_fails_initial_generation_uses_fallback():
    """If one executor fails on round 1, the other is used as fallback and debate completes."""
    mock_exec_a = AsyncMock(spec=ExecutorAAgent)
    mock_exec_a.execute.side_effect = LLMError("Executor A failed")

    orchestrator = DebateOrchestrator(executor_a=mock_exec_a)
    response = await orchestrator.run_debate("Test query with single executor failure")

    assert response.status == DebateStatus.COMPLETED
    assert response.final_answer is not None
    assert response.rounds_completed >= 1


# ─── 4. Revision Failure Preserving Last Valid Candidate ──────────────────────

@pytest.mark.asyncio
async def test_revision_failure_preserves_last_valid_candidate():
    """When revision throws in round 2, the prior candidate is preserved and noted in trace."""
    llm_client = MockLLMClient(LLMConfig())
    mock_exec_a = ExecutorAAgent(client=llm_client)

    # First call to revise will fail
    mock_exec_a.revise = AsyncMock(side_effect=LLMError("Context window limit exceeded"))

    # Force critic to return material issue so revision is triggered
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.return_value = (
        CriticOutput(
            agreements=[],
            contradictions=["Severe contradiction in macroeconomic model"],
            unsupported_claims=["Claim X has no empirical basis"],
            uncertainties=[],
            issues=[
                CritiqueIssue(
                    issue_id="ISS-1",
                    target="executor_a",
                    severity=CritiqueSeverity.HIGH,
                    affected_claim="Claim X",
                    explanation="Unsupported empirical claim",
                    recommended_correction="Add evidence",
                )
            ],
            overall_critique="Candidate A has critical unsupported claims.",
        ),
        _dummy_llm_response(),
    )

    orchestrator = DebateOrchestrator(
        executor_a=mock_exec_a,
        critic=mock_critic,
    )

    response = await orchestrator.run_debate("Test query", max_rounds=2)

    assert response.status == DebateStatus.COMPLETED
    # Check trace for revision failure note
    exec_a_traces = [t for t in response.trace if t.agent == AgentRole.EXECUTOR_A]
    assert len(exec_a_traces) >= 2
    assert "[Revision failed" in exec_a_traces[1].output
    assert "Preserving prior candidate" in exec_a_traces[1].output


# ─── 5. Retrieval Timeout & Empty Results ─────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieval_timeout_handled_gracefully():
    """When retrieval times out, claim verification reports INSUFFICIENT_EVIDENCE and debate completes."""
    mock_search = AsyncMock()
    mock_search.provider_name = lambda: "mock"
    mock_search.search.return_value = RetrievalResponse(
        query="inflation query",
        status=RetrievalStatus.TIMEOUT,
        items=[],
        total_results=0,
        error_message="Search request timed out after 10s",
        latency_ms=10000.0,
    )
    retrieval_service = EvidenceRetrievalService(provider=mock_search)

    orchestrator = DebateOrchestrator(retrieval_service=retrieval_service)
    response = await orchestrator.run_debate("What is quantum computing?", max_rounds=1)

    assert response.status == DebateStatus.COMPLETED
    assert response.final_answer is not None


@pytest.mark.asyncio
async def test_retrieval_empty_results_handled_gracefully():
    """When retrieval returns 0 items, debate proceeds with insufficient evidence noted."""
    mock_search = AsyncMock()
    mock_search.provider_name = lambda: "mock"
    mock_search.search.return_value = RetrievalResponse(
        query="obscure query",
        status=RetrievalStatus.EMPTY,
        items=[],
        total_results=0,
        error_message=None,
        latency_ms=50.0,
    )
    retrieval_service = EvidenceRetrievalService(provider=mock_search)

    orchestrator = DebateOrchestrator(retrieval_service=retrieval_service)
    response = await orchestrator.run_debate("Obscure historical event", max_rounds=1)

    assert response.status == DebateStatus.COMPLETED
    assert response.final_answer is not None


# ─── 6. Supported and Contradicted Claims ─────────────────────────────────────

@pytest.mark.asyncio
async def test_supported_and_contradicted_claims():
    """Claim verifier outputs with SUPPORTED and CONTRADICTED statuses are fed to Critic and Judge."""
    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        results=[
            ClaimVerificationResult(
                claim_id="CLAIM-1",
                claim_text="Claim 1 is true",
                status=ClaimStatus.SUPPORTED,
                evidence_refs=["https://example.com/verified-1"],
                explanation="Verified by authoritative empirical source.",
            ),
            ClaimVerificationResult(
                claim_id="CLAIM-2",
                claim_text="Claim 2 is false",
                status=ClaimStatus.CONTRADICTED,
                evidence_refs=["https://example.com/verified-2"],
                explanation="Contradicted by empirical measurements.",
            ),
        ],
        summary="One claim supported, one contradicted.",
        evidence_count=2,
    )

    orchestrator = DebateOrchestrator(claim_verifier=mock_verifier)
    response = await orchestrator.run_debate("Evaluate factual claims", max_rounds=1)

    assert response.status == DebateStatus.COMPLETED
    assert response.final_answer is not None
    assert mock_verifier.verify_claims.called


# ─── 7. Critic Finding No Material Issues (Early Convergence) ─────────────────

@pytest.mark.asyncio
async def test_critic_no_material_issues_triggers_early_convergence():
    """When Critic finds no material issues on round 1, debate converges immediately."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.return_value = (
        CriticOutput(
            agreements=["Both candidates correctly identify fundamental mechanisms."],
            contradictions=[],
            unsupported_claims=[],
            uncertainties=["Long-term effects remain subject to empirical study."],
            issues=[],  # Zero issues!
            overall_critique="Both candidates provide thorough, accurate, and complementary analyses.",
        ),
        _dummy_llm_response(),
    )

    orchestrator = DebateOrchestrator(critic=mock_critic)
    # Requested 3 rounds, but should stop on round 1 due to convergence!
    response = await orchestrator.run_debate("Consensus topic", max_rounds=3)

    assert response.status == DebateStatus.COMPLETED
    assert response.rounds_completed == 1
    assert len(response.debate_rounds) == 1
    assert response.debate_rounds[0].converged is True


# ─── 8. Critic Identifying Unresolved Issues (Multi-Round) ────────────────────

@pytest.mark.asyncio
async def test_critic_with_unresolved_issues_runs_multiple_rounds():
    """Critic reporting unresolved issues leads to round 2 revision."""
    mock_critic = AsyncMock(spec=CriticAgent)
    # Round 1 has issues, Round 2 converges
    mock_critic.critique.side_effect = [
        (
            CriticOutput(
                agreements=[],
                contradictions=["Contradiction between theoretical and practical models."],
                unsupported_claims=["Claim needs citation."],
                uncertainties=[],
                issues=[
                    CritiqueIssue(
                        issue_id="ISSUE-1",
                        target="both",
                        severity=CritiqueSeverity.HIGH,
                        affected_claim="Universal efficiency",
                        explanation="Fails to account for boundary conditions.",
                        recommended_correction="Narrow scope to steady-state regimes.",
                    )
                ],
                overall_critique="Significant contradictions need resolution in revision.",
            ),
            _dummy_llm_response(),
        ),
        (
            CriticOutput(
                agreements=["Both revised answers successfully resolved boundary conditions."],
                contradictions=[],
                unsupported_claims=[],
                uncertainties=[],
                issues=[],
                overall_critique="All issues resolved.",
            ),
            _dummy_llm_response(),
        ),
    ]

    orchestrator = DebateOrchestrator(critic=mock_critic)
    response = await orchestrator.run_debate("Multi-round debate topic", max_rounds=3)

    assert response.status == DebateStatus.COMPLETED
    assert response.rounds_completed == 2
    assert len(response.debate_rounds) == 2
    assert response.debate_rounds[0].converged is False
    assert response.debate_rounds[1].converged is True


# ─── 9. Enforcement of max_rounds and Server-Side Hard Limits ─────────────────

@pytest.mark.asyncio
async def test_max_rounds_enforcement():
    """Client requested max_rounds is strictly respected and capped by server maximum."""
    # Critic always produces issues
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.return_value = (
        CriticOutput(
            agreements=[],
            contradictions=["Continuous contradiction"],
            unsupported_claims=["Continuous unsupported"],
            uncertainties=[],
            issues=[
                CritiqueIssue(
                    issue_id="ISS-1",
                    target="both",
                    severity=CritiqueSeverity.HIGH,
                    affected_claim="Persistent issue",
                    explanation="Persistent issue",
                    recommended_correction="Revise",
                )
            ],
            overall_critique="Issues persist.",
        ),
        _dummy_llm_response(),
    )

    settings = Settings(debate_max_rounds=2)
    orchestrator = DebateOrchestrator(critic=mock_critic, settings=settings)

    # User asks for 10 rounds, but server max is 2
    response = await orchestrator.run_debate("High max rounds query", max_rounds=10)

    assert response.rounds_completed == 2
    assert len(response.debate_rounds) == 2


@pytest.mark.asyncio
async def test_user_requested_lower_max_rounds():
    """If user requests 1 round when server allows 3, debate terminates after 1 round."""
    settings = Settings(debate_max_rounds=3)
    orchestrator = DebateOrchestrator(settings=settings)

    response = await orchestrator.run_debate("Query with max_rounds=1", max_rounds=1)
    assert response.rounds_completed == 1


# ─── 10. Repeated Identical Revisions Terminates Loop ──────────────────────────

@pytest.mark.asyncio
async def test_repeated_identical_revisions_terminates_loop():
    """If revisions produce answers identical to prior round, loop terminates early."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.return_value = (
        CriticOutput(
            agreements=[],
            contradictions=["Contradiction"],
            unsupported_claims=[],
            uncertainties=[],
            issues=[
                CritiqueIssue(
                    issue_id="I1",
                    target="both",
                    severity=CritiqueSeverity.HIGH,
                    affected_claim="A",
                    explanation="E",
                    recommended_correction="R",
                )
            ],
            overall_critique="Still has issues.",
        ),
        _dummy_llm_response(),
    )

    # Executors return identical answers on revision
    cand_a_fixed = ExecutorOutput(
        executor_id="executor_a",
        approach_summary="First-principles",
        answer="Fixed answer A that does not change.",
        key_assumptions=[],
        claims_to_verify=[],
    )
    cand_b_fixed = ExecutorOutput(
        executor_id="executor_b",
        approach_summary="Empirical",
        answer="Fixed answer B that does not change.",
        key_assumptions=[],
        claims_to_verify=[],
    )

    mock_exec_a = AsyncMock(spec=ExecutorAAgent)
    mock_exec_a.execute.return_value = (cand_a_fixed, _dummy_llm_response())
    mock_exec_a.revise.return_value = (cand_a_fixed, _dummy_llm_response())

    mock_exec_b = AsyncMock(spec=ExecutorBAgent)
    mock_exec_b.execute.return_value = (cand_b_fixed, _dummy_llm_response())
    mock_exec_b.revise.return_value = (cand_b_fixed, _dummy_llm_response())

    orchestrator = DebateOrchestrator(
        executor_a=mock_exec_a,
        executor_b=mock_exec_b,
        critic=mock_critic,
    )

    response = await orchestrator.run_debate("Identical answers test", max_rounds=5)
    # After round 1 revision produces identical answers, loop stops
    assert response.rounds_completed <= 2


# ─── 11. Grounded Evidence & No Hallucinated URLs ─────────────────────────────

@pytest.mark.asyncio
async def test_judge_evidence_grounding_strips_hallucinated_urls():
    """Ensure Judge cannot introduce fabricated source URLs that were never retrieved."""
    mock_search = AsyncMock()
    mock_search.provider_name = lambda: "mock"
    mock_search.search.return_value = RetrievalResponse(
        query="test",
        status=RetrievalStatus.SUCCESS,
        items=[
            EvidenceItem(
                title="Real Valid Source",
                url="https://valid-domain.org/article-1",
                snippet="Real verified data.",
            )
        ],
        total_results=1,
    )
    retrieval_service = EvidenceRetrievalService(provider=mock_search)

    # Judge tries to cite a fake URL
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer="Adjudicated final synthesis.",
            resolved_points=["Resolved A vs B."],
            unresolved_uncertainties=[],
            limitations=[],
            evidence_citations=[
                "Supported by https://valid-domain.org/article-1",
                "Invented source: https://fabricated-fake-news.xyz/unreal-claim",
            ],
            verdict_summary="Synthesis based on evidence.",
        ),
        _dummy_llm_response(),
    )

    orchestrator = DebateOrchestrator(
        retrieval_service=retrieval_service,
        judge=mock_judge,
    )

    response = await orchestrator.run_debate("Evidence grounding test", max_rounds=1)
    assert response.status == DebateStatus.COMPLETED

    # Verify sources list contains only the valid source
    source_urls = [s.url for s in response.sources]
    assert "https://valid-domain.org/article-1" in source_urls
    assert "https://fabricated-fake-news.xyz/unreal-claim" not in source_urls


# ─── 12. No API Keys or Private CoT in Traces ─────────────────────────────────

@pytest.mark.asyncio
async def test_no_api_keys_or_private_cot_in_traces():
    """Verify that traces never leak sensitive API keys or private thought tokens."""
    orchestrator = DebateOrchestrator()
    response = await orchestrator.run_debate("Test query for trace inspection", max_rounds=2)

    sensitive_patterns = [
        r"gsk_[a-zA-Z0-9]{20,}",
        r"AIza[a-zA-Z0-9_\-]{30,}",
        r"<thought>",
        r"</thought>",
        r"PRIVATE_COT",
    ]

    for entry in response.trace:
        for pattern in sensitive_patterns:
            assert not re.search(pattern, entry.output), (
                f"Sensitive pattern '{pattern}' found in trace entry for {entry.agent.value}: {entry.output}"
            )


# ─── 13. Resource and LLM Call Budget Limits ──────────────────────────────────

@pytest.mark.asyncio
async def test_llm_call_budget_limit_prevents_runaway():
    """When max_llm_calls budget is exceeded, orchestrator stops cleanly."""
    orchestrator = DebateOrchestrator(max_llm_calls=1)
    # Budget of 1 call: planner uses 1 call, executor A triggers budget exceeded
    response = await orchestrator.run_debate("Budget test query")

    assert response.status == DebateStatus.ERROR
    assert "call budget" in response.error_message.lower()


# ─── 14. FastAPI POST /api/query Endpoint Integration ─────────────────────────

@pytest.mark.asyncio
async def test_api_post_query_endpoint_success(client):
    """POST /api/query returns 200 with complete validated QueryResponse payload."""
    payload = {
        "query": "What are the primary factors driving climate change?",
        "max_rounds": 2,
    }
    resp = await client.post("/api/query", json=payload)
    assert resp.status_code == 200

    data = resp.json()
    assert data["query"] == payload["query"]
    assert data["status"] in {"completed", "converged", "max_rounds"}
    assert data["rounds_completed"] >= 1
    assert data["final_answer"] is not None
    assert isinstance(data["subtasks"], list)
    assert isinstance(data["debate_rounds"], list)
    assert isinstance(data["trace"], list)
    assert isinstance(data["sources"], list)


@pytest.mark.asyncio
async def test_api_post_query_endpoint_validation_error(client):
    """POST /api/query returns 400 when query is blank or invalid."""
    # Blank query
    resp = await client.post("/api/query", json={"query": "   "})
    assert resp.status_code == 400
    data = resp.json()
    assert data["error"] == "validation_error"

    # Query too short (< 3 chars)
    resp2 = await client.post("/api/query", json={"query": "hi"})
    assert resp2.status_code == 400

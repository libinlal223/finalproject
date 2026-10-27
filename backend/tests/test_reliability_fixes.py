"""
tests/test_reliability_fixes.py
────────────────────────────────
Regression tests covering minimal reliability fixes for confirmed failures:
  1. Critic output truncation and malformed JSON parsing failures (no unsafe repair).
  2. Critic failure with zero completed rounds reporting ERROR status, not MAX_ROUNDS.
  3. Accurate error/degraded status and unadjudicated fallback labeling.
  4. Judge bypass after Critic failure (Judge skipped, no false adjudication implied).
  5. Inconclusive claims remaining visible in fallback output without being marked false.
  6. Unsupported quantitative claims guarded from being presented as verified.
  7. Normal Judge output remaining compatible with the claim verification report.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app.agents.critic import CriticAgent
from app.agents.judge import JudgeAgent
from app.llm.base import (
    LLMConfig,
    LLMError,
    LLMResponse,
    LLMValidationError,
)
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import (
    AgentRole,
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    CriticOutput,
    DebateStatus,
    JudgeOutput,
    QueryResponse,
    VerificationReportStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService

# ─── 1. Critic JSON Truncation and Malformed JSON ─────────────────────────────

def test_critic_truncation_finish_reason_length_raises_validation_error():
    """Critic output cut off mid-stream with finish_reason='length' must raise LLMValidationError."""
    critic = CriticAgent(client=MockLLMClient(LLMConfig()))
    truncated_response = LLMResponse(
        content='{"agreements": ["Both agree on concept"], "contradictions": ["E',
        model_id="gemini-3.5-flash-lite",
        provider="gemini",
        finish_reason="length",
        completion_tokens=1024,
    )

    with pytest.raises(LLMValidationError) as exc_info:
        critic._parse_json_response(truncated_response, CriticOutput)

    err = str(exc_info.value)
    assert "truncated" in err.lower()
    assert "length" in err.lower()


def test_critic_malformed_json_raises_without_unsafe_repair():
    """Critic output with invalid JSON must raise LLMValidationError rather than attempting unsafe repair."""
    critic = CriticAgent(client=MockLLMClient(LLMConfig()))
    malformed_response = LLMResponse(
        content='{"agreements": ["Partial agreement"], "issues": [{"severity": "high", "affected_claim": ',
        model_id="gemini-3.5-flash-lite",
        provider="gemini",
        finish_reason="stop",
    )

    with pytest.raises(LLMValidationError) as exc_info:
        critic._parse_json_response(malformed_response, CriticOutput)

    err = str(exc_info.value)
    assert "not valid json" in err.lower()


def test_critic_default_config_has_expanded_token_budget_and_json_mode():
    """CriticAgent default configuration must allocate >= 2048 tokens and enable json_mode."""
    cfg = CriticAgent.default_config()
    assert cfg.max_tokens >= 2048
    assert cfg.json_mode is True


# ─── 2. Critic Failure with Zero Completed Rounds ────────────────────────────

@pytest.mark.asyncio
async def test_critic_failure_zero_completed_rounds_reports_error_not_max_rounds():
    """Debate aborting on Critic failure before any round completes must report ERROR, not MAX_ROUNDS."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = LLMValidationError(
        "Agent 'critic' output was truncated due to token limit budget (finish_reason='length')."
    )

    orchestrator = DebateOrchestrator(critic=mock_critic)
    resp = await orchestrator.run_debate("Analyze macroeconomic supply-side shocks", max_rounds=2)

    assert isinstance(resp, QueryResponse)
    assert resp.rounds_completed == 0
    assert resp.status == DebateStatus.ERROR
    assert resp.status != DebateStatus.MAX_ROUNDS
    assert resp.error_message is not None
    assert "Critic agent failed" in resp.error_message


# ─── 3. Accurate Fallback Labeling & Status ──────────────────────────────────

@pytest.mark.asyncio
async def test_critic_failure_clearly_labels_fallback_as_unadjudicated():
    """Fallback output must be clearly labeled as unadjudicated and never presented as verified synthesis."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = LLMError("Critic provider connection dropped.")

    orchestrator = DebateOrchestrator(critic=mock_critic)
    resp = await orchestrator.run_debate("Explain neutrino oscillation mechanisms", max_rounds=1)

    assert resp.final_answer is not None
    assert resp.final_answer.startswith("[UNADJUDICATED FALLBACK CANDIDATE - NOT VERIFIED]")
    assert "has NOT been adjudicated, verified, or synthesized by the Judge" in resp.final_answer
    assert "Critique step failed" in (resp.verdict_summary or "")
    assert "without Judge review" in (resp.verdict_summary or "")


# ─── 4. Judge Bypass After Critic Failure ────────────────────────────────────

@pytest.mark.asyncio
async def test_judge_bypassed_after_critic_failure():
    """If Critic fails in round 1, Judge must never be invoked and trace must not imply Judge adjudicated."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = LLMValidationError("Critic malformed JSON output.")

    mock_judge = AsyncMock(spec=JudgeAgent)

    orchestrator = DebateOrchestrator(critic=mock_critic, judge=mock_judge)
    resp = await orchestrator.run_debate("What is quantum entanglement?", max_rounds=2)

    # Judge must be completely bypassed
    mock_judge.judge.assert_not_called()

    # Trace must have no JUDGE entries implying adjudication completed
    judge_traces = [t for t in resp.trace if t.agent == AgentRole.JUDGE]
    assert len(judge_traces) == 0


# ─── 5. Inconclusive Claims Prominently Disclosed in Fallback ────────────────

@pytest.mark.asyncio
async def test_inconclusive_claims_prominently_disclosed_in_fallback_output():
    """Inconclusive claims must be prominently disclosed in fallback output and uncertainties without being marked false."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = RuntimeError("Critic generation failed.")

    # Create mock claim verifier that returns inconclusive claims
    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    inconclusive_claim = ClaimVerificationResult(
        claim_id="CLAIM-1",
        claim_text="Superconductor critical temperature reaches 150K at ambient pressure",
        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
        explanation="Search retrieved no laboratory evidence corroborating 150K ambient superconductivity.",
        evidence=[],
        limitations="Lack of empirical corroboration in retrieved literature.",
    )
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=[inconclusive_claim],
        summary="Evaluated 1 claim: 0 supported, 0 contradicted, 1 inconclusive, 0 not checked.",
        evidence_count=0,
        claims_checked=1,
        claims_not_checked=0,
    )

    orchestrator = DebateOrchestrator(critic=mock_critic, claim_verifier=mock_verifier)
    resp = await orchestrator.run_debate("High temperature superconductivity threshold", max_rounds=1)

    # Verification report must be preserved
    assert resp.verification_report is not None
    assert len(resp.verification_report.results) == 1
    assert resp.verification_report.results[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE

    # Inconclusive claim must be prominently disclosed in fallback answer text
    assert resp.final_answer is not None
    assert "CLAIM-1" in resp.final_answer
    assert "insufficient_evidence" in resp.final_answer.lower() or "unverified" in resp.final_answer.lower()

    # Inconclusive claim must also appear in unresolved_uncertainties
    assert any("CLAIM-1" in unc for unc in resp.unresolved_uncertainties)
    assert any("Superconductor critical temperature" in unc for unc in resp.unresolved_uncertainties)

    # Claim must NOT be falsely marked as contradicted/false
    assert resp.verification_report.results[0].status != ClaimStatus.CONTRADICTED


# ─── 6. Unsupported Quantitative Claims Guard ─────────────────────────────────

@pytest.mark.asyncio
async def test_unsupported_quantitative_claims_guarded_from_verified_status():
    """Unsupported quantitative claims in Judge synthesis must be flagged in unresolved uncertainties."""
    # Setup mock Judge that generates an answer asserting an unverified 99.4% figure
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer="The new propulsion turbine demonstrates 99.4% mechanical efficiency in all regimes.",
            resolved_points=["Reconciled aerodynamic models."],
            unresolved_uncertainties=["General operating tolerances."],
            limitations=["Applies under nominal pressure."],
            evidence_citations=[],
            verdict_summary="Synthesis accepting turbine efficiency.",
            adjudication_rationale="Based on candidate modeling.",
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=5.0),
    )

    # Setup mock verifier that flags 99.4% as INSUFFICIENT_EVIDENCE
    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    unverified_quant_claim = ClaimVerificationResult(
        claim_id="CLAIM-QUANT-1",
        claim_text="Turbine demonstrates 99.4% mechanical efficiency",
        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
        explanation="No empirical test logs retrieved corroborating 99.4% efficiency figure.",
        evidence=[],
    )
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=[unverified_quant_claim],
        summary="Evaluated 1 claim: 1 inconclusive.",
        evidence_count=2,
        claims_checked=1,
        claims_not_checked=0,
    )

    orchestrator = DebateOrchestrator(judge=mock_judge, claim_verifier=mock_verifier)
    resp = await orchestrator.run_debate("Turbine efficiency benchmark", max_rounds=1)

    assert resp.status == DebateStatus.COMPLETED
    assert resp.final_answer is not None

    # The deterministic guard must have flagged the unverified quantitative claim in unresolved_uncertainties
    quant_caveats = [
        unc for unc in resp.unresolved_uncertainties
        if "CLAIM-QUANT-1" in unc or "99.4%" in unc
    ]
    assert len(quant_caveats) > 0
    assert "Unverified quantitative claim" in quant_caveats[0]
    assert "inconclusive" in quant_caveats[0].lower() or "unverified" in quant_caveats[0].lower()


# ─── 7. Normal Judge Output Remains Compatible with Verification Report ──────

@pytest.mark.asyncio
async def test_normal_judge_output_compatible_with_verification_report():
    """Successful debate produces COMPLETED status and a verification report compatible with final answer."""
    orchestrator = DebateOrchestrator()
    resp = await orchestrator.run_debate("What causes aurora borealis?", max_rounds=1)

    assert resp.status in (DebateStatus.COMPLETED, DebateStatus.CONVERGED)
    assert resp.rounds_completed >= 1
    assert resp.final_answer is not None
    assert not resp.final_answer.startswith("[UNADJUDICATED FALLBACK")
    assert resp.verification_report is not None
    assert isinstance(resp.verification_report, ClaimVerificationReport)
    assert resp.error_message is None

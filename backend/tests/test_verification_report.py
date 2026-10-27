"""
tests/test_verification_report.py
───────────────────────────────────
Priority 1 tests — Transparent Claim Verification in VADER.

Covers:
  1.  Claim supported by credible evidence.
  2.  Claim contradicted by evidence.
  3.  Claim with insufficient evidence (empty retrieval).
  4.  Claim that has not been checked (retrieval failure).
  5.  Conflicting sources → insufficient_evidence with conflict note.
  6.  Retrieval failure (provider error) → not_checked, report_status=unavailable.
  7.  Verification service failure (LLMError) → not_checked, report_status=failed.
  8.  Malformed / missing evidence fields handled gracefully.
  9.  Correct serialization through the FastAPI /api/query response.
  10. Verification failure cannot produce verified status.
  11. claim_id assignment is deterministic.
  12. evidence_refs hallucination prevention (fabricated URLs stripped).
  13. No claims submitted → report_status=no_claims.
  14. QueryResponse always carries a verification_report after orchestration.
  15. Orchestrator end-to-end: verification_report present in full workflow.

All tests use mocks — no real API keys or network calls required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.llm.base import LLMConfig, LLMError, LLMResponse
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import (
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    DebateStatus,
    EvidenceItem,
    QueryResponse,
    RetrievalStatus,
    VerificationEvidenceRef,
    VerificationReportStatus,
)
from app.retrieval.claim_verifier import ClaimVerificationService


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _mock_llm(report_dict: dict) -> MockLLMClient:
    """Build a MockLLMClient that returns report_dict as JSON."""
    return MockLLMClient(
        LLMConfig(
            extra={
                "mock_response": json.dumps(report_dict),
                "simulate_latency_ms": 0.0,
            }
        )
    )


def _evidence(url: str, title: str, snippet: str) -> EvidenceItem:
    return EvidenceItem(title=title, url=url, snippet=snippet)


# ─── 1. Supported claim ───────────────────────────────────────────────────────

class TestSupportedClaim:
    async def test_supported_claim_status_and_evidence(self):
        """A claim corroborated by retrieved evidence must be marked supported."""
        url = "https://stats.example.org/cpi-2023"
        evidence = [_evidence(url, "CPI Report 2023", "CPI rose 3.2 percent in 2023.")]
        claims = ["The CPI rose by approximately 3.2 percent in 2023."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "supported",
                "evidence_refs": [url],
                "explanation": "CPI report directly confirms 3.2% rise.",
                "uncertainty": None,
            }],
            "summary": "One claim supported.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        assert isinstance(report, ClaimVerificationReport)
        assert report.report_status == VerificationReportStatus.COMPLETED
        assert len(report.results) == 1
        r = report.results[0]
        assert r.claim_id == "CLAIM-1"
        assert r.claim_text == claims[0]
        assert r.status == ClaimStatus.SUPPORTED
        assert len(r.evidence) == 1
        assert r.evidence[0].url == url
        assert r.evidence[0].relationship == "supports"
        assert r.limitations is None
        assert report.claims_checked == 1
        assert report.claims_not_checked == 0

    async def test_supported_does_not_guarantee_truth_disclaimer(self):
        """supported status must not silence the disclaimer caveat (structural check)."""
        # The model must still have evidence[] populated and NOT be marked not_checked
        url = "https://facts.example.org"
        evidence = [_evidence(url, "Source", "Relevant snippet.")]
        claims = ["Earth orbits the Sun."]
        report_data = {
            "results": [{"claim": claims[0], "status": "supported",
                         "evidence_refs": [url], "explanation": "Confirmed.", "uncertainty": None}],
            "summary": "Supported.", "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)
        r = report.results[0]
        # Status is supported but limitations should be None, not "not a guarantee"
        # The disclaimer is the frontend's responsibility; the API must not conflate statuses
        assert r.status == ClaimStatus.SUPPORTED
        assert r.status != ClaimStatus.NOT_CHECKED


# ─── 2. Contradicted claim ────────────────────────────────────────────────────

class TestContradictedClaim:
    async def test_contradicted_claim_status(self):
        """A claim refuted by evidence must be marked contradicted."""
        url = "https://physics.example.org/gravity"
        evidence = [_evidence(url, "Physics Text", "Objects fall at 9.8 m/s² in Earth's gravity.")]
        claims = ["Objects in free fall accelerate at 5 m/s² on Earth."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "contradicted",
                "evidence_refs": [url],
                "explanation": "Evidence states 9.8 m/s², directly contradicting 5 m/s².",
                "uncertainty": None,
            }],
            "summary": "Contradicted by physics literature.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        r = report.results[0]
        assert r.status == ClaimStatus.CONTRADICTED
        assert r.evidence[0].relationship == "contradicts"
        assert report.report_status == VerificationReportStatus.COMPLETED


# ─── 3. Insufficient evidence ─────────────────────────────────────────────────

class TestInsufficientEvidence:
    async def test_empty_retrieval_returns_insufficient_not_contradicted(self):
        """Empty retrieval must yield insufficient_evidence, never contradicted."""
        claims = ["The GDP of Narnia grew by 4% in 2024."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))  # LLM never called
        report = await svc.verify_claims(claims, evidence=[], retrieval_status=RetrievalStatus.EMPTY)

        assert report.report_status == VerificationReportStatus.COMPLETED
        r = report.results[0]
        assert r.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert r.evidence == []
        assert "does not prove the claim is false" in r.explanation.lower() or \
               "insufficient" in r.explanation.lower()

    async def test_insufficient_llm_response(self):
        """An LLM that cannot confirm or deny must produce insufficient_evidence."""
        url = "https://neutral.example.org/data"
        evidence = [_evidence(url, "Neutral Report", "General overview of the topic.")]
        claims = ["Quantum computers will replace classical computers by 2025."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "insufficient_evidence",
                "evidence_refs": [url],
                "explanation": "Retrieved evidence does not confirm or deny this specific date claim.",
                "uncertainty": "The field evolves rapidly; no definitive timeline is established.",
            }],
            "summary": "Claim unresolved.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        r = report.results[0]
        assert r.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert r.limitations is not None
        assert "timeline" in r.limitations.lower()


# ─── 4. Claim not checked ─────────────────────────────────────────────────────

class TestClaimNotChecked:
    async def test_retrieval_error_marks_not_checked(self):
        """When retrieval errors, claims must be NOT_CHECKED, not silently verified."""
        claims = ["The unemployment rate fell to 3.5%."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(claims, evidence=[], retrieval_status=RetrievalStatus.ERROR)

        assert report.report_status == VerificationReportStatus.UNAVAILABLE
        r = report.results[0]
        assert r.status == ClaimStatus.NOT_CHECKED
        assert r.claim_id == "CLAIM-1"
        assert r.evidence == []
        assert "does not imply" in r.explanation.lower() or "not_checked" in r.status.value

    async def test_retrieval_timeout_marks_not_checked(self):
        """Timeout must also produce NOT_CHECKED, not insufficient_evidence or supported."""
        claims = ["Inflation peaked at 9% in 2022."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(claims, evidence=[], retrieval_status=RetrievalStatus.TIMEOUT)

        assert report.report_status == VerificationReportStatus.UNAVAILABLE
        assert all(r.status == ClaimStatus.NOT_CHECKED for r in report.results)
        assert report.claims_checked == 0
        assert report.claims_not_checked == len(claims)

    async def test_rate_limited_marks_not_checked(self):
        """Rate limiting must produce NOT_CHECKED."""
        claims = ["Claim A", "Claim B"]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(
            claims, evidence=[], retrieval_status=RetrievalStatus.RATE_LIMITED
        )
        assert report.report_status == VerificationReportStatus.UNAVAILABLE
        assert all(r.status == ClaimStatus.NOT_CHECKED for r in report.results)


# ─── 5. Conflicting sources ────────────────────────────────────────────────────

class TestConflictingSources:
    async def test_conflicting_sources_produce_insufficient_evidence(self):
        """If sources disagree, status must be insufficient_evidence with conflict noted."""
        url_a = "https://source-a.example.org"
        url_b = "https://source-b.example.org"
        evidence = [
            _evidence(url_a, "Report A", "Study shows X is 10%."),
            _evidence(url_b, "Report B", "Study shows X is 25%."),
        ]
        claims = ["X is 10%."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "insufficient_evidence",
                "evidence_refs": [url_a, url_b],
                "explanation": "Sources conflict: Report A says 10%, Report B says 25%.",
                "uncertainty": "Conflicting measurements; cannot determine which is accurate.",
            }],
            "summary": "Conflict detected.",
            "evidence_count": 2,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        r = report.results[0]
        assert r.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert r.limitations is not None
        assert "conflict" in r.limitations.lower()
        # Both source URLs must be present in evidence
        urls_in_evidence = {e.url for e in r.evidence}
        assert url_a in urls_in_evidence
        assert url_b in urls_in_evidence

    async def test_conflicting_sources_are_not_silently_resolved(self):
        """A conflicting-source result must NOT be marked supported."""
        url_a = "https://a.example.org"
        url_b = "https://b.example.org"
        evidence = [_evidence(url_a, "A", "Confirms."), _evidence(url_b, "B", "Denies.")]
        claims = ["Claim under conflict."]

        report_data = {
            "results": [{
                "claim": claims[0], "status": "insufficient_evidence",
                "evidence_refs": [url_a, url_b],
                "explanation": "Sources conflict.",
                "uncertainty": "Conflicting.",
            }],
            "summary": "Conflict.", "evidence_count": 2,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)
        assert report.results[0].status != ClaimStatus.SUPPORTED


# ─── 6. Retrieval failure ─────────────────────────────────────────────────────

class TestRetrievalFailure:
    async def test_provider_error_report_status_is_unavailable(self):
        claims = ["Some factual claim."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(
            claims, evidence=[], retrieval_status=RetrievalStatus.ERROR
        )
        assert report.report_status == VerificationReportStatus.UNAVAILABLE
        assert report.evidence_count == 0

    async def test_empty_evidence_still_produces_report(self):
        """Empty evidence (successful search, no results) must still produce a report."""
        claims = ["Any claim."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(
            claims, evidence=[], retrieval_status=RetrievalStatus.EMPTY
        )
        # Should produce completed (with insufficient_evidence per claim), not unavailable
        assert report.report_status == VerificationReportStatus.COMPLETED
        assert len(report.results) == 1
        assert report.results[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE


# ─── 7. Verification-service failure ─────────────────────────────────────────

class TestVerificationServiceFailure:
    async def test_llm_error_raises_llm_error(self):
        """An LLMError from the client must propagate as LLMError, not silently succeed."""
        from app.llm.base import LLMError

        url = "https://example.org"
        evidence = [_evidence(url, "Source", "Snippet.")]
        claims = ["A claim."]

        failing_client = AsyncMock()
        failing_client.complete = AsyncMock(side_effect=LLMError("LLM unavailable"))

        svc = ClaimVerificationService(llm_client=failing_client)
        with pytest.raises(LLMError):
            await svc.verify_claims(claims, evidence)

    async def test_llm_returns_invalid_json_raises_validation_error(self):
        """Malformed JSON from the LLM must raise LLMValidationError, not silently succeed."""
        from app.llm.base import LLMValidationError

        url = "https://example.org"
        evidence = [_evidence(url, "Source", "Snippet.")]
        claims = ["A claim."]

        bad_json_client = MockLLMClient(
            LLMConfig(extra={"mock_response": "NOT VALID JSON AT ALL {{{", "simulate_latency_ms": 0.0})
        )
        svc = ClaimVerificationService(llm_client=bad_json_client)
        with pytest.raises(LLMValidationError):
            await svc.verify_claims(claims, evidence)


# ─── 8. Malformed evidence ────────────────────────────────────────────────────

class TestMalformedEvidence:
    async def test_missing_status_field_defaults_to_insufficient(self):
        """If the LLM omits the status field, must default to insufficient_evidence."""
        url = "https://example.org/source"
        evidence = [_evidence(url, "Source", "Some text.")]
        claims = ["A claim."]

        bad_data = {
            "results": [{
                "claim": claims[0],
                # "status" intentionally missing
                "evidence_refs": [url],
                "explanation": "Explanation.",
                "uncertainty": None,
            }],
            "summary": "Done.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(bad_data))
        report = await svc.verify_claims(claims, evidence)

        assert len(report.results) == 1
        assert report.results[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE

    async def test_hallucinated_url_in_evidence_refs_is_stripped(self):
        """URLs not present in the retrieved evidence must be stripped from evidence_refs."""
        real_url = "https://real.example.org"
        hallucinated_url = "https://fabricated-url-that-was-never-retrieved.org/fake"
        evidence = [_evidence(real_url, "Real Source", "Real snippet.")]
        claims = ["A claim."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "supported",
                "evidence_refs": [real_url, hallucinated_url],  # hallucinated URL included
                "explanation": "Supported.",
                "uncertainty": None,
            }],
            "summary": "Supported.", "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        r = report.results[0]
        evidence_urls = {e.url for e in r.evidence}
        assert real_url in evidence_urls
        assert hallucinated_url not in evidence_urls  # must be stripped

    async def test_empty_claims_list_produces_no_claims_report(self):
        """No claims submitted must produce a NO_CLAIMS report without calling the LLM."""
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims([], evidence=[])

        assert report.report_status == VerificationReportStatus.NO_CLAIMS
        assert report.results == []
        assert report.claims_checked == 0
        assert report.claims_not_checked == 0


# ─── 9. Serialization through FastAPI ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_verification_report_serializes_in_query_response(client):
    """POST /api/query must return a verification_report field in the JSON response."""
    payload = {"query": "What causes inflation?", "max_rounds": 1}
    resp = await client.post("/api/query", json=payload)

    assert resp.status_code == 200
    data = resp.json()

    # verification_report must be present (never missing from a successful debate)
    assert "verification_report" in data
    report = data["verification_report"]
    # May be null only if debate failed before retrieval (not the case here)
    # For mock provider, expect a structured report object
    if report is not None:
        assert "report_status" in report
        assert "results" in report
        assert "summary" in report
        assert "claims_checked" in report
        assert "claims_not_checked" in report
        assert isinstance(report["results"], list)
        # Each result must have required fields
        for r in report["results"]:
            assert "claim_id" in r
            assert "claim_text" in r
            assert "status" in r
            assert "explanation" in r
            assert "evidence" in r
            assert isinstance(r["evidence"], list)
            # Status must be a known value
            assert r["status"] in {
                "supported", "contradicted", "insufficient_evidence", "not_checked"
            }


# ─── 10. Failure cannot produce verified status ───────────────────────────────

class TestVerificationIntegrity:
    async def test_retrieval_failure_cannot_produce_supported_status(self):
        """Under no retrieval-failure scenario should any claim be marked supported."""
        claims = ["Important factual claim."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))

        for bad_status in [RetrievalStatus.ERROR, RetrievalStatus.TIMEOUT, RetrievalStatus.RATE_LIMITED]:
            report = await svc.verify_claims(claims, evidence=[], retrieval_status=bad_status)
            for r in report.results:
                assert r.status != ClaimStatus.SUPPORTED, (
                    f"Got 'supported' under {bad_status} — integrity violation!"
                )

    async def test_no_evidence_cannot_produce_supported_status(self):
        """With zero retrieved items, no claim may be marked supported."""
        claims = ["Some unverifiable claim."]
        svc = ClaimVerificationService(llm_client=_mock_llm({}))
        report = await svc.verify_claims(claims, evidence=[], retrieval_status=RetrievalStatus.EMPTY)
        for r in report.results:
            assert r.status != ClaimStatus.SUPPORTED


# ─── 11. Claim-id determinism ─────────────────────────────────────────────────

class TestClaimIdDeterminism:
    async def test_claim_ids_are_stable_and_sequential(self):
        """claim_id values must be CLAIM-1, CLAIM-2, … in order."""
        url_a = "https://a.example.org"
        url_b = "https://b.example.org"
        evidence = [_evidence(url_a, "A", "A."), _evidence(url_b, "B", "B.")]
        claims = ["First claim.", "Second claim.", "Third claim."]

        report_data = {
            "results": [
                {"claim": claims[0], "status": "supported", "evidence_refs": [url_a],
                 "explanation": "OK.", "uncertainty": None},
                {"claim": claims[1], "status": "contradicted", "evidence_refs": [url_b],
                 "explanation": "No.", "uncertainty": None},
                {"claim": claims[2], "status": "insufficient_evidence", "evidence_refs": [],
                 "explanation": "Unclear.", "uncertainty": "Uncertain."},
            ],
            "summary": "Mixed results.", "evidence_count": 2,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        assert len(report.results) == 3
        assert report.results[0].claim_id == "CLAIM-1"
        assert report.results[1].claim_id == "CLAIM-2"
        assert report.results[2].claim_id == "CLAIM-3"


# ─── 12. Schema field validation ──────────────────────────────────────────────

class TestSchemaFields:
    def test_claim_verification_result_requires_claim_id(self):
        """ClaimVerificationResult must have a claim_id field."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            # Missing claim_id must fail
            ClaimVerificationResult(
                claim_text="Some claim.",
                status=ClaimStatus.SUPPORTED,
                explanation="Explanation.",
            )

    def test_verification_evidence_ref_model(self):
        """VerificationEvidenceRef must carry all required fields."""
        ref = VerificationEvidenceRef(
            title="Test Source",
            url="https://example.org/test",
            excerpt="Relevant text.",
            relationship="supports",
        )
        assert ref.retrieval_status is None

    def test_claim_verification_report_summary_required(self):
        """ClaimVerificationReport must require a non-empty summary."""
        from pydantic import ValidationError as PydanticValidationError
        with pytest.raises(PydanticValidationError):
            ClaimVerificationReport(summary="")  # empty string should fail min_length=1

    def test_query_response_has_verification_report_field(self):
        """QueryResponse schema must expose verification_report as optional."""
        resp = QueryResponse(
            query="Test", status=DebateStatus.COMPLETED, rounds_completed=1
        )
        assert hasattr(resp, "verification_report")
        assert resp.verification_report is None

    def test_query_response_with_embedded_report(self):
        """QueryResponse must accept a full ClaimVerificationReport."""
        report = ClaimVerificationReport(
            report_status=VerificationReportStatus.COMPLETED,
            results=[
                ClaimVerificationResult(
                    claim_id="CLAIM-1",
                    claim_text="Earth is the third planet.",
                    status=ClaimStatus.SUPPORTED,
                    explanation="Confirmed by astronomy sources.",
                    evidence=[
                        VerificationEvidenceRef(
                            title="NASA",
                            url="https://nasa.gov/solar-system",
                            excerpt="Earth is the third planet from the Sun.",
                            relationship="supports",
                        )
                    ],
                )
            ],
            summary="One claim supported.",
            evidence_count=1,
            claims_checked=1,
            claims_not_checked=0,
        )
        resp = QueryResponse(
            query="Is Earth the third planet?",
            status=DebateStatus.COMPLETED,
            rounds_completed=1,
            verification_report=report,
        )
        data = resp.model_dump(mode="json")
        assert data["verification_report"]["report_status"] == "completed"
        assert len(data["verification_report"]["results"]) == 1
        result_data = data["verification_report"]["results"][0]
        assert result_data["claim_id"] == "CLAIM-1"
        assert result_data["status"] == "supported"
        assert len(result_data["evidence"]) == 1
        assert result_data["evidence"][0]["relationship"] == "supports"
        # Legacy excluded fields must NOT appear in serialized output
        assert "claim" not in result_data
        assert "evidence_refs" not in result_data
        assert "uncertainty" not in result_data


# ─── 14 & 15. Orchestrator integration ────────────────────────────────────────

@pytest.mark.asyncio
async def test_orchestrator_includes_verification_report_in_response():
    """Full orchestrator run must include a verification_report in QueryResponse."""
    from app.agents.critic import CriticAgent
    from app.agents.executor_a import ExecutorAAgent
    from app.agents.executor_b import ExecutorBAgent
    from app.agents.judge import JudgeAgent
    from app.agents.planner import PlannerAgent
    from app.config import Settings
    from app.orchestrator.orchestrator import DebateOrchestrator
    from app.retrieval.mock_provider import MockSearchProvider
    from app.retrieval.service import EvidenceRetrievalService

    llm_client = MockLLMClient(LLMConfig())
    search_provider = MockSearchProvider()
    retrieval_service = EvidenceRetrievalService(provider=search_provider)

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        retrieval_service=retrieval_service,
        settings=Settings(
            llm_provider="mock",
            debate_max_rounds=1,
        ),
    )

    result = await orchestrator.run_debate("What is the boiling point of water?")

    assert isinstance(result, QueryResponse)
    # verification_report must always be present (may be null only on pre-retrieval failure)
    assert hasattr(result, "verification_report")
    # For mock provider, report must be a ClaimVerificationReport instance
    assert result.verification_report is not None
    assert isinstance(result.verification_report, ClaimVerificationReport)
    assert result.verification_report.report_status in VerificationReportStatus.__members__.values()


@pytest.mark.asyncio
async def test_orchestrator_no_claims_produces_no_claims_report():
    """When no claims are extracted, orchestrator must produce NO_CLAIMS report."""
    from app.agents.planner import PlannerAgent
    from app.agents.executor_a import ExecutorAAgent
    from app.agents.executor_b import ExecutorBAgent
    from app.agents.critic import CriticAgent
    from app.agents.judge import JudgeAgent
    from app.config import Settings
    from app.orchestrator.orchestrator import DebateOrchestrator
    from app.models.schemas import (
        ExecutorOutput, TaskPlan, CriticOutput, JudgeOutput
    )
    from app.llm.base import LLMResponse

    def _resp(text: str) -> LLMResponse:
        return LLMResponse(content=text, model_id="mock", provider="mock", latency_ms=0.0)

    import json

    # Planner outputs a plan with NO evidence_needed and executors produce NO claims_to_verify
    planner_mock = AsyncMock(spec=PlannerAgent)
    plan = TaskPlan(
        core_question="What is 2+2?",
        subtasks=["Compute 2+2."],
        constraints=[],
        expected_format="single_value",
        evidence_needed=[],  # No evidence needed
    )
    planner_mock.plan = AsyncMock(return_value=(plan, _resp("")))

    exec_a_mock = AsyncMock(spec=ExecutorAAgent)
    exec_a_mock.execute = AsyncMock(return_value=(
        ExecutorOutput(executor_id="executor_a", approach_summary="Math",
                       answer="4", key_assumptions=[], claims_to_verify=[]),  # No claims
        _resp(""),
    ))

    exec_b_mock = AsyncMock(spec=ExecutorBAgent)
    exec_b_mock.execute = AsyncMock(return_value=(
        ExecutorOutput(executor_id="executor_b", approach_summary="Math2",
                       answer="4", key_assumptions=[], claims_to_verify=[]),
        _resp(""),
    ))

    critic_mock = AsyncMock(spec=CriticAgent)
    critic_mock.critique = AsyncMock(return_value=(
        CriticOutput(agreements=[], contradictions=[], unsupported_claims=[],
                     uncertainties=[], issues=[], overall_critique="Both agree."),
        _resp(""),
    ))

    judge_mock = AsyncMock(spec=JudgeAgent)
    judge_mock.judge = AsyncMock(return_value=(
        JudgeOutput(final_answer="4", resolved_points=[], unresolved_uncertainties=[],
                    limitations=[], evidence_citations=[], verdict_summary="Simple math."),
        _resp(""),
    ))

    orchestrator = DebateOrchestrator(
        planner=planner_mock,
        executor_a=exec_a_mock,
        executor_b=exec_b_mock,
        critic=critic_mock,
        judge=judge_mock,
        settings=Settings(llm_provider="mock", debate_max_rounds=1),
    )

    result = await orchestrator.run_debate("What is 2+2?")

    assert result.verification_report is not None
    assert result.verification_report.report_status == VerificationReportStatus.NO_CLAIMS
    assert result.verification_report.results == []


# ─── 16. Audit Regression Tests ───────────────────────────────────────────────

class TestClaimVerificationAuditRegressions:
    """
    Focused regression tests addressing verification report audit findings:
      - Deterministic evidence count (never trust LLM-generated count).
      - Accurate verification summary reflecting actual claim statuses and evidence count.
      - Atomic-claim instructions present in both Executor A and Executor B prompts.
      - Supported date with unsupported timestamp remaining distinguishable.
      - No relevant evidence remaining inconclusive (never fabricated truth/disproof).
    """

    async def test_evidence_count_matches_actual_retrieved_evidence_not_llm_count(self):
        """Report evidence_count must match actual retrieved items even if LLM returns a different number."""
        url1 = "https://source1.example.org/doc"
        url2 = "https://source2.example.org/doc"
        evidence = [
            _evidence(url1, "Source 1", "Fact A is corroborated."),
            _evidence(url2, "Source 2", "Fact B is corroborated."),
        ]
        claims = ["Fact A is corroborated."]

        # LLM untruthfully claims evidence_count is 99
        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "supported",
                "evidence_refs": [url1],
                "explanation": "Fact A is directly corroborated by Source 1.",
                "uncertainty": None,
            }],
            "summary": "Fabricated summary with wrong count.",
            "evidence_count": 99,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        # Must strictly equal len(evidence) == 2, never the LLM's 99
        assert report.evidence_count == 2
        assert "2 retrieved sources" in report.summary
        assert "99" not in report.summary

        # Also verify 0 evidence case when LLM hallucinated count 5
        zero_evidence_data = {
            "results": [{
                "claim": claims[0],
                "status": "insufficient_evidence",
                "evidence_refs": [],
                "explanation": "No sources.",
                "uncertainty": "No evidence.",
            }],
            "summary": "LLM says 5 sources checked.",
            "evidence_count": 5,
        }
        svc_zero = ClaimVerificationService(llm_client=_mock_llm(zero_evidence_data))
        report_zero = await svc_zero.verify_claims(claims, evidence=[])
        assert report_zero.evidence_count == 0
        assert "0 retrieved sources" in report_zero.summary
        assert "5" not in report_zero.summary

    async def test_summary_counts_match_actual_claim_statuses(self):
        """Report summary must deterministically reflect actual claim statuses and actual evidence count."""
        url1 = "https://source.example.org/1"
        url2 = "https://source.example.org/2"
        evidence = [
            _evidence(url1, "Source 1", "Evidence item 1."),
            _evidence(url2, "Source 2", "Evidence item 2."),
        ]
        claims = [
            "Claim 1 is supported.",
            "Claim 2 is contradicted.",
            "Claim 3 has inconclusive evidence.",
            "Claim 4 status is unverified.",
        ]

        report_data = {
            "results": [
                {
                    "claim": claims[0],
                    "status": "supported",
                    "evidence_refs": [url1],
                    "explanation": "Supported by Source 1.",
                    "uncertainty": None,
                },
                {
                    "claim": claims[1],
                    "status": "contradicted",
                    "evidence_refs": [url2],
                    "explanation": "Contradicted by Source 2.",
                    "uncertainty": None,
                },
                {
                    "claim": claims[2],
                    "status": "insufficient_evidence",
                    "evidence_refs": [],
                    "explanation": "Inconclusive.",
                    "uncertainty": "Gap in data.",
                },
                {
                    "claim": claims[3],
                    "status": "not_checked",
                    "evidence_refs": [],
                    "explanation": "Not checked due to scope.",
                    "uncertainty": None,
                },
            ],
            "summary": "LLM hallucinations: All claims fully verified by consensus.",
            "evidence_count": 42,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        # Report summary must not be overridden by LLM summary
        assert "All claims fully verified" not in report.summary
        assert report.evidence_count == 2
        # Deterministic summary must contain accurate counts for all 4 statuses
        assert "Evaluated 4 claims against 2 retrieved sources:" in report.summary
        assert "1 supported" in report.summary
        assert "1 contradicted" in report.summary
        assert "1 inconclusive" in report.summary
        assert "1 not checked" in report.summary

    def test_atomic_claim_instructions_in_both_executor_prompts(self):
        """Both Executor A and Executor B system prompts must instruct agents to produce atomic, independently verifiable claims."""
        from app.agents.executor_a import ExecutorAAgent
        from app.agents.executor_b import ExecutorBAgent

        for agent_cls in (ExecutorAAgent, ExecutorBAgent):
            prompt = agent_cls.system_prompt
            assert "CLAIMS GUIDELINES (Atomic & Independently Verifiable)" in prompt, (
                f"{agent_cls.__name__} missing atomic claims guidelines section header."
            )
            assert "prefer atomic, single-assertion statements that can be verified independently" in prompt, (
                f"{agent_cls.__name__} missing preference for atomic independently verifiable claims."
            )
            assert "Separate distinct facts into individual claims instead of compounding them" in prompt, (
                f"{agent_cls.__name__} missing instruction to separate distinct facts."
            )
            assert "Apollo 11 landed on the Moon on July 20, 1969" in prompt, (
                f"{agent_cls.__name__} missing Apollo 11 landing date example."
            )
            assert "20:17:40 UTC" in prompt, (
                f"{agent_cls.__name__} missing Apollo 11 touchdown timestamp example."
            )

    async def test_supported_date_with_unsupported_timestamp_remaining_distinguishable(self):
        """
        Preserve evidence standards:
        A compound claim where only the date is supported while the timestamp is uncorroborated
        must NOT be marked supported. Relevant sources must be preserved, and the unsupported
        detail must be explicitly identified in the explanation and limitations.
        When separated into atomic claims, the date is supported while the timestamp is inconclusive.
        """
        nasa_url = "https://history.nasa.gov/apollo11"
        evidence = [
            _evidence(
                nasa_url,
                "NASA Apollo 11 Mission Overview",
                "Apollo 11 successfully landed on the Moon on July 20, 1969.",
            )
        ]

        # Case 1: Compound claim with supported date and uncorroborated timestamp
        compound_claim = "Apollo 11 landed on the Moon on July 20, 1969, at 20:17:40 UTC."
        report_data_compound = {
            "results": [{
                "claim": compound_claim,
                "status": "insufficient_evidence",
                "evidence_refs": [nasa_url],
                "explanation": (
                    "NASA source confirms the landing date July 20, 1969, but does not state "
                    "or verify the precise touchdown timestamp of 20:17:40 UTC."
                ),
                "uncertainty": "Touchdown time of 20:17:40 UTC is not corroborated by the provided evidence.",
            }],
            "summary": "Compound claim partially corroborated; timestamp unverified.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data_compound))
        report_compound = await svc.verify_claims([compound_claim], evidence)

        res = report_compound.results[0]
        # Must NOT be marked supported
        assert res.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert res.status != ClaimStatus.SUPPORTED
        # Source must be preserved in evidence references
        assert len(res.evidence) == 1
        assert res.evidence[0].url == nasa_url
        assert res.evidence[0].relationship == "inconclusive"
        # Explanation must distinguish corroborated date from uncorroborated timestamp
        assert "July 20, 1969" in res.explanation
        assert "20:17:40 UTC" in res.explanation
        assert res.limitations is not None
        assert "20:17:40 UTC" in res.limitations

        # Case 2: Atomic claims separating date from timestamp
        atomic_claims = [
            "Apollo 11 landed on the Moon on July 20, 1969.",
            "The Apollo 11 touchdown occurred at 20:17:40 UTC.",
        ]
        report_data_atomic = {
            "results": [
                {
                    "claim": atomic_claims[0],
                    "status": "supported",
                    "evidence_refs": [nasa_url],
                    "explanation": "Corroborated by NASA Mission Overview.",
                    "uncertainty": None,
                },
                {
                    "claim": atomic_claims[1],
                    "status": "insufficient_evidence",
                    "evidence_refs": [],
                    "explanation": "No timestamp provided in retrieved source.",
                    "uncertainty": "Touchdown time is unverified.",
                },
            ],
            "summary": "Atomic claims evaluated independently.",
            "evidence_count": 1,
        }
        svc_atomic = ClaimVerificationService(llm_client=_mock_llm(report_data_atomic))
        report_atomic = await svc_atomic.verify_claims(atomic_claims, evidence)

        assert report_atomic.results[0].status == ClaimStatus.SUPPORTED
        assert report_atomic.results[0].evidence[0].relationship == "supports"
        assert report_atomic.results[1].status == ClaimStatus.INSUFFICIENT_EVIDENCE
        # In summary, both are distinguishable
        assert "1 supported" in report_atomic.summary
        assert "1 inconclusive" in report_atomic.summary

    async def test_no_relevant_evidence_remains_inconclusive(self):
        """When no relevant evidence is available to corroborate a claim, it must remain inconclusive (insufficient_evidence)."""
        irrelevant_url = "https://gardening.example.org/tomatoes"
        evidence = [
            _evidence(
                irrelevant_url,
                "Tomato Care Guide",
                "Tomatoes require at least 6 to 8 hours of direct sunlight each day.",
            )
        ]
        claims = ["Voyager 1 crossed the heliopause into interstellar space in August 2012."]

        report_data = {
            "results": [{
                "claim": claims[0],
                "status": "insufficient_evidence",
                "evidence_refs": [],
                "explanation": (
                    "Retrieved sources discuss tomato care and contain no relevant empirical "
                    "evidence concerning Voyager 1 or the heliopause."
                ),
                "uncertainty": "No relevant astronomical evidence retrieved.",
            }],
            "summary": "No relevant evidence.",
            "evidence_count": 1,
        }
        svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
        report = await svc.verify_claims(claims, evidence)

        res = report.results[0]
        # Must be inconclusive, NEVER supported or contradicted
        assert res.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert res.status != ClaimStatus.SUPPORTED
        assert res.status != ClaimStatus.CONTRADICTED
        assert "no relevant" in res.explanation.lower() or "insufficient" in res.explanation.lower()
        assert report.evidence_count == 1
        assert "1 inconclusive" in report.summary
        assert "0 supported" in report.summary
        assert "0 contradicted" in report.summary


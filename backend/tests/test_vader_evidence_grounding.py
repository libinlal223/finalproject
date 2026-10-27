"""
tests/test_vader_evidence_grounding.py
───────────────────────────────────────
Regression and unit tests for evidence retrieval and final-answer grounding
improvements in VADER.

Verifies:
  1. Each claim produces an appropriate targeted retrieval query preserving
     entities, metrics, timeframe, and subject.
  2. Retrieved evidence is associated with the correct claim.
  3. Duplicate URLs are handled correctly across claim searches (canonicalization & deduplication).
  4. Evidence from an unrelated claim cannot support the current claim.
  5. A precise numerical claim remains inconclusive when its quantity is absent from evidence.
  6. A claim can be supported when a reliable mocked passage directly establishes subject, quantity, and timeframe.
  7. Inconclusive and contradicted claims are handled differently (absence != falsity).
  8. Unsupported quantitative and qualitative assertions do not remain unqualified in successful Judge output.
  9. The same protection works on fallback output.
 10. End-to-end orchestrator targeted retrieval and verification flow.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.agents.critic import CriticAgent
from app.agents.judge import JudgeAgent
from app.llm.base import LLMConfig, LLMResponse
from app.llm.mock_adapter import MockLLMClient
from app.models.schemas import (
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    DebateStatus,
    EvidenceItem,
    JudgeOutput,
    QueryResponse,
    RetrievalResponse,
    RetrievalStatus,
    VerificationReportStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.passage_extractor import extract_passage_for_claim
from app.retrieval.service import EvidenceRetrievalService
from app.retrieval.targeted_retrieval import derive_claim_query, retrieve_for_claims
from app.retrieval.url_normalizer import canonicalize_url


# ─── Helper utilities ─────────────────────────────────────────────────────────

def _make_mock_llm(report_dict: dict) -> MockLLMClient:
    return MockLLMClient(
        LLMConfig(
            extra={
                "mock_response": json.dumps(report_dict),
                "simulate_latency_ms": 0.0,
            }
        )
    )


def _item(url: str, title: str, snippet: str) -> EvidenceItem:
    return EvidenceItem(title=title, url=url, snippet=snippet)


# ─── 1. Targeted retrieval query formulation ──────────────────────────────────

class TestTargetedQueryFormulation:
    def test_preserves_entities_metrics_and_timeframes(self):
        """Query formulation must preserve core entity, numbers/metrics, and dates."""
        claim = "IBM Osprey achieved 433-qubit capacity in November 2022."
        query = derive_claim_query(claim)

        assert "IBM Osprey" in query
        assert "433-qubit" in query
        assert "2022" in query

    def test_preserves_leading_years_and_metrics(self):
        """Query formulation must not strip leading years or numerical metrics."""
        claim_year = "2023 saw inflation rise to 3.2% according to official figures."
        query_year = derive_claim_query(claim_year)
        assert query_year.startswith("2023")
        assert "3.2%" in query_year

        claim_metric = "433 superconducting qubits were demonstrated by IBM."
        query_metric = derive_claim_query(claim_metric)
        assert query_metric.startswith("433")
        assert "IBM" in query_metric

    def test_strips_boilerplate_prefixes_and_meta_phrases(self):
        """Conversational prefixes, bullet markers, and meta-phrasing must be stripped."""
        claim_bullet = "1. Claim 1: It is claimed that Google Sycamore has 53 qubits requiring verification."
        query = derive_claim_query(claim_bullet)

        assert not query.startswith("1.")
        assert not query.lower().startswith("claim 1")
        assert not query.lower().startswith("it is claimed that")
        assert "requiring verification" not in query.lower()
        assert "Google Sycamore" in query
        assert "53 qubits" in query

    def test_bounds_query_length_cleanly(self):
        """Very long claims must be cleanly bounded at a word boundary."""
        long_claim = "Detailed factual assertion: " + "quantum " * 50
        query = derive_claim_query(long_claim, max_query_len=60)
        assert len(query) <= 60
        assert not query.endswith(" ")


# ─── 2. Associate evidence with correct claim ─────────────────────────────────

@pytest.mark.asyncio
async def test_retrieved_evidence_associated_with_correct_claim():
    """Each claim must be mapped to its dedicated retrieved evidence items."""
    claims = [
        "IBM Osprey operates with 433 qubits.",
        "Google Sycamore demonstrated quantum supremacy with 53 qubits.",
    ]

    mock_provider = AsyncMock()
    mock_provider.provider_name = lambda: "mock"

    async def mock_search(query: str, max_results: int = 5):
        if "IBM" in query:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.SUCCESS,
                items=[_item("https://ibm.com/osprey", "IBM Osprey", "IBM built 433-qubit processor.")],
                total_results=1,
            )
        else:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.SUCCESS,
                items=[_item("https://google.com/sycamore", "Google Sycamore", "Google tested 53-qubit Sycamore.")],
                total_results=1,
            )

    mock_provider.search.side_effect = mock_search
    retrieval_service = EvidenceRetrievalService(provider=mock_provider)

    overall_status, all_items, claim_evidence_map, claim_status_map = await retrieve_for_claims(
        claims=claims,
        retrieval_service=retrieval_service,
        max_results_per_claim=2,
    )

    assert overall_status == RetrievalStatus.SUCCESS
    assert len(all_items) == 2

    # Verify dedicated association
    ibm_items = claim_evidence_map[claims[0]]
    google_items = claim_evidence_map[claims[1]]

    assert len(ibm_items) == 1
    assert ibm_items[0].url == "https://ibm.com/osprey"
    assert len(google_items) == 1
    assert google_items[0].url == "https://google.com/sycamore"

    # Distinct bundles
    assert ibm_items[0].url not in [i.url for i in google_items]


# ─── 3. Canonical URL deduplication across claim searches ─────────────────────

@pytest.mark.asyncio
async def test_duplicate_urls_handled_correctly_across_claim_searches():
    """Duplicate/tracking URLs across claims must be canonicalized and deduplicated in global pool."""
    claims = ["Claim A about quantum", "Claim B about post-quantum"]

    mock_provider = AsyncMock()
    mock_provider.provider_name = lambda: "mock"

    async def mock_search(query: str, max_results: int = 5):
        if "Claim A" in query or "quantum" in query:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.SUCCESS,
                items=[_item("https://example.org/report?utm_source=twitter&ref=social", "Shared Report", "Shared evidence snippet")],
                total_results=1,
            )
        else:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.SUCCESS,
                items=[_item("https://example.org/report#section2", "Shared Report", "Shared evidence snippet")],
                total_results=1,
            )

    mock_provider.search.side_effect = mock_search
    retrieval_service = EvidenceRetrievalService(provider=mock_provider)

    overall_status, all_items, claim_evidence_map, _ = await retrieve_for_claims(
        claims=claims,
        retrieval_service=retrieval_service,
    )

    assert overall_status == RetrievalStatus.SUCCESS
    # Both URLs canonicalize to https://example.org/report
    assert len(all_items) == 1
    assert all_items[0].url == "https://example.org/report"

    # Both claims still reference the canonical item
    assert len(claim_evidence_map[claims[0]]) == 1
    assert len(claim_evidence_map[claims[1]]) == 1
    assert claim_evidence_map[claims[0]][0].url == "https://example.org/report"


# ─── 4. Unrelated claim evidence cannot support current claim ─────────────────

@pytest.mark.asyncio
async def test_unrelated_claim_evidence_cannot_support_current_claim():
    """Evidence from an unrelated claim must not be cited or accepted to support another claim."""
    claim_a = "Company X acquired Startup Y for $500 million."
    claim_b = "Company Z released an open-source model with 70 billion parameters."

    item_b = _item("https://company-z.org/release", "Company Z", "Company Z released 70B parameter model.")

    # Claim A has no evidence; Claim B has item_b
    claim_evidence_map = {
        claim_a: [],
        claim_b: [item_b],
    }

    # Malicious/hallucinated LLM response attempts to mark Claim A as supported using Claim B's URL
    mock_report = {
        "results": [
            {
                "claim": claim_a,
                "status": "supported",
                "evidence_refs": ["https://company-z.org/release"],  # URL from Claim B!
                "explanation": "Corroborated by release notes.",
            },
            {
                "claim": claim_b,
                "status": "supported",
                "evidence_refs": ["https://company-z.org/release"],
                "explanation": "Directly confirmed by announcement.",
            },
        ],
        "summary": "Both claims evaluated.",
        "evidence_count": 1,
    }

    verifier = ClaimVerificationService(llm_client=_make_mock_llm(mock_report))
    report = await verifier.verify_claims(
        claims=[claim_a, claim_b],
        evidence=[item_b],
        claim_evidence_map=claim_evidence_map,
    )

    # Claim A must NOT be supported because item_b belongs to Claim B, not Claim A
    result_a = report.results[0]
    assert result_a.claim_text == claim_a
    assert result_a.status == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert len(result_a.evidence) == 0

    # Claim B remains supported by its own valid evidence
    result_b = report.results[1]
    assert result_b.status == ClaimStatus.SUPPORTED
    assert len(result_b.evidence) == 1


# ─── 5. Precise numerical claim remains inconclusive when quantity is absent ─

@pytest.mark.asyncio
async def test_precise_numerical_claim_inconclusive_when_quantity_absent():
    """A precise numerical claim remains inconclusive when its specific quantity is absent from evidence."""
    claim = "Turbine propulsion demonstrated 99.4% mechanical efficiency."
    # Evidence talks about high turbine efficiency but NEVER mentions 99.4%
    evidence = [_item("https://aero.example.org/turbine", "Turbine Test", "Turbine demonstrated high mechanical efficiency under flight regimes.")]

    # LLM hallucinates supported status
    mock_report = {
        "results": [
            {
                "claim": claim,
                "status": "supported",
                "evidence_refs": ["https://aero.example.org/turbine"],
                "explanation": "Article confirms high mechanical efficiency.",
            }
        ],
        "summary": "One claim evaluated.",
        "evidence_count": 1,
    }

    verifier = ClaimVerificationService(llm_client=_make_mock_llm(mock_report))
    report = await verifier.verify_claims([claim], evidence)

    result = report.results[0]
    # Because 99.4% is absent from the snippet, it must be demoted to INSUFFICIENT_EVIDENCE
    assert result.status == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert "99.4%" in result.explanation
    assert result.limitations is not None
    assert "quantity" in result.limitations.lower() or "numerical" in result.limitations.lower()


# ─── 6. Reliable mocked passage directly establishes subject, quantity, date ─

@pytest.mark.asyncio
async def test_claim_supported_when_mocked_passage_directly_establishes_facts():
    """A claim can be supported when evidence directly corroborates subject, quantity, and date."""
    claim = "IBM deployed its 433-qubit Osprey processor in November 2022."
    url = "https://news.ibm.com/osprey-2022"
    evidence = [_item(url, "IBM Osprey Release", "IBM deployed its 433-qubit Osprey quantum processor in November 2022.")]

    mock_report = {
        "results": [
            {
                "claim": claim,
                "status": "supported",
                "evidence_refs": [url],
                "explanation": "Release announcement directly confirms 433-qubit Osprey in November 2022.",
            }
        ],
        "summary": "One claim supported.",
        "evidence_count": 1,
    }

    verifier = ClaimVerificationService(llm_client=_make_mock_llm(mock_report))
    report = await verifier.verify_claims([claim], evidence)

    result = report.results[0]
    assert result.status == ClaimStatus.SUPPORTED
    assert len(result.evidence) == 1
    assert result.evidence[0].relationship == "supports"
    assert "433-qubit" in result.evidence[0].excerpt
    assert "November 2022" in result.evidence[0].excerpt


# ─── 7. Inconclusive vs Contradicted distinction ──────────────────────────────

@pytest.mark.asyncio
async def test_inconclusive_and_contradicted_claims_handled_differently():
    """Missing evidence must never be interpreted as proof of falsity."""
    claim_inconclusive = "Superconductor reaches room temperature at ambient pressure."
    claim_contradicted = "RSA-2048 encryption was cracked by a commercial quantum computer in 2023."

    url_contradiction = "https://crypto.example.org/rsa-status"
    evidence_contradiction = _item(
        url_contradiction,
        "NIST Cryptography Status",
        "RSA-2048 has not been broken; current quantum systems possess fewer than 1,200 noisy qubits.",
    )

    # Empty evidence for claim_inconclusive -> INSUFFICIENT_EVIDENCE
    # Direct refutation for claim_contradicted -> CONTRADICTED
    verifier = ClaimVerificationService(llm_client=_make_mock_llm({}))
    report_inconclusive = await verifier.verify_claims([claim_inconclusive], evidence=[], retrieval_status=RetrievalStatus.EMPTY)

    assert report_inconclusive.results[0].status == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert report_inconclusive.results[0].status != ClaimStatus.CONTRADICTED
    assert "not prove" in report_inconclusive.results[0].explanation.lower() or "lack" in report_inconclusive.results[0].explanation.lower()

    # Contradicted claim evaluation
    mock_contra_report = {
        "results": [
            {
                "claim": claim_contradicted,
                "status": "contradicted",
                "evidence_refs": [url_contradiction],
                "explanation": "NIST confirms RSA-2048 remains secure and unbroken.",
            }
        ],
        "summary": "One claim contradicted.",
        "evidence_count": 1,
    }
    verifier_contra = ClaimVerificationService(llm_client=_make_mock_llm(mock_contra_report))
    report_contra = await verifier_contra.verify_claims([claim_contradicted], [evidence_contradiction])

    assert report_contra.results[0].status == ClaimStatus.CONTRADICTED
    assert report_contra.results[0].evidence[0].relationship == "contradicts"


# ─── 8. Unsupported assertions qualified in Judge output ──────────────────────

@pytest.mark.asyncio
async def test_unsupported_assertions_qualified_in_judge_output():
    """Unsupported quantitative and qualitative assertions must not remain unqualified in Judge output."""
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer=(
                "The propulsion turbine demonstrates 99.4% mechanical efficiency in all regimes. "
                "Furthermore, according to NIST, all organizations must deprecate RSA keys by 2025."
            ),
            resolved_points=["Resolved aerodynamic models."],
            unresolved_uncertainties=["Operating envelope limits."],
            limitations=["Applies under nominal pressure."],
            evidence_citations=[],
            verdict_summary="Synthesis accepting turbine efficiency and NIST deprecation timeline.",
            adjudication_rationale="Based on candidate modeling.",
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=5.0),
    )

    verifications = [
        ClaimVerificationResult(
            claim_id="CLAIM-1",
            claim_text="Turbine demonstrates 99.4% mechanical efficiency",
            status=ClaimStatus.INSUFFICIENT_EVIDENCE,
            explanation="No test logs corroborate 99.4% efficiency figure.",
            evidence=[],
        ),
        ClaimVerificationResult(
            claim_id="CLAIM-2",
            claim_text="NIST recommends deprecating RSA keys by 2025",
            status=ClaimStatus.CONTRADICTED,
            explanation="NIST SP 800-131A Rev. 2 specifies deprecation by 2030, not 2025.",
            evidence=[],
        ),
    ]

    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=verifications,
        summary="Evaluated 2 claims: 0 supported, 1 contradicted, 1 inconclusive, 0 not checked.",
        evidence_count=2,
        claims_checked=2,
        claims_not_checked=0,
    )

    mock_provider = AsyncMock()
    mock_provider.provider_name = lambda: "mock"
    mock_provider.search.return_value = RetrievalResponse(
        query="test",
        status=RetrievalStatus.SUCCESS,
        items=[],
        total_results=0,
    )
    mock_retrieval = EvidenceRetrievalService(provider=mock_provider)

    orchestrator = DebateOrchestrator(
        judge=mock_judge,
        claim_verifier=mock_verifier,
        retrieval_service=mock_retrieval,
    )
    resp = await orchestrator.run_debate("Turbine efficiency and NIST cryptographic guidance", max_rounds=1)

    assert resp.status == DebateStatus.COMPLETED
    assert resp.final_answer is not None

    # Final answer must NOT contain the assertions without qualification
    assert "99.4%" in resp.final_answer
    assert "[Unverified / Inconclusive:" in resp.final_answer
    assert "[Contradicted by evidence:" in resp.final_answer

    # Both must appear in unresolved_uncertainties
    assert any("CLAIM-1" in unc or "99.4%" in unc for unc in resp.unresolved_uncertainties)
    assert any("CLAIM-2" in unc or "2025" in unc for unc in resp.unresolved_uncertainties)


# ─── 9. Protection works on fallback output ───────────────────────────────────

@pytest.mark.asyncio
async def test_protection_works_on_fallback_output():
    """The same grounding protection must qualify assertions in unadjudicated fallback output."""
    mock_critic = AsyncMock(spec=CriticAgent)
    mock_critic.critique.side_effect = RuntimeError("Critic crashed.")

    verifications = [
        ClaimVerificationResult(
            claim_id="CLAIM-1",
            claim_text="Turbine achieves 99.4% efficiency",
            status=ClaimStatus.INSUFFICIENT_EVIDENCE,
            explanation="Uncorroborated by external sources.",
            evidence=[],
        ),
        ClaimVerificationResult(
            claim_id="CLAIM-2",
            claim_text="Quantum computer cracked RSA-2048 in 2023",
            status=ClaimStatus.CONTRADICTED,
            explanation="Contradicted by cryptographic literature.",
            evidence=[],
        ),
    ]

    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=verifications,
        summary="Evaluated 2 claims: 1 contradicted, 1 inconclusive.",
        evidence_count=0,
        claims_checked=2,
        claims_not_checked=0,
    )

    mock_provider = AsyncMock()
    mock_provider.provider_name = lambda: "mock"
    mock_provider.search.return_value = RetrievalResponse(
        query="test",
        status=RetrievalStatus.SUCCESS,
        items=[],
        total_results=0,
    )
    mock_retrieval = EvidenceRetrievalService(provider=mock_provider)

    orchestrator = DebateOrchestrator(
        critic=mock_critic,
        claim_verifier=mock_verifier,
        retrieval_service=mock_retrieval,
    )

    # Patch cand_a to assert the unsupported claims
    raw_candidate = (
        "Turbine achieves 99.4% efficiency under flight regimes. "
        "Also, quantum computer cracked RSA-2048 in 2023."
    )
    with patch.object(
        orchestrator._executor_a, "execute",
        AsyncMock(return_value=(
            AsyncMock(answer=raw_candidate, approach_summary="A", key_assumptions=[], claims_to_verify=[]),
            LLMResponse(content="", model_id="m", provider="p", latency_ms=1.0),
        ))
    ):
        resp = await orchestrator.run_debate("Fallback test query", max_rounds=1)

    assert resp.final_answer is not None
    assert "[UNADJUDICATED FALLBACK CANDIDATE - NOT VERIFIED]" in resp.final_answer

    # Inside the candidate body, assertions must be qualified
    assert "[Unverified / Inconclusive:" in resp.final_answer
    assert "[Contradicted by evidence:" in resp.final_answer

    # Prominent disclosure must be present at bottom
    assert "[UNVERIFIED / INCONCLUSIVE CLAIMS DISCLOSURE]:" in resp.final_answer
    assert "[CONTRADICTED CLAIMS DISCLOSURE]:" in resp.final_answer

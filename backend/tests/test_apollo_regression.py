"""
tests/test_apollo_regression.py
───────────────────────────────
Regression tests covering:
  1. Search relevance: Query construction and result filtering prioritizing authoritative NASA sources.
  2. Rejecting unrelated football search results: Keyword overlap alone ("Apollo", "touchdown") does NOT support claims.
  3. Date-only evidence: Supports landing date (July 20, 1969) but does not establish exact touchdown time.
  4. Evidence that does not establish exact touchdown time: Preserves INSUFFICIENT_EVIDENCE (inconclusive).
  5. Judge consistency: Judge cannot describe an unverified claim as verified when marked INSUFFICIENT_EVIDENCE.
  6. Readable final synthesis: Removes repeated machine-generated warning blocks from main prose,
     stating each key uncertainty once in a concise uncertainty section.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

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
    RetrievalStatus,
    VerificationReportStatus,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.relevance import get_source_authority_score, is_result_topic_relevant
from app.retrieval.targeted_retrieval import derive_claim_query


def _mock_llm(report_dict: dict) -> MockLLMClient:
    return MockLLMClient(
        LLMConfig(
            extra={
                "mock_response": json.dumps(report_dict),
                "simulate_latency_ms": 0.0,
            }
        )
    )


# ─── 1. Search Relevance & Query Construction ──────────────────────────────────

def test_apollo_11_query_construction_prioritizes_nasa_mission():
    """Query construction from an Apollo 11 claim should include authoritative NASA mission keywords."""
    claim = "Apollo 11 landed on the Moon on July 20, 1969."
    query = derive_claim_query(claim)

    assert "Apollo 11" in query
    assert "July 20 1969" in query or "July 20" in query
    assert "NASA" in query or "mission" in query


def test_authoritative_nasa_source_scoring():
    """NASA mission documentation must receive top authority scores over general sources."""
    nasa_item = EvidenceItem(
        title="Apollo 11 Mission Overview - NASA",
        url="https://www.nasa.gov/mission_pages/apollo/apollo-11.html",
        snippet="Apollo 11 launched on July 16, 1969 and landed on July 20, 1969.",
        domain="www.nasa.gov",
    )
    wiki_item = EvidenceItem(
        title="Apollo 11 - Wikipedia",
        url="https://en.wikipedia.org/wiki/Apollo_11",
        snippet="Apollo 11 was the American spaceflight that first landed humans on the Moon.",
        domain="en.wikipedia.org",
    )
    commercial_item = EvidenceItem(
        title="Apollo Memorabilia Store",
        url="https://commercialstore.example.com/apollo",
        snippet="Buy commemorative Apollo 11 landing pins and space memorabilia.",
        domain="commercialstore.example.com",
    )

    query = "Apollo 11 Moon landing"
    score_nasa = get_source_authority_score(nasa_item, query)
    score_wiki = get_source_authority_score(wiki_item, query)
    score_commercial = get_source_authority_score(commercial_item, query)

    assert score_nasa > score_wiki > score_commercial
    assert score_nasa >= 100.0


# ─── 2. Unrelated Football Search Results Filtering & Keyword Overlap ──────────

def test_unrelated_football_results_are_rejected():
    """Football search results with keyword overlap ('Apollo', 'touchdown') must be rejected as off-topic."""
    football_item = EvidenceItem(
        title="Apollo High School Football Roster and Schedule",
        url="https://maxpreps.example.com/apollo-football",
        snippet="The Apollo Eagles varsity football team scored a 4th-quarter touchdown to win Friday's conference game 24-21.",
        domain="maxpreps.example.com",
    )
    query = "Apollo 11 touchdown time on the Moon"

    is_relevant = is_result_topic_relevant(query, football_item)
    assert is_relevant is False, "Football search result should be rejected for an Apollo 11 lunar query."


@pytest.mark.asyncio
async def test_keyword_overlap_football_source_cannot_support_claim():
    """Keyword overlap alone ('Apollo', 'touchdown') from a football source must not support an Apollo 11 claim."""
    football_url = "https://maxpreps.example.com/apollo-football"
    evidence = [
        EvidenceItem(
            title="Apollo Football Touchdown News",
            url=football_url,
            snippet="Apollo scored an unbelievable touchdown with 20 seconds remaining in the game.",
            domain="maxpreps.example.com",
        )
    ]
    claim = "The Apollo 11 touchdown occurred at 20:17:40 UTC."

    # Even if an uncalibrated LLM returned 'supported' due to keyword matching:
    report_data = {
        "results": [
            {
                "claim": claim,
                "status": "supported",
                "evidence_refs": [football_url],
                "explanation": "Matched 'Apollo' and 'touchdown' in snippet.",
                "uncertainty": None,
            }
        ],
        "summary": "Keyword match evaluation.",
        "evidence_count": 1,
    }

    svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
    report = await svc.verify_claims([claim], evidence)

    result = report.results[0]
    # Must NOT be supported; must remain INSUFFICIENT_EVIDENCE
    assert result.status == ClaimStatus.INSUFFICIENT_EVIDENCE
    assert result.status != ClaimStatus.SUPPORTED
    assert len(result.evidence) == 0 or all(e.relationship == "inconclusive" for e in result.evidence)
    assert "unrelated" in result.explanation.lower() or "off-topic" in result.explanation.lower() or "insufficient" in result.explanation.lower()


# ─── 3. Date-Only Evidence Supports Landing Date ──────────────────────────────

@pytest.mark.asyncio
async def test_date_only_evidence_supports_landing_date():
    """A general Apollo 11 NASA page explicitly stating July 20, 1969 supports the landing date claim."""
    nasa_url = "https://www.nasa.gov/mission_pages/apollo/apollo-11.html"
    evidence = [
        EvidenceItem(
            title="Apollo 11 Mission Overview - NASA",
            url=nasa_url,
            snippet="Apollo 11 launched from Cape Kennedy on July 16, 1969. Neil Armstrong and Buzz Aldrin landed the Lunar Module Eagle on the Moon on July 20, 1969.",
            domain="www.nasa.gov",
        )
    ]
    date_claim = "Apollo 11 landed on the Moon on July 20, 1969."

    report_data = {
        "results": [
            {
                "claim": date_claim,
                "status": "supported",
                "evidence_refs": [nasa_url],
                "explanation": "Corroborated by NASA Mission Overview: lunar landing occurred on July 20, 1969.",
                "uncertainty": None,
            }
        ],
        "summary": "1 claim supported.",
        "evidence_count": 1,
    }

    svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
    report = await svc.verify_claims([date_claim], evidence)

    res = report.results[0]
    assert res.status == ClaimStatus.SUPPORTED
    assert len(res.evidence) == 1
    assert res.evidence[0].url == nasa_url
    assert res.evidence[0].relationship == "supports"
    assert "1 supported" in report.summary


# ─── 4. Evidence Lacking Touchdown Time Preserves Inconclusive ─────────────────

@pytest.mark.asyncio
async def test_date_only_evidence_does_not_support_touchdown_time_or_met():
    """Date-only evidence does not support exact touchdown time (20:17:40 UTC) or mission elapsed time (102:45:40)."""
    nasa_url = "https://www.nasa.gov/mission_pages/apollo/apollo-11.html"
    evidence = [
        EvidenceItem(
            title="Apollo 11 Mission Overview - NASA",
            url=nasa_url,
            snippet="Apollo 11 launched on July 16, 1969. The Lunar Module Eagle landed on the Moon on July 20, 1969.",
            domain="www.nasa.gov",
        )
    ]

    claims = [
        "The Apollo 11 touchdown occurred at 20:17:40 UTC.",
        "The mission elapsed time at lunar touchdown was 102:45:40.",
    ]

    # Even if LLM erroneously marked both 'supported':
    report_data = {
        "results": [
            {
                "claim": claims[0],
                "status": "supported",
                "evidence_refs": [nasa_url],
                "explanation": "Extrapolated from general Apollo 11 page.",
                "uncertainty": None,
            },
            {
                "claim": claims[1],
                "status": "supported",
                "evidence_refs": [nasa_url],
                "explanation": "Assumed from mission overview.",
                "uncertainty": None,
            },
        ],
        "summary": "Evaluated timestamp claims.",
        "evidence_count": 1,
    }

    svc = ClaimVerificationService(llm_client=_mock_llm(report_data))
    report = await svc.verify_claims(claims, evidence)

    for res in report.results:
        # Both must remain INSUFFICIENT_EVIDENCE (inconclusive)
        assert res.status == ClaimStatus.INSUFFICIENT_EVIDENCE
        assert res.status != ClaimStatus.SUPPORTED
        assert "exact touchdown time" in res.explanation.lower() or "mission elapsed time" in res.explanation.lower() or "uncorroborated" in (res.limitations or "").lower()

    assert "2 inconclusive" in report.summary
    assert "0 supported" in report.summary


# ─── 5. Judge Consistency: Inconclusive Claim Cannot Be Presented as Verified ─

@pytest.mark.asyncio
async def test_judge_cannot_describe_inconclusive_touchdown_time_as_verified():
    """Judge cannot describe touchdown time as verified when claim verification marks it INSUFFICIENT_EVIDENCE."""
    mock_judge = AsyncMock(spec=JudgeAgent)
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer=(
                "Apollo 11 landed on the Moon on July 20, 1969. "
                "The touchdown time of 20:17:40 UTC has been verified by NASA mission documentation."
            ),
            resolved_points=["Reconciled landing timeline."],
            unresolved_uncertainties=["Exact touchdown time."],
            limitations=["Based on retrieved documentation."],
            evidence_citations=["https://www.nasa.gov/mission_pages/apollo/apollo-11.html"],
            verdict_summary="Touchdown time verified by mission records.",
            adjudication_rationale="Accepted candidate synthesis.",
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=5.0),
    )

    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    inconclusive_touchdown = ClaimVerificationResult(
        claim_id="CLAIM-TIME-1",
        claim_text="The Apollo 11 touchdown occurred at 20:17:40 UTC",
        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
        explanation="Retrieved source confirms landing date July 20, 1969 but lacks direct evidence for exact timestamp 20:17:40 UTC.",
        evidence=[],
        limitations="Exact touchdown time uncorroborated in retrieved records.",
    )
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=[inconclusive_touchdown],
        summary="1 inconclusive claim.",
        evidence_count=1,
        claims_checked=1,
        claims_not_checked=0,
    )

    orchestrator = DebateOrchestrator(judge=mock_judge, claim_verifier=mock_verifier)
    resp = await orchestrator.run_debate("When did Apollo 11 land on the Moon?", max_rounds=1)

    assert resp.status == DebateStatus.COMPLETED
    assert resp.final_answer is not None

    # The Judge's false assertion of verification MUST be prevented:
    # It must NOT state that 20:17:40 UTC has been verified
    assert "verified by NASA" not in resp.final_answer
    assert "verified by mission documentation" not in resp.final_answer

    # Verdict summary must also NOT claim all claims / touchdown time are verified
    assert "verified" not in (resp.verdict_summary or "").lower() or "unverified" in (resp.verdict_summary or "").lower() or "inconclusive" in (resp.verdict_summary or "").lower()

    # The unverified touchdown time must appear in unresolved_uncertainties
    assert any("CLAIM-TIME-1" in unc or "20:17:40" in unc for unc in resp.unresolved_uncertainties)


# ─── 6. Readable Final Synthesis: No Repeated Inline Warning Blocks ────────────

@pytest.mark.asyncio
async def test_readable_final_synthesis_no_repeated_inline_warnings():
    """Main prose must be clean and readable without repeated machine-generated warning blocks spliced into sentences."""
    mock_judge = AsyncMock(spec=JudgeAgent)
    raw_answer_with_tags = (
        "Apollo 11 landed on the Moon on July 20, 1969 [Unverified / Inconclusive: duplicate machine tag]. "
        "Neil Armstrong and Buzz Aldrin touched down at 20:17:40 UTC [Unverified / Inconclusive: duplicate machine tag]. "
        "The Lunar Module Eagle was safely on the surface [Unverified / Inconclusive: duplicate machine tag]."
    )
    mock_judge.judge.return_value = (
        JudgeOutput(
            final_answer=raw_answer_with_tags,
            resolved_points=[],
            unresolved_uncertainties=[],
            limitations=[],
            evidence_citations=[],
            verdict_summary="Synthesis completed.",
            adjudication_rationale="Evidence-grounded adjudication.",
        ),
        LLMResponse(content="", model_id="mock", provider="mock", latency_ms=5.0),
    )

    inconclusive_claim = ClaimVerificationResult(
        claim_id="CLAIM-1",
        claim_text="Touchdown occurred at 20:17:40 UTC",
        status=ClaimStatus.INSUFFICIENT_EVIDENCE,
        explanation="No timestamp in retrieved general overview.",
        evidence=[],
    )
    mock_verifier = AsyncMock(spec=ClaimVerificationService)
    mock_verifier.verify_claims.return_value = ClaimVerificationReport(
        report_status=VerificationReportStatus.COMPLETED,
        results=[inconclusive_claim],
        summary="1 inconclusive claim.",
        evidence_count=1,
        claims_checked=1,
        claims_not_checked=0,
    )

    orchestrator = DebateOrchestrator(judge=mock_judge, claim_verifier=mock_verifier)
    resp = await orchestrator.run_debate("When did Apollo 11 land on the Moon?", max_rounds=1)

    assert resp.final_answer is not None

    # Check main prose: the repeated inline warning tags must be stripped from sentences
    main_prose = resp.final_answer.split("### Key Uncertainties")[0]
    assert "[Unverified / Inconclusive: duplicate machine tag]" not in main_prose

    # In the Key Uncertainties section, the uncertainty should appear once concisely:
    assert "[Unverified / Inconclusive:" in resp.final_answer
    assert resp.final_answer.count("### Key Uncertainties") <= 1

    # Detailed explanation remains intact in the verification report
    assert resp.verification_report.results[0].explanation == "No timestamp in retrieved general overview."

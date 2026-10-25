"""
tests/test_schemas.py
──────────────────────
Tests for Pydantic request/response schema validation.

Verifies:
  ✓ Valid QueryRequest is accepted
  ✓ Short queries (<3 chars) are rejected
  ✓ Blank/whitespace-only queries are rejected
  ✓ Queries exceeding max_length are rejected
  ✓ max_rounds out-of-range values are rejected
  ✓ Unknown llm_provider values are rejected
  ✓ Valid BaselineRequest is accepted
  ✓ HealthResponse serialises without error
  ✓ ErrorDetail serialises correctly
  ✓ Input is normalised (stripped, whitespace collapsed)
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import (
    AgentRole,
    AgentTraceEntry,
    BaselineRequest,
    DebateRound,
    DebateStatus,
    ErrorDetail,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)


class TestQueryRequest:
    """QueryRequest schema validation."""

    def test_valid_query(self):
        req = QueryRequest(query="What are the causes of inflation?")
        assert req.query == "What are the causes of inflation?"

    def test_query_is_stripped(self):
        req = QueryRequest(query="  hello world  ")
        assert req.query == "hello world"

    def test_internal_whitespace_collapsed(self):
        req = QueryRequest(query="what   is   AI")
        assert req.query == "what is AI"

    def test_query_too_short_raises(self):
        with pytest.raises(ValidationError) as exc_info:
            QueryRequest(query="hi")
        errors = exc_info.value.errors()
        assert any("min_length" in str(e) or "at least" in str(e).lower() for e in errors)

    def test_blank_query_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="   ")

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_query_at_max_length_accepted(self):
        req = QueryRequest(query="a" * 4000)
        assert len(req.query) == 4000

    def test_query_exceeds_max_length_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="a" * 4001)

    def test_max_rounds_valid(self):
        req = QueryRequest(query="valid query here", max_rounds=5)
        assert req.max_rounds == 5

    def test_max_rounds_zero_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query here", max_rounds=0)

    def test_max_rounds_too_high_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query here", max_rounds=11)

    def test_max_rounds_none_is_valid(self):
        req = QueryRequest(query="valid query here", max_rounds=None)
        assert req.max_rounds is None

    def test_invalid_llm_provider_raises(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="valid query here", llm_provider="openai")

    def test_valid_llm_provider_mock(self):
        req = QueryRequest(query="valid query here", llm_provider="mock")
        assert req.llm_provider.value == "mock"

    def test_valid_llm_provider_groq(self):
        req = QueryRequest(query="valid query here", llm_provider="groq")
        assert req.llm_provider.value == "groq"


class TestBaselineRequest:
    """BaselineRequest schema validation."""

    def test_valid_baseline_request(self):
        req = BaselineRequest(query="Explain quantum entanglement.")
        assert req.query == "Explain quantum entanglement."

    def test_short_query_raises(self):
        with pytest.raises(ValidationError):
            BaselineRequest(query="ab")

    def test_query_stripped(self):
        req = BaselineRequest(query="  padded query here  ")
        assert req.query == "padded query here"


class TestHealthResponse:
    """HealthResponse serialisation."""

    def test_health_response_serialises(self):
        resp = HealthResponse(
            version="0.1.0",
            environment="test",
            llm_provider="mock",
            llm_key_present=True,
        )
        data = resp.model_dump()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"
        assert data["llm_key_present"] is True

    def test_health_response_json_round_trip(self):
        resp = HealthResponse(
            version="0.1.0",
            environment="development",
            llm_provider="groq",
            llm_key_present=False,
        )
        json_str = resp.model_dump_json()
        assert "ok" in json_str
        assert "groq" in json_str


class TestErrorDetail:
    """ErrorDetail serialisation."""

    def test_error_detail_without_details(self):
        err = ErrorDetail(error="not_found", message="Resource not found.")
        data = err.model_dump()
        assert data["error"] == "not_found"
        assert data["details"] is None

    def test_error_detail_with_details(self):
        err = ErrorDetail(
            error="validation_error",
            message="Query too short.",
            details={"field": "query", "min_length": 3},
        )
        data = err.model_dump()
        assert data["details"]["field"] == "query"


class TestAgentTraceEntry:
    """AgentTraceEntry validation."""

    def test_valid_trace_entry(self):
        entry = AgentTraceEntry(
            agent=AgentRole.CRITIC,
            round_number=1,
            output="The claim lacks supporting evidence.",
        )
        assert entry.agent == AgentRole.CRITIC
        assert entry.evidence == []
        assert entry.unresolved_issues == []

    def test_negative_round_raises(self):
        with pytest.raises(ValidationError):
            AgentTraceEntry(
                agent=AgentRole.JUDGE,
                round_number=-1,
                output="Final answer.",
            )


class TestDebateRound:
    """DebateRound validation."""

    def test_valid_debate_round(self):
        dr = DebateRound(
            round_number=1,
            executor_a_answer="Answer A",
            executor_b_answer="Answer B",
            critique="Both answers are incomplete.",
        )
        assert dr.converged is False
        assert dr.similarity_score is None

    def test_similarity_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            DebateRound(
                round_number=1,
                executor_a_answer="A",
                executor_b_answer="B",
                critique="C",
                similarity_score=1.5,
            )

    def test_round_number_zero_raises(self):
        with pytest.raises(ValidationError):
            DebateRound(
                round_number=0,
                executor_a_answer="A",
                executor_b_answer="B",
                critique="C",
            )


class TestQueryResponse:
    """QueryResponse schema."""

    def test_minimal_query_response(self):
        resp = QueryResponse(
            query="Test query",
            status=DebateStatus.PENDING,
            rounds_completed=0,
        )
        assert resp.final_answer is None
        assert resp.trace == []
        assert resp.sources == []
        assert resp.debate_rounds == []

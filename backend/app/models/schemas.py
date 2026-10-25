"""
app/models/schemas.py
─────────────────────
Pydantic v2 schemas for all API request/response contracts.

Rules enforced here:
  - All user-supplied strings are stripped and length-bounded.
  - Enum fields constrain accepted values to a known set.
  - No internal chain-of-thought is exposed through response models.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

# ─── Shared helpers ───────────────────────────────────────────────────────────

def _strip(v: str) -> str:
    """Strip leading/trailing whitespace."""
    return v.strip() if isinstance(v, str) else v


# ─── Enumerations ─────────────────────────────────────────────────────────────

class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR_A = "executor_a"
    EXECUTOR_B = "executor_b"
    CRITIC = "critic"
    JUDGE = "judge"
    BASELINE = "baseline"


class DebateStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    CONVERGED = "converged"       # similarity threshold met
    MAX_ROUNDS = "max_rounds"     # hard round limit reached
    COMPLETED = "completed"       # judge produced final answer
    ERROR = "error"


class LLMProviderEnum(str, Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    MOCK = "mock"


# ─── Health endpoint ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response from GET /api/health."""

    status: str = Field(
        default="ok",
        description="Always 'ok' when the service is reachable.",
    )
    version: str = Field(description="Application version string.")
    environment: str = Field(description="Runtime environment name.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp of the health check.",
    )
    llm_provider: str = Field(description="Active LLM provider name.")
    llm_key_present: bool = Field(
        description=(
            "True if the API key for the active LLM provider is configured. "
            "A missing key does NOT prevent the health endpoint from responding."
        ),
    )

    model_config = {"json_schema_extra": {"example": {
        "status": "ok",
        "version": "0.1.0",
        "environment": "development",
        "timestamp": "2026-10-16T00:00:00Z",
        "llm_provider": "mock",
        "llm_key_present": True,
    }}}


# ─── Query request ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """
    POST /api/query — user submits a question to the multi-agent debate.
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=4000,
        description="The user's question or topic for debate.",
    )
    max_rounds: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description=(
            "Override the server-level max rounds for this request. "
            "Omit to use the server default."
        ),
    )
    llm_provider: LLMProviderEnum | None = Field(
        default=None,
        description=(
            "Override the active LLM provider for this request. "
            "Omit to use the server default."
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _clean_query(cls, v: str) -> str:
        v = _strip(v)
        # Collapse internal whitespace runs to single spaces (normalise input)
        v = re.sub(r"[ \t]{2,}", " ", v)
        return v

    @model_validator(mode="after")
    def _validate_query_not_blank(self) -> QueryRequest:
        if not self.query:
            raise ValueError("query must not be blank after stripping whitespace.")
        return self

    model_config = {"json_schema_extra": {"example": {
        "query": "What are the main causes of inflation?",
        "max_rounds": 3,
    }}}


# ─── Agent structured output schemas ──────────────────────────────────────────

class TaskPlan(BaseModel):
    """Structured plan produced by the Planner agent."""

    core_question: str = Field(description="The primary question to answer.")
    subtasks: list[str] = Field(
        min_length=1,
        description="Logical breakdown of analytical subtasks or subquestions.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Identified boundaries, scoping rules, or constraints.",
    )
    expected_format: str = Field(
        default="structured_synthesis",
        description="Expected structure or format of the candidate answers.",
    )
    evidence_needed: list[str] = Field(
        default_factory=list,
        description="Topics, factual claims, or empirical points requiring external evidence.",
    )


class ExecutorOutput(BaseModel):
    """Structured candidate answer produced by Executor A or Executor B."""

    executor_id: str = Field(
        description="Identifier of the executing agent ('executor_a' or 'executor_b').",
    )
    approach_summary: str = Field(
        description="Concise description of the reasoning strategy applied.",
    )
    answer: str = Field(
        min_length=1,
        description="The candidate answer addressing the task plan.",
    )
    key_assumptions: list[str] = Field(
        default_factory=list,
        description="Key assumptions made in reaching the conclusion.",
    )
    claims_to_verify: list[str] = Field(
        default_factory=list,
        description="Factual or empirical claims that warrant verification.",
    )


class CritiqueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CritiqueIssue(BaseModel):
    """A specific flaw, contradiction, or unsupported claim found by the Critic."""

    issue_id: str = Field(description="Unique issue key, e.g. 'ISSUE-1'.")
    target: str = Field(
        description="Which candidate is affected: 'executor_a', 'executor_b', or 'both'.",
    )
    severity: CritiqueSeverity = Field(
        description="Severity level of the critique issue.",
    )
    affected_claim: str = Field(
        description="The exact claim or premise being challenged.",
    )
    explanation: str = Field(
        description="Why the claim is unsupported, contradictory, or flawed.",
    )
    recommended_correction: str = Field(
        description="How the candidate should revise or qualify the claim.",
    )


class CriticOutput(BaseModel):
    """Structured adversarial critique produced by the Critic agent."""

    agreements: list[str] = Field(
        default_factory=list,
        description="Claims agreed upon by both executors (not assumed to be true).",
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Direct contradictions or inconsistencies between the two executors.",
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Claims made without sufficient logical or factual support.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Explicitly identified areas where evidence is inconclusive.",
    )
    issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="Detailed list of specific critique issues.",
    )
    overall_critique: str = Field(
        min_length=1,
        description="Concise overall evaluation of both candidate answers.",
    )


class JudgeOutput(BaseModel):
    """Final synthesis and adjudication produced by the Judge agent."""

    final_answer: str = Field(
        min_length=1,
        description="The synthesised final answer addressing the user query.",
    )
    resolved_points: list[str] = Field(
        default_factory=list,
        description="Points of disagreement that were resolved with a defensible basis.",
    )
    unresolved_uncertainties: list[str] = Field(
        default_factory=list,
        description="Genuine uncertainties preserved due to lack of conclusive evidence.",
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Key limitations or boundary conditions of the final answer.",
    )
    evidence_citations: list[str] = Field(
        default_factory=list,
        description="Citations or references to supporting evidence items.",
    )
    verdict_summary: str = Field(
        min_length=1,
        description="Concise rationale for the final adjudication decision.",
    )


# ─── Evidence / source ────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of retrieved evidence used by the Critic and verification services."""

    title: str = Field(description="Title of the source document or page.")
    url: str = Field(description="URL of the source.")
    snippet: str = Field(description="Relevant excerpt from the source.")
    domain: str | None = Field(default=None, description="Host domain of the source.")
    source: str | None = Field(default=None, description="Retrieval source provider name.")
    score: float | None = Field(default=None, description="Relevance or confidence score.")
    retrieved_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


# ─── Retrieval schemas ────────────────────────────────────────────────────────

class RetrievalStatus(str, Enum):
    SUCCESS = "success"
    EMPTY = "empty"
    ERROR = "error"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"


class RetrievalResponse(BaseModel):
    """Standardized response from the evidence retrieval service."""

    query: str = Field(description="Search query executed.")
    status: RetrievalStatus = Field(description="Status of the search retrieval.")
    items: list[EvidenceItem] = Field(
        default_factory=list,
        description="Deduplicated and validated evidence items.",
    )
    total_results: int = Field(default=0, description="Number of valid items returned.")
    error_message: str | None = Field(
        default=None,
        description="Error details if status is not 'success' or 'empty'.",
    )
    latency_ms: float = Field(default=0.0, description="Retrieval latency in ms.")


# ─── Claim verification schemas ───────────────────────────────────────────────

class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class ClaimVerificationResult(BaseModel):
    """Verification assessment for an individual factual claim."""

    claim: str = Field(description="The original claim evaluated.")
    status: ClaimStatus = Field(description="Assessment status based on evidence.")
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="URLs or titles of verified evidence items directly cited.",
    )
    explanation: str = Field(
        min_length=1,
        description="Concise rationale explaining the evidence support, contradiction, or insufficiency.",
    )
    uncertainty: str | None = Field(
        default=None,
        description="Explicit uncertainties, source limitations, or conflicting evidence.",
    )


class ClaimVerificationReport(BaseModel):
    """Aggregated claim verification report produced by the claim-checking service."""

    results: list[ClaimVerificationResult] = Field(
        default_factory=list,
        description="Individual claim verification findings.",
    )
    summary: str = Field(
        min_length=1,
        description="Overall summary of claim verification results.",
    )
    evidence_count: int = Field(
        default=0,
        description="Total number of evidence items considered.",
    )


# ─── Per-agent trace entry ────────────────────────────────────────────────────

class AgentTraceEntry(BaseModel):
    """
    A single recorded action by one agent in one debate round.

    Internal chain-of-thought is NOT included.  Only the agent's
    public output (answer, critique, revision, etc.) is exposed.
    """

    agent: AgentRole
    round_number: int = Field(ge=0)
    output: str = Field(description="The agent's public output for this step.")
    evidence: list[EvidenceItem] = Field(
        default_factory=list,
        description="Evidence retrieved and used in this step (Critic only).",
    )
    unresolved_issues: list[str] = Field(
        default_factory=list,
        description="Issues the Critic raised that were not resolved in this round.",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


# ─── Per-debate-round summary ─────────────────────────────────────────────────

class DebateRound(BaseModel):
    """Summary of one full debate round (both executors + critic)."""

    round_number: int = Field(ge=1)
    executor_a_answer: str
    executor_b_answer: str
    critique: str
    similarity_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Cosine-similarity between executor answers (optional convergence signal). "
            "NOT a correctness metric."
        ),
    )
    converged: bool = Field(
        default=False,
        description="True if similarity exceeded the threshold in this round.",
    )


# ─── Query response ───────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """
    POST /api/query response — full debate result.
    """

    query: str = Field(description="The original user query (echoed back).")
    status: DebateStatus
    rounds_completed: int = Field(ge=0)
    final_answer: str | None = Field(
        default=None,
        description="Judge's synthesised answer (present when status is 'completed').",
    )
    subtasks: list[str] = Field(
        default_factory=list,
        description="Planner's decomposition of the original query.",
    )
    debate_rounds: list[DebateRound] = Field(default_factory=list)
    trace: list[AgentTraceEntry] = Field(
        default_factory=list,
        description="Full transparent activity trace of all agent steps.",
    )
    sources: list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence items surfaced across all rounds.",
    )
    error_message: str | None = Field(
        default=None,
        description="Human-readable error description (present when status is 'error').",
    )


# ─── Baseline request / response ─────────────────────────────────────────────

class BaselineRequest(BaseModel):
    """
    POST /api/baseline — run a single-model response for comparison.
    """

    query: str = Field(..., min_length=3, max_length=4000)
    llm_provider: LLMProviderEnum | None = Field(default=None)

    @field_validator("query", mode="before")
    @classmethod
    def _clean(cls, v: str) -> str:
        return _strip(v)


class BaselineResponse(BaseModel):
    """Single-model baseline result."""

    query: str
    answer: str
    llm_provider: str
    model_id: str
    latency_ms: float = Field(description="Wall-clock time for the LLM call in ms.")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


# ─── Error response ───────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    """Standardised error payload returned on 4xx / 5xx responses."""

    error: str = Field(description="Short machine-readable error code.")
    message: str = Field(description="Human-readable description.")
    details: dict[str, Any] | None = Field(
        default=None,
        description="Optional structured details (e.g. validation errors).",
    )

    model_config = {"json_schema_extra": {"example": {
        "error": "validation_error",
        "message": "query must be at least 3 characters long.",
        "details": {"field": "query", "value": "hi"},
    }}}

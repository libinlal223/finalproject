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
    addressed_issues: list[str] = Field(
        default_factory=list,
        description="IDs of critique issues addressed in this candidate or revision (e.g. ['ISSUE-1']).",
    )
    retained_claims: list[str] = Field(
        default_factory=list,
        description="Supported claims intentionally preserved during revision.",
    )
    revision_notes: str | None = Field(
        default=None,
        description="Concise description of changes made since prior round.",
    )


class CritiqueSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CritiqueIssue(BaseModel):
    """A specific flaw, contradiction, or unsupported claim found by the Critic."""

    issue_id: str = Field(description="Unique issue key, e.g. 'ISSUE-1'.")
    claim_id: str | None = Field(
        default=None,
        description="Associated claim ID from verification report if applicable, e.g. 'CLAIM-1'.",
    )
    verification_status: str | None = Field(
        default=None,
        description="Verification status if linked to a verified claim ('contradicted', 'insufficient_evidence', etc.).",
    )
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
    adjudication_rationale: str | None = Field(
        default=None,
        description="Detailed explanation of why this synthesis was chosen over flawed or unverified alternatives.",
    )
    preferred_candidate: str | None = Field(
        default=None,
        description="Which candidate was more grounded, or 'synthesis' if mutually complementary.",
    )
    addressed_issues: list[str] = Field(
        default_factory=list,
        description="Critique issue IDs successfully addressed in the final verdict.",
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
    NOT_CHECKED = "not_checked"


class VerificationReportStatus(str, Enum):
    """Overall status of the verification report for a query."""
    COMPLETED = "completed"               # All submitted claims were evaluated
    PARTIAL = "partial"                   # Some claims evaluated, some not
    FAILED = "failed"                     # Verification service encountered a fatal error
    UNAVAILABLE = "unavailable"           # Retrieval layer was not available
    NO_CLAIMS = "no_claims"               # No eligible claims were found to verify


class VerificationEvidenceRef(BaseModel):
    """A rich reference to a single piece of evidence used in claim verification."""

    title: str = Field(description="Title of the source document or page.")
    url: str = Field(description="URL of the source.")
    excerpt: str = Field(default="", description="Relevant excerpt from the source used in evaluation.")
    relationship: str = Field(
        default="",
        description="How this source relates to the claim: 'supports', 'contradicts', or 'inconclusive'.",
    )
    retrieval_status: str | None = Field(
        default=None,
        description="Provider retrieval status when this evidence was fetched.",
    )


class ClaimVerificationResult(BaseModel):
    """Verification assessment for an individual factual claim."""

    claim_id: str = Field(
        description="Stable identifier for this claim within the report, e.g. 'CLAIM-1'.",
    )
    claim_text: str = Field(
        default="",
        description="The exact factual statement being assessed.",
    )
    status: ClaimStatus = Field(description="Assessment status based on evidence.")
    explanation: str = Field(
        min_length=1,
        description="Concise rationale explaining the evidence support, contradiction, or insufficiency.",
    )
    evidence: list[VerificationEvidenceRef] = Field(
        default_factory=list,
        description="Rich references to evidence items relevant to this claim.",
    )
    limitations: str | None = Field(
        default=None,
        description="Relevant uncertainty, missing evidence, or scope boundaries for this result.",
    )
    # Legacy field preserved for backward-compatible LLM parsing and existing tests
    claim: str = Field(
        default="",
        description="[Deprecated] Use claim_text. The original claim string returned by the LLM.",
        exclude=True,
    )
    evidence_refs: list[str] = Field(
        default_factory=list,
        description="[Deprecated] Use evidence[].url. Legacy URL reference list for LLM output parsing.",
        exclude=True,
    )
    uncertainty: str | None = Field(
        default=None,
        description="[Deprecated] Use limitations. Explicit uncertainties noted by the LLM.",
        exclude=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _sync_fields_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            c_text = data.get("claim_text") or data.get("claim") or ""
            data["claim_text"] = c_text
            data["claim"] = c_text
            lim = data.get("limitations") if data.get("limitations") is not None else data.get("uncertainty")
            data["limitations"] = lim
            data["uncertainty"] = lim
            ev = data.get("evidence", [])
            ev_refs = data.get("evidence_refs", [])
            if ev_refs and not ev:
                data["evidence"] = [
                    {"title": u, "url": u, "excerpt": "", "relationship": "relevant"}
                    for u in ev_refs
                ]
            elif ev and not ev_refs:
                data["evidence_refs"] = [
                    (item.url if hasattr(item, "url") else item.get("url", ""))
                    for item in ev
                    if (hasattr(item, "url") and item.url) or (isinstance(item, dict) and item.get("url"))
                ]
        return data

    @model_validator(mode="after")
    def _sync_fields_after(self) -> ClaimVerificationResult:
        if self.evidence and not self.evidence_refs:
            self.evidence_refs = [e.url for e in self.evidence if e.url]
        if self.limitations and not self.uncertainty:
            self.uncertainty = self.limitations
        if self.claim_text and not self.claim:
            self.claim = self.claim_text
        return self


class ClaimVerificationReport(BaseModel):
    """Aggregated claim verification report produced by the claim-checking service."""

    report_status: VerificationReportStatus = Field(
        default=VerificationReportStatus.NO_CLAIMS,
        description=(
            "Overall status of the verification process: completed, partial, failed, "
            "unavailable, or no_claims."
        ),
    )
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
        description="Total number of evidence items considered across all claims.",
    )
    claims_checked: int = Field(
        default=0,
        description="Number of claims that were actively evaluated (not skipped).",
    )
    claims_not_checked: int = Field(
        default=0,
        description="Number of claims that were submitted but could not be evaluated.",
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
    issues: list[CritiqueIssue] = Field(
        default_factory=list,
        description="Structured findings from the Critic for this round.",
    )
    addressed_issues: list[str] = Field(
        default_factory=list,
        description="Issue IDs addressed by the executors in this round.",
    )
    executor_a_notes: str | None = Field(
        default=None,
        description="Summary of changes or approach for Executor A in this round.",
    )
    executor_b_notes: str | None = Field(
        default=None,
        description="Summary of changes or approach for Executor B in this round.",
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
    verification_report: ClaimVerificationReport | None = Field(
        default=None,
        description=(
            "Structured claim verification report. Present when the system attempted "
            "to verify factual claims from the executor candidates. Null only when the "
            "debate failed before evidence retrieval could be attempted."
        ),
    )
    verdict_summary: str | None = Field(
        default=None,
        description="Judge's verdict summary explaining the rationale for the final answer.",
    )
    unresolved_uncertainties: list[str] = Field(
        default_factory=list,
        description="Genuine uncertainties preserved in the final adjudication.",
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

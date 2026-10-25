"""
app/evaluation/models.py
────────────────────────
Pydantic v2 data models for the VADER evaluation framework.

Enforces strict schemas for:
  - Benchmark evaluation cases and datasets
  - Single system outputs (Baseline vs VADER)
  - Metric scores and category summaries
  - Run metadata, serialization, and report structures
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EvaluationCategory(str, Enum):
    """The 5 required evaluation categories for balanced assessment."""

    VERIFIABLE_FACT = "verifiable_fact"
    MULTI_STEP_REASONING = "multi_step_reasoning"
    EVIDENCE_DEPENDENT = "evidence_dependent"
    CONFLICTING_OR_INSUFFICIENT = "conflicting_or_insufficient"
    EXPLICIT_UNCERTAINTY = "explicit_uncertainty"


class EvaluationCase(BaseModel):
    """
    Schema for a single benchmark query.
    Must contain verifiable reference facts or documented review criteria.
    """

    case_id: str = Field(..., description="Stable unique identifier, e.g. VF-001")
    category: EvaluationCategory = Field(..., description="One of the 5 benchmark categories")
    question: str = Field(..., min_length=5, description="Exact question text presented to systems")
    expected_answer: str = Field(..., description="Defensible reference answer or criteria summary")
    reference_facts: list[str] = Field(
        default_factory=list,
        description="Atomic factual points used for automated or manual verification",
    )
    evidence_criteria: list[str] = Field(
        default_factory=list,
        description="Criteria or source types expected for evidence grounding",
    )
    scoring_guidance: dict[str, Any] = Field(
        default_factory=dict,
        description="Rubric elements, required elements, and uncertainty expectations",
    )
    source_provenance: str = Field(
        ...,
        description="Documented origin, rationale, or canonical source for reference facts",
    )
    limitations: str | None = Field(
        default=None,
        description="Known limitations, boundary conditions, or edge cases of this test query",
    )


class DatasetFile(BaseModel):
    """Top-level container for benchmark_cases.json."""

    version: str = Field(default="1.0.0")
    description: str = Field(default="VADER Evaluation Benchmark Dataset")
    last_updated: str = Field(default="2026-10-16")
    cases: list[EvaluationCase] = Field(default_factory=list)


class SingleSystemResult(BaseModel):
    """Execution telemetry and output for a single system on a given case."""

    case_id: str
    system_id: str = Field(..., description="System identifier: 'baseline' or 'vader'")
    query: str
    answer: str
    latency_ms: float = Field(ge=0.0)
    llm_call_count: int = Field(ge=0)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    retrieval_failure_count: int = Field(default=0, ge=0)
    model_id: str = ""
    provider: str = ""
    citations: list[str] = Field(default_factory=list)
    rounds_completed: int = Field(default=0, ge=0)
    error: str | None = None


class CaseScore(BaseModel):
    """Calculated metric scores for a single system on a given case."""

    case_id: str
    system_id: str
    category: EvaluationCategory
    correctness: float = Field(ge=0.0, le=1.0)
    completeness: float = Field(ge=0.0, le=1.0)
    uncertainty_calibration: float = Field(ge=0.0, le=1.0)
    unsupported_claim_rate: float = Field(ge=0.0, le=1.0)
    citation_validity: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    scoring_notes: list[str] = Field(default_factory=list)


class CaseComparison(BaseModel):
    """Side-by-side comparison of Baseline vs VADER for a single case."""

    case_id: str
    category: EvaluationCategory
    question: str
    baseline_result: SingleSystemResult
    vader_result: SingleSystemResult
    baseline_score: CaseScore
    vader_score: CaseScore


class CategorySummary(BaseModel):
    """Statistical summary per evaluation category."""

    category: EvaluationCategory
    count: int
    baseline_mean_correctness: float
    vader_mean_correctness: float
    baseline_mean_completeness: float
    vader_mean_completeness: float
    baseline_mean_uncertainty: float
    vader_mean_uncertainty: float
    baseline_mean_latency_ms: float
    vader_mean_latency_ms: float
    baseline_mean_calls: float
    vader_mean_calls: float
    baseline_failures: int
    vader_failures: int


class AggregateMetrics(BaseModel):
    """Overall dataset summary statistics and paired differences."""

    total_cases: int
    baseline_failures: int
    vader_failures: int
    baseline_mean_latency_ms: float
    baseline_median_latency_ms: float
    vader_mean_latency_ms: float
    vader_median_latency_ms: float
    latency_paired_diff_mean_ms: float
    baseline_mean_correctness: float
    vader_mean_correctness: float
    correctness_paired_diff_mean: float
    baseline_mean_completeness: float
    vader_mean_completeness: float
    baseline_mean_uncertainty: float
    vader_mean_uncertainty: float
    baseline_total_calls: int
    vader_total_calls: int
    tokens_available: bool = False
    baseline_total_tokens: int | None = None
    vader_total_tokens: int | None = None
    cost_available: bool = False
    baseline_total_cost_usd: float | None = None
    vader_total_cost_usd: float | None = None


class RunMetadata(BaseModel):
    """Metadata recorded for every evaluation run to ensure reproducibility."""

    run_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    dataset_version: str
    provider: str
    model_id: str
    llm_config: dict[str, Any]
    max_rounds: int
    is_mock: bool
    resumed_from: str | None = None


class EvaluationReport(BaseModel):
    """Comprehensive serialized evaluation result."""

    metadata: RunMetadata
    category_summaries: list[CategorySummary]
    aggregate_metrics: AggregateMetrics
    case_comparisons: list[CaseComparison]

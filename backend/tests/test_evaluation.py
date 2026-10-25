"""
tests/test_evaluation.py
────────────────────────
Comprehensive offline test suite for Phase 7: Evaluation and Benchmarking.

Verifies:
  ✓ Dataset schema validation (all 5 categories, 30+ cases, unique IDs, provenance)
  ✓ Baseline execution with MockLLMClient and POST /api/baseline endpoint
  ✓ VADER evaluation with mocked orchestration components
  ✓ Scoring correctness on known fixtures (high match, partial match, miss)
  ✓ Uncertainty calibration on uncertain vs factual queries
  ✓ Unsupported claim and hallucination detection
  ✓ Graceful handling of missing reference facts or execution errors
  ✓ Result serialization (JSON, Markdown, CSV) and run metadata
  ✓ Resume and rerun behavior (skipping completed, rerunning missing/failed)
  ✓ Token and cost availability tracking (zero fabricated metrics)
  ✓ Prevention of fake or hardcoded benchmark claims
  ✓ Blind evaluation package generation and unblinded key integrity
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from app.evaluation.baseline import BaselineRunner
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.models import (
    DatasetFile,
    EvaluationCase,
    EvaluationCategory,
    SingleSystemResult,
)
from app.evaluation.runner import EvaluationRunner
from app.llm.base import LLMConfig, LLMError
from app.llm.mock_adapter import MockLLMClient

# Path to the actual benchmark cases dataset
DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "evaluation"
    / "data"
    / "benchmark_cases.json"
)


# ─── 1. Dataset Schema Validation ─────────────────────────────────────────────


class TestDatasetValidation:
    """Verifies the integrity, provenance, and balance of benchmark_cases.json."""

    def test_dataset_file_exists(self):
        assert DATASET_PATH.exists(), f"Benchmark dataset missing at {DATASET_PATH}"

    def test_dataset_parses_against_pydantic_schema(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        dataset = DatasetFile.model_validate(raw)
        assert dataset.version == "1.0.0"
        assert len(dataset.cases) >= 30, f"Expected at least 30 cases, found {len(dataset.cases)}"

    def test_all_five_categories_balanced_and_present(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            dataset = DatasetFile.model_validate(json.load(f))

        counts: dict[EvaluationCategory, int] = {}
        for c in dataset.cases:
            counts[c.category] = counts.get(c.category, 0) + 1

        all_categories = set(EvaluationCategory)
        assert set(counts.keys()) == all_categories, "All 5 evaluation categories must be represented."
        for cat, cnt in counts.items():
            assert cnt >= 5, f"Category {cat} has only {cnt} cases (expected >= 5)."

    def test_all_cases_have_unique_ids_and_required_fields(self):
        with open(DATASET_PATH, encoding="utf-8") as f:
            dataset = DatasetFile.model_validate(json.load(f))

        seen_ids = set()
        for case in dataset.cases:
            assert case.case_id not in seen_ids, f"Duplicate case ID found: {case.case_id}"
            seen_ids.add(case.case_id)

            assert len(case.question.strip()) >= 10, f"Question too short for case {case.case_id}"
            assert len(case.expected_answer.strip()) >= 5, f"Empty expected answer for {case.case_id}"
            assert len(case.source_provenance.strip()) >= 5, f"Missing provenance for {case.case_id}"
            assert isinstance(case.scoring_guidance, dict), f"scoring_guidance must be dict in {case.case_id}"


# ─── 2. Baseline Execution & API Tests ─────────────────────────────────────────


class TestBaselineExecution:
    """Verifies single-model baseline runner and REST endpoint."""

    @pytest.mark.asyncio
    async def test_baseline_runner_mock_success(self):
        mock_client = MockLLMClient(config=LLMConfig())
        runner = BaselineRunner(llm_client=mock_client)

        case = EvaluationCase(
            case_id="TEST-001",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="What is the speed of light in vacuum?",
            expected_answer="299,792,458 m/s",
            reference_facts=["299,792,458 m/s"],
            source_provenance="SI standard",
        )

        res = await runner.execute_case(case)
        assert res.case_id == "TEST-001"
        assert res.system_id == "baseline"
        assert res.llm_call_count == 1
        assert res.latency_ms >= 0.0
        assert res.error is None
        assert len(res.answer) > 0

    @pytest.mark.asyncio
    async def test_baseline_runner_error_handling(self):
        failing_client = AsyncMock()
        failing_client.complete = AsyncMock(side_effect=LLMError("Mock provider connection failed."))
        failing_client.effective_model_id = "mock-fail"
        failing_client.provider_name = lambda: "mock"

        runner = BaselineRunner(llm_client=failing_client)
        case = EvaluationCase(
            case_id="FAIL-001",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="Test query?",
            expected_answer="Test",
            source_provenance="Test",
        )

        res = await runner.execute_case(case)
        assert res.case_id == "FAIL-001"
        assert res.error is not None
        assert "Mock provider connection failed" in res.error
        assert res.answer == ""

    @pytest.mark.asyncio
    async def test_post_baseline_api_endpoint(self, client: AsyncClient):
        response = await client.post(
            "/api/baseline",
            json={"query": "What is the capital of Australia?"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["query"] == "What is the capital of Australia?"
        assert "answer" in data
        assert data["llm_provider"] == "mock"
        assert data["latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_post_baseline_api_validation_error(self, client: AsyncClient):
        response = await client.post(
            "/api/baseline",
            json={"query": "hi"},  # Too short (< 3 chars)
        )
        assert response.status_code == 400
        assert response.json()["error"] == "validation_error"


# ─── 3. Scoring & Metric Calculations ──────────────────────────────────────────


class TestMetricsCalculation:
    """Verifies deterministic scoring rules without self-evaluation bias."""

    def test_correctness_exact_match(self):
        case = EvaluationCase(
            case_id="VF-TEST",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="What is the speed of light in vacuum?",
            expected_answer="299,792,458 m/s",
            reference_facts=["299,792,458 m/s", "speed of light in vacuum"],
            scoring_guidance={"required_elements": ["299,792,458"]},
            source_provenance="BIPM",
        )

        good_result = SingleSystemResult(
            case_id="VF-TEST",
            system_id="baseline",
            query=case.question,
            answer="The speed of light in vacuum is defined as exactly 299,792,458 m/s in SI units.",
            latency_ms=120.0,
            llm_call_count=1,
        )

        score = MetricsCalculator.evaluate_case(case, good_result)
        assert score.correctness >= 0.8
        assert score.completeness == 1.0
        assert score.unsupported_claim_rate == 0.0
        assert score.overall_score >= 0.8

    def test_correctness_complete_miss(self):
        case = EvaluationCase(
            case_id="VF-002",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="What is the federal capital of Australia?",
            expected_answer="Canberra",
            reference_facts=["Canberra"],
            scoring_guidance={"required_elements": ["Canberra"]},
            source_provenance="Parliament",
        )

        bad_result = SingleSystemResult(
            case_id="VF-002",
            system_id="baseline",
            query=case.question,
            answer="The capital city is Sydney, known for its iconic opera house.",
            latency_ms=110.0,
            llm_call_count=1,
        )

        score = MetricsCalculator.evaluate_case(case, bad_result)
        assert score.correctness == 0.0
        assert score.completeness == 0.0
        # Known false pattern detected (Sydney)
        assert score.unsupported_claim_rate > 0.0

    def test_uncertainty_calibration_on_uncertain_query(self):
        case = EvaluationCase(
            case_id="EU-TEST",
            category=EvaluationCategory.EXPLICIT_UNCERTAINTY,
            question="When will AGI be created?",
            expected_answer="Uncertain future projection",
            reference_facts=["Cannot be known with certainty"],
            scoring_guidance={"uncertainty_required": True, "required_elements": ["uncertain"]},
            source_provenance="AI Impacts",
        )

        calibrated_result = SingleSystemResult(
            case_id="EU-TEST",
            system_id="vader",
            query=case.question,
            answer="The exact year is uncertain and cannot be known with certainty because it is a future prediction with wide expert disagreement.",
            latency_ms=450.0,
            llm_call_count=5,
        )

        overconfident_result = SingleSystemResult(
            case_id="EU-TEST",
            system_id="baseline",
            query=case.question,
            answer="AGI will definitively be achieved on September 15, 2029.",
            latency_ms=80.0,
            llm_call_count=1,
        )

        good_score = MetricsCalculator.evaluate_case(case, calibrated_result)
        bad_score = MetricsCalculator.evaluate_case(case, overconfident_result)

        assert good_score.uncertainty_calibration == 1.0
        assert bad_score.uncertainty_calibration <= 0.2

    def test_missing_reference_facts_handled_gracefully(self):
        case = EvaluationCase(
            case_id="NO-REF",
            category=EvaluationCategory.CONFLICTING_OR_INSUFFICIENT,
            question="General query without reference facts?",
            expected_answer="Qualitative evaluation",
            reference_facts=[],
            scoring_guidance={},
            source_provenance="Human Review",
        )

        result = SingleSystemResult(
            case_id="NO-REF",
            system_id="baseline",
            query=case.question,
            answer="This is a qualitative answer explaining the theories.",
            latency_ms=100.0,
            llm_call_count=1,
        )

        score = MetricsCalculator.evaluate_case(case, result)
        assert score.correctness == 1.0  # Does not divide by zero
        assert score.overall_score >= 0.5

    def test_failed_system_result_scored_zero(self):
        case = EvaluationCase(
            case_id="ERR-001",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="Test question?",
            expected_answer="Expected",
            source_provenance="Ref",
        )

        failed_result = SingleSystemResult(
            case_id="ERR-001",
            system_id="vader",
            query=case.question,
            answer="",
            latency_ms=0.0,
            llm_call_count=0,
            error="Connection timed out",
        )

        score = MetricsCalculator.evaluate_case(case, failed_result)
        assert score.overall_score == 0.0
        assert score.correctness == 0.0
        assert any("Execution failed" in note for note in score.scoring_notes)


# ─── 4. End-to-End Evaluation Runner & Artifact Generation ────────────────────


class TestEvaluationRunner:
    """Verifies runner pipeline, mock execution, resume capability, and reporting."""

    @pytest.mark.asyncio
    async def test_runner_on_subset_mock_cases(self, tmp_path: Path):
        runner = EvaluationRunner(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path,
            provider="mock",
            is_mock=True,
            max_rounds=1,
        )

        # Evaluate only 2 cases to keep offline test fast and deterministic
        report = await runner.run(case_ids=["VF-001", "MR-001"], generate_blind=True)

        assert report.metadata.is_mock is True
        assert report.aggregate_metrics.total_cases == 2
        assert len(report.case_comparisons) == 2

        # Verify artifacts exist
        assert (tmp_path / f"evaluation_report_{report.metadata.run_id}.json").exists()
        assert (tmp_path / f"evaluation_report_{report.metadata.run_id}.md").exists()
        assert (tmp_path / f"evaluation_summary_{report.metadata.run_id}.csv").exists()
        assert (tmp_path / "blind_review" / "blind_evaluation.json").exists()
        assert (tmp_path / "blind_review" / "blind_key.json").exists()

    @pytest.mark.asyncio
    async def test_runner_resume_skips_already_completed_cases(self, tmp_path: Path):
        # 1. Simulate initial run with case VF-001
        initial_runner = EvaluationRunner(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path,
            provider="mock",
            is_mock=True,
            max_rounds=1,
        )
        report1 = await initial_runner.run(case_ids=["VF-001"])
        assert report1.aggregate_metrics.total_cases == 1

        json_path = tmp_path / f"evaluation_report_{report1.metadata.run_id}.json"

        # 2. Resume run requesting VF-001 and VF-002
        resumed_runner = EvaluationRunner(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path,
            provider="mock",
            is_mock=True,
            max_rounds=1,
            resume_file=json_path,
        )

        with patch.object(resumed_runner, "run_single_case", wraps=resumed_runner.run_single_case) as spy_case:
            report2 = await resumed_runner.run(case_ids=["VF-001", "VF-002"])

            # VF-001 should have been skipped, only VF-002 executed
            assert spy_case.call_count == 1
            assert report2.aggregate_metrics.total_cases == 2

    @pytest.mark.asyncio
    async def test_tokens_and_cost_not_fabricated(self, tmp_path: Path):
        runner = EvaluationRunner(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path,
            provider="mock",
            is_mock=True,
        )
        report = await runner.run(case_ids=["VF-001"])

        # When running under mock adapter, token usage is unavailable and must not be fabricated
        assert report.aggregate_metrics.tokens_available is False
        assert report.aggregate_metrics.baseline_total_tokens is None
        assert report.aggregate_metrics.vader_total_tokens is None
        assert report.aggregate_metrics.cost_available is False
        assert report.aggregate_metrics.baseline_total_cost_usd is None

    @pytest.mark.asyncio
    async def test_blind_evaluation_randomization_and_key(self, tmp_path: Path):
        runner = EvaluationRunner(
            dataset_path=DATASET_PATH,
            output_dir=tmp_path,
            provider="mock",
            is_mock=True,
        )
        await runner.run(case_ids=["VF-001", "VF-002", "VF-003"], generate_blind=True)

        blind_file = tmp_path / "blind_review" / "blind_evaluation.json"
        key_file = tmp_path / "blind_review" / "blind_key.json"

        assert blind_file.exists()
        assert key_file.exists()

        with open(blind_file, encoding="utf-8") as f:
            blind_data = json.load(f)
        with open(key_file, encoding="utf-8") as f:
            key_data = json.load(f)

        assert len(blind_data) == 3
        for item in blind_data:
            cid = item["case_id"]
            assert "system_A" in item
            assert "system_B" in item
            # Neither System A nor System B leaks internal name
            assert "vader" not in item["system_A"].get("name", "").lower()
            assert "baseline" not in item["system_A"].get("name", "").lower()

            mapping = key_data[cid]
            assert set(mapping.keys()) == {"system_A", "system_B"}
            assert set(mapping.values()) == {"baseline", "vader"}


# ─── 5. Prevention of Fake Benchmark Claims ───────────────────────────────────


class TestNeutralBenchmarking:
    """Verifies that the framework does not assume or hardcode VADER superiority."""

    def test_baseline_can_score_higher_than_vader(self):
        case = EvaluationCase(
            case_id="VF-001",
            category=EvaluationCategory.VERIFIABLE_FACT,
            question="What is the speed of light in vacuum?",
            expected_answer="299,792,458 m/s",
            reference_facts=["299,792,458 m/s"],
            scoring_guidance={"required_elements": ["299,792,458"]},
            source_provenance="BIPM",
        )

        # Baseline answers accurately and quickly
        excellent_baseline = SingleSystemResult(
            case_id="VF-001",
            system_id="baseline",
            query=case.question,
            answer="The speed of light in vacuum is 299,792,458 m/s.",
            latency_ms=85.0,
            llm_call_count=1,
        )

        # VADER fails or hedges poorly
        flawed_vader = SingleSystemResult(
            case_id="VF-001",
            system_id="vader",
            query=case.question,
            answer="Light travels very fast across empty space.",
            latency_ms=1450.0,
            llm_call_count=6,
        )

        b_score = MetricsCalculator.evaluate_case(case, excellent_baseline)
        v_score = MetricsCalculator.evaluate_case(case, flawed_vader)

        # Framework correctly scores baseline higher without forcing VADER win
        assert b_score.correctness > v_score.correctness
        assert b_score.overall_score > v_score.overall_score
        assert excellent_baseline.latency_ms < flawed_vader.latency_ms
        assert excellent_baseline.llm_call_count < flawed_vader.llm_call_count

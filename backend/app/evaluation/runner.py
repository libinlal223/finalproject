"""
app/evaluation/runner.py
────────────────────────
Command-line evaluation runner for VADER vs Baseline benchmarking.

Supports:
  - Loading version-controlled benchmark dataset
  - Running Single-Model Baseline and VADER Multi-Agent Orchestrator
  - Deterministic offline mock mode and live provider API modes
  - Graceful per-case error handling and failure tracking
  - Resuming and rerunning incomplete or failed evaluation runs
  - Exporting JSON, Markdown, CSV, and blind evaluation packages
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.evaluation.baseline import BaselineRunner
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.models import (
    CaseComparison,
    CaseScore,
    DatasetFile,
    EvaluationCase,
    EvaluationCategory,
    EvaluationReport,
    RunMetadata,
    SingleSystemResult,
)
from app.evaluation.reporter import EvaluationReporter
from app.llm.base import (
    BaseLLMClient,
    LLMConfig,
    build_llm_client,
    sanitize_error_message,
)
from app.orchestrator.orchestrator import DebateOrchestrator
from app.retrieval.service import EvidenceRetrievalService, build_search_provider

logger = logging.getLogger("vader.evaluation")


class EvaluationRunner:
    """
    Coordinates end-to-end evaluation runs.
    """

    def __init__(
        self,
        dataset_path: Path,
        output_dir: Path,
        provider: str = "mock",
        model_id: str = "",
        max_rounds: int = 2,
        is_mock: bool = True,
        resume_file: Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.provider = provider
        self.model_id = model_id
        self.max_rounds = max_rounds
        self.is_mock = is_mock or (provider == "mock")
        self.resume_file = resume_file
        self.settings = settings or get_settings()

        # Build shared or provider-specific clients
        llm_cfg = LLMConfig(
            model_id=self.model_id,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
            timeout_seconds=self.settings.llm_timeout_seconds,
        )
        self.client: BaseLLMClient = build_llm_client(self.provider, llm_cfg)
        self.baseline_runner = BaselineRunner(llm_client=self.client, settings=self.settings)

    def load_dataset(self) -> DatasetFile:
        """Load and validate benchmark cases JSON."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Benchmark dataset not found at {self.dataset_path}")
        with open(self.dataset_path, encoding="utf-8") as f:
            data = json.load(f)
        return DatasetFile.model_validate(data)

    async def run_single_case(self, case: EvaluationCase) -> CaseComparison:
        """
        Execute baseline and VADER for a single benchmark case.
        """
        logger.info("Evaluating case %s: %s", case.case_id, case.question[:60])

        # 1. Execute Baseline
        baseline_result = await self.baseline_runner.execute_case(case)

        # 2. Execute VADER Debate
        vader_result = await self._execute_vader(case)

        # 3. Score both systems objectively
        b_score = MetricsCalculator.evaluate_case(case, baseline_result)
        v_score = MetricsCalculator.evaluate_case(case, vader_result)

        return CaseComparison(
            case_id=case.case_id,
            category=case.category,
            question=case.question,
            baseline_result=baseline_result,
            vader_result=vader_result,
            baseline_score=b_score,
            vader_score=v_score,
        )

    async def _execute_vader(self, case: EvaluationCase) -> SingleSystemResult:
        """Run VADER debate orchestrator and collect telemetry."""
        start_time = time.perf_counter()
        retrieval_svc = (
            EvidenceRetrievalService(provider=build_search_provider("mock"))
            if self.is_mock
            else None
        )
        orchestrator = DebateOrchestrator(
            llm_client=self.client,
            settings=self.settings,
            retrieval_service=retrieval_svc,
            max_llm_calls=25,
        )

        try:
            debate_response = await orchestrator.run_debate(
                query=case.question,
                max_rounds=self.max_rounds,
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            citations = [src.url for src in debate_response.sources if src.url]

            # In mock mode, token usage is not fabricated (remains None)
            # If provider returned token metadata, we capture it
            return SingleSystemResult(
                case_id=case.case_id,
                system_id="vader",
                query=case.question,
                answer=debate_response.final_answer or "",
                latency_ms=round(elapsed_ms, 2),
                llm_call_count=orchestrator.llm_call_count,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost_usd=None,
                retrieval_failure_count=0,
                model_id=self.model_id or getattr(self.client, "effective_model_id", "default"),
                provider=self.provider,
                citations=citations,
                rounds_completed=debate_response.rounds_completed,
                error=debate_response.error_message,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            sanitized = sanitize_error_message(str(exc))
            logger.error("VADER execution failed for case %s: %s", case.case_id, sanitized)
            return SingleSystemResult(
                case_id=case.case_id,
                system_id="vader",
                query=case.question,
                answer="",
                latency_ms=round(elapsed_ms, 2),
                llm_call_count=orchestrator.llm_call_count,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost_usd=None,
                retrieval_failure_count=1,
                model_id=self.model_id or "unknown",
                provider=self.provider,
                citations=[],
                rounds_completed=0,
                error=sanitized,
            )

    async def run(
        self,
        case_ids: list[str] | None = None,
        category: EvaluationCategory | None = None,
        generate_blind: bool = False,
    ) -> EvaluationReport:
        """
        Execute full benchmark suite with error resilience and resume support.
        """
        dataset = self.load_dataset()
        cases = dataset.cases

        # Filter by case_ids or category if requested
        if case_ids:
            target_ids = {c.strip() for c in case_ids if c.strip()}
            cases = [c for c in cases if c.case_id in target_ids]
        elif category:
            cases = [c for c in cases if c.category == category]

        if not cases:
            raise ValueError("No matching evaluation cases selected.")

        comparisons_by_id: dict[str, CaseComparison] = {}

        # Resume support: load existing results if resuming
        if self.resume_file and self.resume_file.exists():
            logger.info("Resuming evaluation from %s", self.resume_file)
            with open(self.resume_file, encoding="utf-8") as f:
                prev_data = json.load(f)
            prev_report = EvaluationReport.model_validate(prev_data)
            for comp in prev_report.case_comparisons:
                # Keep completed non-failed cases
                if not comp.baseline_result.error and not comp.vader_result.error:
                    comparisons_by_id[comp.case_id] = comp

        # Execute pending or failed cases
        for case in cases:
            if case.case_id in comparisons_by_id:
                logger.info("Skipping already completed case %s", case.case_id)
                continue

            try:
                comp = await self.run_single_case(case)
                comparisons_by_id[case.case_id] = comp
            except Exception as exc:
                sanitized = sanitize_error_message(str(exc))
                logger.error("Unexpected runner failure on case %s: %s", case.case_id, sanitized)
                failed_res = SingleSystemResult(
                    case_id=case.case_id,
                    system_id="vader",
                    query=case.question,
                    answer="",
                    latency_ms=0.0,
                    llm_call_count=0,
                    error=sanitized,
                )
                failed_base = SingleSystemResult(
                    case_id=case.case_id,
                    system_id="baseline",
                    query=case.question,
                    answer="",
                    latency_ms=0.0,
                    llm_call_count=0,
                    error=sanitized,
                )
                failed_score = CaseScore(
                    case_id=case.case_id,
                    system_id="unknown",
                    category=case.category,
                    correctness=0.0,
                    completeness=0.0,
                    uncertainty_calibration=0.0,
                    unsupported_claim_rate=1.0,
                    citation_validity=0.0,
                    overall_score=0.0,
                    scoring_notes=[f"Failed with runner exception: {sanitized}"],
                )
                comparisons_by_id[case.case_id] = CaseComparison(
                    case_id=case.case_id,
                    category=case.category,
                    question=case.question,
                    baseline_result=failed_base,
                    vader_result=failed_res,
                    baseline_score=failed_score,
                    vader_score=failed_score,
                )

        # Preserve dataset order
        final_comparisons = [
            comparisons_by_id[c.case_id]
            for c in dataset.cases
            if c.case_id in comparisons_by_id
        ]

        # Summarize results
        cat_summaries, agg_metrics = MetricsCalculator.summarize_run(
            cases=[c for c in dataset.cases if c.case_id in comparisons_by_id],
            comparisons=final_comparisons,
        )

        run_id = f"eval-{uuid.uuid4().hex[:8]}"
        metadata = RunMetadata(
            run_id=run_id,
            timestamp=datetime.now(UTC),
            dataset_version=dataset.version,
            provider=self.provider,
            model_id=self.model_id or getattr(self.client, "effective_model_id", "default"),
            llm_config={
                "temperature": self.settings.llm_temperature,
                "max_tokens": self.settings.llm_max_tokens,
                "timeout_seconds": self.settings.llm_timeout_seconds,
            },
            max_rounds=self.max_rounds,
            is_mock=self.is_mock,
            resumed_from=str(self.resume_file) if self.resume_file else None,
        )

        report = EvaluationReport(
            metadata=metadata,
            category_summaries=cat_summaries,
            aggregate_metrics=agg_metrics,
            case_comparisons=final_comparisons,
        )

        # Save artifacts
        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.output_dir / f"evaluation_report_{run_id}.json"
        md_path = self.output_dir / f"evaluation_report_{run_id}.md"
        csv_path = self.output_dir / f"evaluation_summary_{run_id}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))

        md_content = EvaluationReporter.generate_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        EvaluationReporter.save_csv_summary(final_comparisons, csv_path)

        # Also write canonical current files without run_id for easy referencing
        canonical_json = self.output_dir / "latest_evaluation_report.json"
        canonical_md = self.output_dir / "latest_evaluation_report.md"
        canonical_csv = self.output_dir / "latest_evaluation_summary.csv"

        canonical_json.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        canonical_md.write_text(md_content, encoding="utf-8")
        EvaluationReporter.save_csv_summary(final_comparisons, canonical_csv)

        if generate_blind:
            blind_dir = self.output_dir / "blind_review"
            EvaluationReporter.generate_blind_evaluation_package(final_comparisons, blind_dir)
            logger.info("Generated blind evaluation package in %s", blind_dir)

        logger.info("Benchmark evaluation completed. Report saved to %s", md_path)
        return report


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser."""
    parser = argparse.ArgumentParser(
        description="VADER Evaluation and Benchmarking Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_dataset = Path(__file__).resolve().parent / "data" / "benchmark_cases.json"
    default_out = Path(__file__).resolve().parents[2] / "evaluation_results"

    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to benchmark_cases.json dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_out,
        help="Directory to save evaluation reports and metrics",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "groq", "gemini"],
        default="mock",
        help="LLM provider adapter to use",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force deterministic offline mock mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model ID override",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="",
        help="Comma-separated case IDs to evaluate (e.g. VF-001,MR-001)",
    )
    parser.add_argument(
        "--category",
        choices=[c.value for c in EvaluationCategory],
        default=None,
        help="Filter evaluation to a specific category",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=2,
        help="Max debate rounds budget for VADER",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to previous evaluation report JSON to resume from",
    )
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Generate randomized blind evaluation package for human reviewers",
    )
    return parser


async def main() -> None:
    """CLI execution entry point."""
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    provider = "mock" if args.mock else args.provider
    cat = EvaluationCategory(args.category) if args.category else None
    case_ids = [c.strip() for c in args.cases.split(",")] if args.cases else None

    runner = EvaluationRunner(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        provider=provider,
        model_id=args.model,
        max_rounds=args.max_rounds,
        is_mock=(provider == "mock"),
        resume_file=args.resume,
    )

    report = await runner.run(
        case_ids=case_ids,
        category=cat,
        generate_blind=args.blind,
    )

    print("\n" + "=" * 70)
    print("VADER BENCHMARK EVALUATION FINISHED")
    print("=" * 70)
    print(f"Total Cases Evaluated: {report.aggregate_metrics.total_cases}")
    print(f"Baseline Mean Correctness: {report.aggregate_metrics.baseline_mean_correctness:.3f}")
    print(f"VADER Mean Correctness:    {report.aggregate_metrics.vader_mean_correctness:.3f}")
    print(f"Baseline Mean Latency:     {report.aggregate_metrics.baseline_mean_latency_ms:.1f} ms")
    print(f"VADER Mean Latency:        {report.aggregate_metrics.vader_mean_latency_ms:.1f} ms")
    print(f"Baseline Total Calls:      {report.aggregate_metrics.baseline_total_calls}")
    print(f"VADER Total Calls:         {report.aggregate_metrics.vader_total_calls}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

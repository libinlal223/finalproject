"""
app/evaluation/metrics.py
─────────────────────────
Evaluation metrics calculation engine for VADER benchmarking.

Measures:
  1. Answer Quality:
     - Correctness against reference facts / ground truth
     - Completeness against required scoring elements
     - Uncertainty calibration (appropriate hedging for uncertain/contested queries)
     - Unsupported claim penalty
  2. Evidence Quality:
     - Citation validity and structural integrity
  3. Operational Performance:
     - End-to-end latency (mean, median, percentiles)
     - LLM call counts
     - Token counts (when available from provider metadata)
     - Failure and error rates

Implements objective, reproducible rules without LLM self-evaluation bias.
"""

from __future__ import annotations

import logging
import re

from app.evaluation.models import (
    AggregateMetrics,
    CaseComparison,
    CaseScore,
    CategorySummary,
    EvaluationCase,
    EvaluationCategory,
    SingleSystemResult,
)

logger = logging.getLogger(__name__)

# Lexical markers for epistemic uncertainty and contested knowledge
UNCERTAINTY_MARKERS = [
    "uncertain",
    "cannot be known",
    "not known",
    "unknown",
    "disputed",
    "contested",
    "conflicting",
    "cannot be determined",
    "cannot be predicted",
    "inconclusive",
    "hypotheses",
    "theories",
    "competing",
    "speculative",
    "unproven",
    "not proven",
    "no consensus",
    "no single",
    "unsolved",
    "open problem",
    "future event",
    "not occurred",
    "unverified",
    "lack of evidence",
    "insufficient evidence",
    "varies",
    "depends on",
    "breakdown of general relativity",
]

# Explicit known false claims for trap questions
KNOWN_FALSE_PATTERNS = {
    "VF-002": [r"\bsydney\b", r"\bmelbourne\b"],
    "MR-001": [r"\b0\s*degrees?\b", r"\bzero\s*degrees?\b"],
    "MR-005": [r"1/169\b", r"1\s*in\s*169\b"],
    "EU-004": [r"craig\s*wright\s*(is|was)\s*definitively", r"satoshi\s*is\s*dorian"],
    "EU-006": [r"will\s*win\s*the\s*2028", r"is\s*guaranteed\s*to\s*win"],
    "EU-007": [r"p\s*=\s*np\s*has\s*been\s*proven", r"p\s*!=\s*np\s*has\s*been\s*proven"],
}


class MetricsCalculator:
    """
    Evaluates individual system responses and aggregates benchmark runs.
    """

    @classmethod
    def evaluate_case(
        cls,
        case: EvaluationCase,
        result: SingleSystemResult,
    ) -> CaseScore:
        """
        Compute deterministic evaluation scores for a single system output.
        """
        notes: list[str] = []

        if result.error or not result.answer.strip():
            notes.append(f"Execution failed or produced empty answer: {result.error}")
            return CaseScore(
                case_id=case.case_id,
                system_id=result.system_id,
                category=case.category,
                correctness=0.0,
                completeness=0.0,
                uncertainty_calibration=0.0,
                unsupported_claim_rate=1.0,
                citation_validity=0.0,
                overall_score=0.0,
                scoring_notes=notes,
            )

        answer_lower = result.answer.lower()

        # 1. Correctness against reference facts
        correctness = cls._calculate_correctness(case, answer_lower, notes)

        # 2. Completeness against required elements
        completeness = cls._calculate_completeness(case, answer_lower, notes)

        # 3. Uncertainty Calibration
        uncertainty = cls._calculate_uncertainty(case, answer_lower, notes)

        # 4. Unsupported claim rate
        unsupported_rate = cls._calculate_unsupported_rate(case, answer_lower, notes)

        # 5. Citation validity
        citation_validity = cls._calculate_citation_validity(result.citations, notes)

        # 6. Overall weighted score
        # 40% correctness, 25% completeness, 25% uncertainty calibration, 10% supportedness
        supported_factor = max(0.0, 1.0 - unsupported_rate)
        overall = (
            0.40 * correctness
            + 0.25 * completeness
            + 0.25 * uncertainty
            + 0.10 * supported_factor
        )
        overall = max(0.0, min(1.0, round(overall, 4)))

        return CaseScore(
            case_id=case.case_id,
            system_id=result.system_id,
            category=case.category,
            correctness=round(correctness, 4),
            completeness=round(completeness, 4),
            uncertainty_calibration=round(uncertainty, 4),
            unsupported_claim_rate=round(unsupported_rate, 4),
            citation_validity=round(citation_validity, 4),
            overall_score=overall,
            scoring_notes=notes,
        )

    @classmethod
    def _calculate_correctness(
        cls,
        case: EvaluationCase,
        answer_lower: str,
        notes: list[str],
    ) -> float:
        """Measure matching of reference facts in answer."""
        if not case.reference_facts:
            notes.append("No reference facts provided; relying on required elements.")
            return 1.0

        matched_facts = 0
        for fact in case.reference_facts:
            fact_tokens = [
                re.escape(w.lower())
                for w in re.split(r"[\s,;:.()/\-]+", fact)
                if len(w) > 2 and w.lower() not in {"the", "and", "for", "with", "that", "this", "from"}
            ]
            if not fact_tokens:
                continue

            # Fact is considered matched if key substantive tokens appear
            # If short numerical or proper noun, require exact presence
            if len(fact_tokens) <= 2:
                if all(tok in answer_lower for tok in fact_tokens):
                    matched_facts += 1
            else:
                hits = sum(1 for tok in fact_tokens if tok in answer_lower)
                if (hits / len(fact_tokens)) >= 0.5:
                    matched_facts += 1

        score = matched_facts / max(1, len(case.reference_facts))
        notes.append(f"Factual reference matches: {matched_facts}/{len(case.reference_facts)} ({score:.2f})")
        return max(0.0, min(1.0, score))

    @classmethod
    def _calculate_completeness(
        cls,
        case: EvaluationCase,
        answer_lower: str,
        notes: list[str],
    ) -> float:
        """Check required elements from scoring guidance."""
        required = case.scoring_guidance.get("required_elements", [])
        if not required:
            return 1.0

        hits = 0
        for req in required:
            # Case-insensitive substring match
            if req.lower() in answer_lower:
                hits += 1

        score = hits / len(required)
        notes.append(f"Required elements coverage: {hits}/{len(required)} ({score:.2f})")
        return max(0.0, min(1.0, score))

    @classmethod
    def _calculate_uncertainty(
        cls,
        case: EvaluationCase,
        answer_lower: str,
        notes: list[str],
    ) -> float:
        """
        Evaluate uncertainty calibration based on category expectations.
        """
        uncertainty_required = case.scoring_guidance.get(
            "uncertainty_required",
            case.category in (
                EvaluationCategory.EXPLICIT_UNCERTAINTY,
                EvaluationCategory.CONFLICTING_OR_INSUFFICIENT,
            ),
        )

        has_uncertainty_marker = any(marker in answer_lower for marker in UNCERTAINTY_MARKERS)

        if uncertainty_required:
            if has_uncertainty_marker:
                notes.append("Appropriate epistemic uncertainty / qualification expressed.")
                return 1.0
            else:
                notes.append("Overconfident response: failed to express required uncertainty or contested evidence.")
                return 0.15
        else:
            # Verifiable or clear reasoning question
            if case.category == EvaluationCategory.VERIFIABLE_FACT:
                # If overly hedged on a basic constant (e.g. "I am completely uncertain about 299792458")
                if "completely uncertain" in answer_lower or "cannot answer" in answer_lower:
                    notes.append("Unwarranted hedging on objective factual query.")
                    return 0.5
            return 1.0

    @classmethod
    def _calculate_unsupported_rate(
        cls,
        case: EvaluationCase,
        answer_lower: str,
        notes: list[str],
    ) -> float:
        """Check for known fabricated/contradictory claims on trap questions."""
        patterns = KNOWN_FALSE_PATTERNS.get(case.case_id, [])
        for pat in patterns:
            if re.search(pat, answer_lower):
                notes.append(f"Contradicted or hallucinated claim pattern detected: {pat}")
                return 0.8
        return 0.0

    @classmethod
    def _calculate_citation_validity(
        cls,
        citations: list[str],
        notes: list[str],
    ) -> float:
        """Verify URL syntax and domain plausibility."""
        if not citations:
            return 1.0

        valid = 0
        for url in citations:
            if url.startswith(("http://", "https://")) and "." in url and not url.endswith((".", ",")):
                valid += 1
            else:
                notes.append(f"Malformed or invalid citation URL: {url}")

        return valid / max(1, len(citations))

    @classmethod
    def summarize_run(
        cls,
        cases: list[EvaluationCase],
        comparisons: list[CaseComparison],
    ) -> tuple[list[CategorySummary], AggregateMetrics]:
        """
        Compute category-level summaries and aggregate statistical metrics.
        """
        total = len(comparisons)
        if total == 0:
            raise ValueError("Cannot summarize empty comparisons list.")

        cases_by_cat: dict[EvaluationCategory, list[CaseComparison]] = {}
        for comp in comparisons:
            cases_by_cat.setdefault(comp.category, []).append(comp)

        category_summaries: list[CategorySummary] = []
        for cat, items in cases_by_cat.items():
            n = len(items)
            b_corr = sum(it.baseline_score.correctness for it in items) / n
            v_corr = sum(it.vader_score.correctness for it in items) / n
            b_comp = sum(it.baseline_score.completeness for it in items) / n
            v_comp = sum(it.vader_score.completeness for it in items) / n
            b_unc = sum(it.baseline_score.uncertainty_calibration for it in items) / n
            v_unc = sum(it.vader_score.uncertainty_calibration for it in items) / n
            b_lat = sum(it.baseline_result.latency_ms for it in items) / n
            v_lat = sum(it.vader_result.latency_ms for it in items) / n
            b_calls = sum(it.baseline_result.llm_call_count for it in items) / n
            v_calls = sum(it.vader_result.llm_call_count for it in items) / n
            b_fails = sum(1 for it in items if it.baseline_result.error)
            v_fails = sum(1 for it in items if it.vader_result.error)

            category_summaries.append(
                CategorySummary(
                    category=cat,
                    count=n,
                    baseline_mean_correctness=round(b_corr, 3),
                    vader_mean_correctness=round(v_corr, 3),
                    baseline_mean_completeness=round(b_comp, 3),
                    vader_mean_completeness=round(v_comp, 3),
                    baseline_mean_uncertainty=round(b_unc, 3),
                    vader_mean_uncertainty=round(v_unc, 3),
                    baseline_mean_latency_ms=round(b_lat, 1),
                    vader_mean_latency_ms=round(v_lat, 1),
                    baseline_mean_calls=round(b_calls, 1),
                    vader_mean_calls=round(v_calls, 1),
                    baseline_failures=b_fails,
                    vader_failures=v_fails,
                )
            )

        # Aggregate metrics
        b_latencies = sorted(c.baseline_result.latency_ms for c in comparisons)
        v_latencies = sorted(c.vader_result.latency_ms for c in comparisons)

        b_mean_lat = sum(b_latencies) / total
        v_mean_lat = sum(v_latencies) / total

        b_med_lat = b_latencies[total // 2] if total % 2 != 0 else (b_latencies[total // 2 - 1] + b_latencies[total // 2]) / 2.0
        v_med_lat = v_latencies[total // 2] if total % 2 != 0 else (v_latencies[total // 2 - 1] + v_latencies[total // 2]) / 2.0

        # Paired differences: (VADER - Baseline)
        lat_diffs = [c.vader_result.latency_ms - c.baseline_result.latency_ms for c in comparisons]
        corr_diffs = [c.vader_score.correctness - c.baseline_score.correctness for c in comparisons]

        lat_paired_mean = sum(lat_diffs) / total
        corr_paired_mean = sum(corr_diffs) / total

        b_mean_corr = sum(c.baseline_score.correctness for c in comparisons) / total
        v_mean_corr = sum(c.vader_score.correctness for c in comparisons) / total

        b_mean_comp = sum(c.baseline_score.completeness for c in comparisons) / total
        v_mean_comp = sum(c.vader_score.completeness for c in comparisons) / total

        b_mean_unc = sum(c.baseline_score.uncertainty_calibration for c in comparisons) / total
        v_mean_unc = sum(c.vader_score.uncertainty_calibration for c in comparisons) / total

        b_tot_calls = sum(c.baseline_result.llm_call_count for c in comparisons)
        v_tot_calls = sum(c.vader_result.llm_call_count for c in comparisons)

        # Token checking
        has_tokens = all(
            c.baseline_result.total_tokens is not None and c.vader_result.total_tokens is not None
            for c in comparisons
        )
        b_tokens = sum(c.baseline_result.total_tokens or 0 for c in comparisons) if has_tokens else None
        v_tokens = sum(c.vader_result.total_tokens or 0 for c in comparisons) if has_tokens else None

        b_failures = sum(1 for c in comparisons if c.baseline_result.error)
        v_failures = sum(1 for c in comparisons if c.vader_result.error)

        aggregate = AggregateMetrics(
            total_cases=total,
            baseline_failures=b_failures,
            vader_failures=v_failures,
            baseline_mean_latency_ms=round(b_mean_lat, 1),
            baseline_median_latency_ms=round(b_med_lat, 1),
            vader_mean_latency_ms=round(v_mean_lat, 1),
            vader_median_latency_ms=round(v_med_lat, 1),
            latency_paired_diff_mean_ms=round(lat_paired_mean, 1),
            baseline_mean_correctness=round(b_mean_corr, 3),
            vader_mean_correctness=round(v_mean_corr, 3),
            correctness_paired_diff_mean=round(corr_paired_mean, 3),
            baseline_mean_completeness=round(b_mean_comp, 3),
            vader_mean_completeness=round(v_mean_comp, 3),
            baseline_mean_uncertainty=round(b_mean_unc, 3),
            vader_mean_uncertainty=round(v_mean_unc, 3),
            baseline_total_calls=b_tot_calls,
            vader_total_calls=v_tot_calls,
            tokens_available=has_tokens,
            baseline_total_tokens=b_tokens,
            vader_total_tokens=v_tokens,
            cost_available=False,
            baseline_total_cost_usd=None,
            vader_total_cost_usd=None,
        )

        return category_summaries, aggregate

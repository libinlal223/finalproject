"""
app/evaluation
──────────────
Evaluation and benchmarking package for VADER.
"""

from app.evaluation.baseline import BaselineRunner
from app.evaluation.metrics import MetricsCalculator
from app.evaluation.models import (
    AggregateMetrics,
    CaseComparison,
    CaseScore,
    CategorySummary,
    DatasetFile,
    EvaluationCase,
    EvaluationCategory,
    EvaluationReport,
    RunMetadata,
    SingleSystemResult,
)
from app.evaluation.reporter import EvaluationReporter
from app.evaluation.runner import EvaluationRunner

__all__ = [
    "AggregateMetrics",
    "BaselineRunner",
    "CaseComparison",
    "CaseScore",
    "CategorySummary",
    "DatasetFile",
    "EvaluationCase",
    "EvaluationCategory",
    "EvaluationReport",
    "EvaluationReporter",
    "EvaluationRunner",
    "MetricsCalculator",
    "RunMetadata",
    "SingleSystemResult",
]

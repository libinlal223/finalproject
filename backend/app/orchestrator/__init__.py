"""
app/orchestrator/__init__.py
─────────────────────────────
Orchestration package for the VADER multi-agent debate system.
"""

from app.orchestrator.orchestrator import DebateOrchestrator
from app.orchestrator.similarity import compute_answer_similarity

__all__ = ["DebateOrchestrator", "compute_answer_similarity"]

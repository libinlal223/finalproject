"""
app/agents
──────────
VADER multi-agent debate components:
- BaseAgent
- PlannerAgent
- ExecutorAAgent
- ExecutorBAgent
- CriticAgent
- JudgeAgent
"""

from app.agents.base import BaseAgent
from app.agents.critic import CriticAgent
from app.agents.executor_a import ExecutorAAgent
from app.agents.executor_b import ExecutorBAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent

__all__ = [
    "BaseAgent",
    "PlannerAgent",
    "ExecutorAAgent",
    "ExecutorBAgent",
    "CriticAgent",
    "JudgeAgent",
]

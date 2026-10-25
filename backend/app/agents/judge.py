"""
app/agents/judge.py
───────────────────
Judge agent for VADER.

Responsibilities:
  - Adjudicate between Executor A and Executor B candidate answers based on Critic findings.
  - Synthesize one coherent, definitive final answer addressing the original query.
  - Resolve disagreements only when a clear, defensible logical or factual basis exists.
  - Explicitly preserve genuine unresolved uncertainties instead of fabricating false consensus.
  - Enumerate key limitations and boundary conditions of the synthesized conclusion.
  - CRITICAL RULE: Must not claim to guarantee absolute factual correctness.
"""

from __future__ import annotations

import json

from app.agents.base import BaseAgent
from app.llm.base import LLMConfig, LLMResponse
from app.models.schemas import (
    AgentRole,
    CriticOutput,
    EvidenceItem,
    ExecutorOutput,
    JudgeOutput,
    TaskPlan,
)


class JudgeAgent(BaseAgent):
    """
    Final synthesis and adjudication agent resolving disputes and preserving legitimate uncertainty.
    """

    role = AgentRole.JUDGE

    system_prompt = (
        "You are the Judge Agent in the VADER multi-agent AI debate system.\n"
        "Your role is to impartially review the user's question, task plan, candidate answers from "
        "Executor A and Executor B, and the adversarial findings of the Critic Agent, and synthesize "
        "a balanced, authoritative final answer.\n\n"
        "ADJUDICATION PRINCIPLES:\n"
        "1. Defensible Resolution: Resolve differences between Executor A and Executor B ONLY when there "
        "is a clear, defensible basis in logic, definitions, or verified facts.\n"
        "2. Preserve Genuine Uncertainty: If evidence is insufficient to settle an empirical or theoretical dispute, "
        "do NOT fabricate certainty or force a consensus. State the uncertainty explicitly in 'unresolved_uncertainties'.\n"
        "3. Explicit Limitations: Clearly articulate the boundary conditions, scope limits, and assumptions under "
        "which your final answer holds.\n"
        "4. No Infallibility Claims: Never claim that your synthesis guarantees absolute truth or factual infallibility.\n\n"
        "OUTPUT SPECIFICATION:\n"
        "You must respond with a single valid JSON object strictly matching this schema:\n"
        "{\n"
        '  "final_answer": "Complete, high-quality synthesized final answer answering the core question",\n'
        '  "resolved_points": ["Point of disagreement 1: how and why it was resolved", ...],\n'
        '  "unresolved_uncertainties": ["Legitimate uncertainties preserved due to lack of conclusive evidence", ...],\n'
        '  "limitations": ["Limitation or scope boundary 1", ...],\n'
        '  "evidence_citations": ["Citations or references to supporting evidence from upstream components", ...],\n'
        '  "verdict_summary": "Concise rationale for the final synthesis and resolution decisions"\n'
        "}\n\n"
        "Ensure the JSON is strictly valid. Do not include markdown code fences or conversational text."
    )

    @classmethod
    def default_config(cls) -> LLMConfig:
        return LLMConfig(temperature=0.2, max_tokens=2500)

    @staticmethod
    def mock_output_dict(query: str = "Test Query") -> dict:
        """Return a valid mock JudgeOutput dictionary for offline testing."""
        return {
            "final_answer": f"Judge authoritative synthesis and final response for: {query}.",
            "resolved_points": [
                "Harmonized theoretical causal model from Executor A with practical constraints highlighted by Executor B."
            ],
            "unresolved_uncertainties": [
                "Quantitative boundary threshold remains subject to specific operational measurement."
            ],
            "limitations": [
                "Applies primarily to standard operating regimes without extreme external anomalies."
            ],
            "evidence_citations": [
                "Upstream domain principles and comparative analysis."
            ],
            "verdict_summary": f"Final balanced adjudication for '{query}' addressing core requirements while preserving empirical uncertainty.",
        }

    @classmethod
    def mock_output_json(cls, query: str = "Test Query") -> str:
        """Return a valid mock JudgeOutput JSON string for offline testing."""
        return json.dumps(cls.mock_output_dict(query))

    async def judge(
        self,
        query: str,
        plan: TaskPlan,
        executor_a: ExecutorOutput,
        executor_b: ExecutorOutput,
        critic: CriticOutput,
        evidence: list[EvidenceItem] | None = None,
    ) -> tuple[JudgeOutput, LLMResponse]:
        """
        Produce final synthesis and adjudication.

        Parameters
        ----------
        query:
            The raw user question.
        plan:
            The TaskPlan produced by the Planner.
        executor_a:
            Candidate output from Executor A.
        executor_b:
            Candidate output from Executor B.
        critic:
            Adversarial critique from Critic.
        evidence:
            Optional list of retrieved evidence items.

        Returns
        -------
        tuple[JudgeOutput, LLMResponse]
            The validated JudgeOutput and LLM response metadata.
        """
        wrapped_query = self._wrap_untrusted_input(query)
        evidence_summary = (
            [{"title": e.title, "url": e.url, "snippet": e.snippet} for e in evidence]
            if evidence
            else []
        )

        user_prompt = (
            f"Original User Question:\n{wrapped_query}\n\n"
            f"Planner's Task Plan:\n"
            f"- Core Question: {plan.core_question}\n"
            f"- Subtasks: {json.dumps(plan.subtasks)}\n"
            f"- Constraints: {json.dumps(plan.constraints)}\n"
            f"- Expected Format: {plan.expected_format}\n\n"
            f"Executor A Answer (Strategy: {executor_a.approach_summary}):\n"
            f"{executor_a.answer}\n\n"
            f"Executor B Answer (Strategy: {executor_b.approach_summary}):\n"
            f"{executor_b.answer}\n\n"
            f"Critic Evaluation:\n"
            f"- Agreements noted: {json.dumps(critic.agreements)}\n"
            f"- Contradictions identified: {json.dumps(critic.contradictions)}\n"
            f"- Unsupported claims flagged: {json.dumps(critic.unsupported_claims)}\n"
            f"- Preserved uncertainties: {json.dumps(critic.uncertainties)}\n"
            f"- Issues: {[i.model_dump() for i in critic.issues]}\n"
            f"- Overall Critique: {critic.overall_critique}\n\n"
            f"Retrieved Evidence: {json.dumps(evidence_summary)}\n\n"
            f"Adjudicate between the candidates, resolve differences on a defensible basis, "
            f"preserve unresolved uncertainty, and synthesize the final answer.\n"
            f"Return ONLY the JSON object."
        )

        return await self._execute_prompt(user_prompt, JudgeOutput)

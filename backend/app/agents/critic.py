"""
app/agents/critic.py
────────────────────
Critic agent for VADER.

Responsibilities:
  - Adversarially examine candidate answers from Executor A and Executor B.
  - Identify contradictions, unsupported claims, missing requirements, and logical flaws.
  - Return structured findings with severity, affected claim, explanation, and recommended correction.
  - Explicitly mark uncertainty when evidence is insufficient.
  - CRITICAL RULES:
    1. Agreement between executors is NOT proof of truth; shared fallacies must be called out.
    2. Must NOT invent sources or pretend to have conducted a live web search.
"""

from __future__ import annotations

import json

from app.agents.base import BaseAgent
from app.llm.base import LLMConfig, LLMResponse
from app.models.schemas import (
    AgentRole,
    ClaimVerificationResult,
    CriticOutput,
    CritiqueSeverity,
    ExecutorOutput,
    TaskPlan,
)


class CriticAgent(BaseAgent):
    """
    Adversarial evaluation agent identifying contradictions, flaws, and unsupported claims.
    """

    role = AgentRole.CRITIC

    system_prompt = (
        "You are the Critic Agent in the VADER multi-agent AI system.\n"
        "Your role is to rigorously and adversarially evaluate candidate answers produced by Executor A and Executor B.\n\n"
        "EVALUATION CRITERIA:\n"
        "1. Contradictions: Direct tensions, incompatible claims, or factual clashes between the candidates.\n"
        "2. Unsupported Claims: Bold assertions, statistical figures, or factual declarations lacking evidence or logical backing.\n"
        "3. Missing Requirements: Failure to address specific constraints or subtasks outlined in the Task Plan.\n"
        "4. Logical Flaws: Non-sequiturs, false dichotomies, circular reasoning, or unstated fragile assumptions.\n\n"
        "CRITICAL BEHAVIORAL DIRECTIVES:\n"
        "- AGREEMENT IS NOT TRUTH: If both Executor A and Executor B agree on a point, do NOT treat it as validated truth. "
        "Check whether both are committing the same assumption or error.\n"
        "- NO INVENTED SOURCES: Do not invent bibliographic citations, fake URLs, or pretend you performed a real-time web search.\n"
        "- EXPLICIT UNCERTAINTY: If neither candidate has provided sufficient evidence to resolve a factual disagreement, "
        "explicitly record it as an uncertainty rather than taking a guess.\n\n"
        "OUTPUT SPECIFICATION:\n"
        "You must respond with a single valid JSON object strictly matching this schema:\n"
        "{\n"
        '  "agreements": ["Points where both executors agree (with notes on whether the point is truly verified)", ...],\n'
        '  "contradictions": ["Explicit points where Executor A and B contradict each other", ...],\n'
        '  "unsupported_claims": ["Specific claims made without sufficient logical or factual justification", ...],\n'
        '  "uncertainties": ["Explicit areas where evidence is inconclusive or insufficient", ...],\n'
        '  "issues": [\n'
        "    {\n"
        '      "issue_id": "ISSUE-1",\n'
        '      "target": "executor_a" | "executor_b" | "both",\n'
        '      "severity": "low" | "medium" | "high",\n'
        '      "affected_claim": "The exact claim under scrutiny",\n'
        '      "explanation": "Detailed explanation of the weakness, flaw, or missing proof",\n'
        '      "recommended_correction": "How this should be corrected or qualified"\n'
        "    }\n"
        "  ],\n"
        '  "overall_critique": "Concise summary evaluation synthesizing the key critique points"\n'
        "}\n\n"
        "Ensure the JSON is strictly valid. Do not include markdown code fences or conversational text."
    )

    @classmethod
    def default_config(cls) -> LLMConfig:
        return LLMConfig(temperature=0.2, max_tokens=2048)

    @staticmethod
    def mock_output_dict(query: str = "Test Query") -> dict:
        """Return a valid mock CriticOutput dictionary for offline testing."""
        return {
            "agreements": [
                "Both executors agree on the core conceptual definition, though empirical validation remains needed."
            ],
            "contradictions": [
                "Executor A claims strict deterministic causality, whereas Executor B highlights contextual dependencies."
            ],
            "unsupported_claims": [
                "Executor A did not provide empirical rate figures; Executor B did not specify sample boundaries."
            ],
            "uncertainties": [
                "The exact quantitative impact under extreme conditions remains uncertain without empirical measurement."
            ],
            "issues": [
                {
                    "issue_id": "ISSUE-1",
                    "target": "executor_a",
                    "severity": CritiqueSeverity.MEDIUM.value,
                    "affected_claim": "Strict deterministic causality across all regimes",
                    "explanation": "Overgeneralizes beyond the theoretical boundary conditions specified in the plan.",
                    "recommended_correction": "Explicitly qualify the domain of applicability.",
                },
                {
                    "issue_id": "ISSUE-2",
                    "target": "executor_b",
                    "severity": CritiqueSeverity.LOW.value,
                    "affected_claim": "Friction limits all practical implementations",
                    "explanation": "Broad generalization without distinguishing low-friction modern alternatives.",
                    "recommended_correction": "Provide nuance regarding operational mitigations.",
                },
            ],
            "overall_critique": f"Adversarial critique for: {query}. Both candidates provide valuable partial insights, but require qualification regarding boundary conditions.",
        }

    @classmethod
    def mock_output_json(cls, query: str = "Test Query") -> str:
        """Return a valid mock CriticOutput JSON string for offline testing."""
        return json.dumps(cls.mock_output_dict(query))

    async def critique(
        self,
        query: str,
        plan: TaskPlan,
        executor_a: ExecutorOutput,
        executor_b: ExecutorOutput,
        verifications: list[ClaimVerificationResult] | None = None,
    ) -> tuple[CriticOutput, LLMResponse]:
        """
        Evaluate candidate answers from Executor A and B.

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
        verifications:
            Optional list of ClaimVerificationResult instances from the retrieval layer.

        Returns
        -------
        tuple[CriticOutput, LLMResponse]
            The validated CriticOutput and LLM response metadata.
        """
        wrapped_query = self._wrap_untrusted_input(query)
        verifications_text = (
            "\n".join([
                f"- Claim: '{v.claim}' | Status: {v.status.value} | Explanation: {v.explanation}"
                for v in verifications
            ])
            if verifications
            else "No claims verified with external search."
        )

        user_prompt = (
            f"User Question:\n{wrapped_query}\n\n"
            f"Planner's Task Plan:\n"
            f"- Core Question: {plan.core_question}\n"
            f"- Subtasks: {json.dumps(plan.subtasks)}\n"
            f"- Constraints: {json.dumps(plan.constraints)}\n"
            f"- Expected Format: {plan.expected_format}\n\n"
            f"Candidate Answer from Executor A (Strategy: {executor_a.approach_summary}):\n"
            f"Answer: {executor_a.answer}\n"
            f"Assumptions: {json.dumps(executor_a.key_assumptions)}\n"
            f"Claims to verify: {json.dumps(executor_a.claims_to_verify)}\n\n"
            f"Candidate Answer from Executor B (Strategy: {executor_b.approach_summary}):\n"
            f"Answer: {executor_b.answer}\n"
            f"Assumptions: {json.dumps(executor_b.key_assumptions)}\n"
            f"Claims to verify: {json.dumps(executor_b.claims_to_verify)}\n\n"
            f"Verified Claim Findings (from Retrieval Layer):\n"
            f"{verifications_text}\n\n"
            f"Perform a rigorous adversarial critique comparing both answers against the question, plan, and verified evidence.\n"
            f"Remember: agreement is not proof of truth, and do not invent sources.\n"
            f"Return ONLY the JSON object."
        )

        return await self._execute_prompt(user_prompt, CriticOutput)

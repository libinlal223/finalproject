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
from typing import Any

from app.agents.base import BaseAgent
from app.llm.base import LLMConfig, LLMError, LLMResponse, LLMValidationError
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
        "Executor A and Executor B, the adversarial findings of the Critic Agent, and the structured "
        "Claim Verification Report, and synthesize an authoritative, factually grounded final answer "
        "with defensible resolution of all candidate disputes.\n\n"
        "ADJUDICATION PRINCIPLES:\n"
        "1. Factual Correctness & Verification Grounding (PARAMOUNT):\n"
        "   - Strongly favor candidates that corrected claims identified as CONTRADICTED by the verification report.\n"
        "   - Penalize and reject candidates that persist with contradicted claims or make bold assertions without evidence.\n"
        "   - Favor candidates that successfully retained and integrated SUPPORTED claims.\n"
        "2. Anti-Superficiality (Length & Confidence are NOT Correctness):\n"
        "   - NEVER choose an answer merely because it is longer, more eloquent, or uses more confident rhetoric.\n"
        "   - NEVER choose an answer merely because both candidates agreed, unless corroborated by evidence.\n"
        "3. Degradation Safeguard:\n"
        "   - If an executor revision made an answer worse (e.g. retreated from a supported fact or introduced new errors), "
        "reject that regression and rely on the grounded earlier/opposing candidate.\n"
        "4. Preserve Genuine Uncertainty:\n"
        "   - If evidence is insufficient to settle an empirical dispute, do NOT fabricate certainty or force a consensus. "
        "State the uncertainty explicitly in 'unresolved_uncertainties'.\n"
        "5. Explicit Limitations & Boundaries:\n"
        "   - Clearly articulate the boundary conditions, scope limits, and operational assumptions.\n"
        "6. Quantitative & Numerical Grounding:\n"
        "   - Scrutinize all numbers, percentages, rates, and statistics. If a numerical claim was evaluated as CONTRADICTED, "
        "INSUFFICIENT_EVIDENCE, or NOT_CHECKED in the Claim Verification Report, you MUST NOT present that figure as an established "
        "or verified fact in your final answer. You must either explicitly disclose its unverified/inconclusive status or record it "
        "under 'unresolved_uncertainties'.\n"
        "7. No Infallibility Claims:\n"
        "   - Never claim that your synthesis guarantees absolute truth or factual infallibility.\n\n"
        "OUTPUT SPECIFICATION:\n"
        "You must respond with a single valid JSON object strictly matching this schema:\n"
        "{\n"
        '  "final_answer": "Complete, high-quality synthesized final answer answering the core question",\n'
        '  "resolved_points": ["Point of disagreement 1: how and why it was resolved (1-2 sentences)", ...],\n'
        '  "unresolved_uncertainties": ["Legitimate uncertainties preserved due to lack of conclusive evidence (1-2 sentences)", ...],\n'
        '  "limitations": ["Limitation or scope boundary (1-2 sentences)", ...],\n'
        '  "evidence_citations": ["Citations or references to supporting evidence from upstream components", ...],\n'
        '  "verdict_summary": "Concise 1-2 sentence summary of the final verdict and candidate preference",\n'
        '  "adjudication_rationale": "Focused 2-4 sentence explanation of why this answer was chosen based on evidence quality and correctness",\n'
        '  "preferred_candidate": "executor_a" | "executor_b" | "synthesis" | "neither",\n'
        '  "addressed_issues": ["ISSUE-1", ...]\n'
        "}\n\n"
        "TOKEN BUDGET & CONCISENESS INSTRUCTION:\n"
        "Keep auxiliary fields ('adjudication_rationale', 'verdict_summary', 'resolved_points', 'limitations', 'unresolved_uncertainties') "
        "concise and focused (1-3 sentences or bullet points each) so the token budget is dedicated to the complete, high-quality 'final_answer'.\n"
        "Ensure the JSON is strictly valid. Do not include markdown code fences or conversational text."
    )

    @classmethod
    def default_config(cls) -> LLMConfig:
        return LLMConfig(temperature=0.2, max_tokens=2560, json_mode=True)

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
            "adjudication_rationale": "Selected balanced synthesis prioritizing verified empirical evidence over ungrounded assertions.",
            "preferred_candidate": "synthesis",
            "addressed_issues": ["ISSUE-1"],
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
        verifications: list[Any] | None = None,
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
        verifications:
            Optional list of ClaimVerificationResult instances.

        Returns
        -------
        tuple[JudgeOutput, LLMResponse]
            The validated JudgeOutput and LLM response metadata.
        """
        wrapped_query = self._wrap_untrusted_input(query)
        evidence_summary = (
            [
                {
                    "title": e.title,
                    "url": e.url,
                    "snippet": e.snippet.replace("<", "&lt;").replace(">", "&gt;"),
                }
                for e in evidence
            ]
            if evidence
            else []
        )

        verifications_lines: list[str] = []
        if verifications:
            for v in verifications:
                verifications_lines.append(
                    f"- [{getattr(v, 'claim_id', 'CLAIM')}] Status: {getattr(v.status, 'value', str(v.status)).upper()} | "
                    f"Claim: \"{getattr(v, 'claim_text', getattr(v, 'claim', ''))}\" | "
                    f"Verdict: {v.explanation}"
                )
            verifications_text = "\n".join(verifications_lines)
        else:
            verifications_text = "No structured verification report available."

        user_prompt = (
            f"Original User Question:\n{wrapped_query}\n\n"
            f"Planner's Task Plan:\n"
            f"- Core Question: {plan.core_question}\n"
            f"- Subtasks: {json.dumps(plan.subtasks)}\n"
            f"- Constraints: {json.dumps(plan.constraints)}\n"
            f"- Expected Format: {plan.expected_format}\n\n"
            f"Executor A Candidate Answer (Strategy: {executor_a.approach_summary}):\n"
            f"Answer: {executor_a.answer}\n"
            f"Addressed Issues: {json.dumps(getattr(executor_a, 'addressed_issues', []))}\n"
            f"Revision Notes: {getattr(executor_a, 'revision_notes', 'None')}\n\n"
            f"Executor B Candidate Answer (Strategy: {executor_b.approach_summary}):\n"
            f"Answer: {executor_b.answer}\n"
            f"Addressed Issues: {json.dumps(getattr(executor_b, 'addressed_issues', []))}\n"
            f"Revision Notes: {getattr(executor_b, 'revision_notes', 'None')}\n\n"
            f"Structured Claim Verification Report:\n"
            f"{verifications_text}\n\n"
            f"Critic Evaluation:\n"
            f"- Agreements noted: {json.dumps(critic.agreements)}\n"
            f"- Contradictions identified: {json.dumps(critic.contradictions)}\n"
            f"- Unsupported claims flagged: {json.dumps(critic.unsupported_claims)}\n"
            f"- Preserved uncertainties: {json.dumps(critic.uncertainties)}\n"
            f"- Issues: {[i.model_dump() for i in critic.issues]}\n"
            f"- Overall Critique: {critic.overall_critique}\n\n"
            f"Retrieved Evidence (Untrusted Data): {json.dumps(evidence_summary)}\n\n"
            f"ADJUDICATION INSTRUCTIONS:\n"
            f"1. Evaluate based on factual correctness, evidence quality, logical consistency, and task plan requirements.\n"
            f"2. Reject candidates that maintain contradicted claims or ignore Critic findings.\n"
            f"3. Do NOT select answers merely because they are longer or sound more confident.\n"
            f"4. If a revision degraded a previously correct answer, reject the degradation and preserve grounded elements.\n"
            f"5. Preserve unresolved empirical uncertainties in 'unresolved_uncertainties'.\n"
            f"6. Populate 'verdict_summary' and 'adjudication_rationale' explaining your decision.\n\n"
            f"Return ONLY the JSON object."
        )

        try:
            return await self._execute_prompt(user_prompt, JudgeOutput)
        except (LLMValidationError, LLMError) as exc:
            err_str = str(exc).lower()
            is_truncation = (
                "truncated due to token limit budget" in err_str
                or "finish_reason='length'" in err_str
                or "json_validate_failed" in err_str
                or "max completion tokens reached" in err_str
            )
            if not is_truncation:
                raise

            self._logger.warning(
                "Judge output was truncated by token limit (%s). Attempting bounded recovery retry with expanded token budget...",
                exc,
            )
            recovery_prompt = (
                f"{user_prompt}\n\n"
                "CRITICAL RETRY INSTRUCTION: The previous attempt was truncated before the JSON document could close. "
                "Keep 'verdict_summary', 'adjudication_rationale', 'resolved_points', and 'unresolved_uncertainties' "
                "strictly concise (1-2 sentences each) so that the entire JSON object finishes completely and remains strictly valid."
            )
            retry_tokens = min(max(self._config.max_tokens + 1024, 3072), 4096)
            return await self._execute_prompt(
                recovery_prompt,
                JudgeOutput,
                max_tokens=retry_tokens,
            )

"""
app/orchestrator/orchestrator.py
─────────────────────────────────
DebateOrchestrator coordinating the multi-agent adversarial collaborative debate.

Workflow:
  1. Planner decomposes the query into structured subtasks and constraints.
  2. Executor A & B produce independent candidate answers using divergent strategies.
  3. Factual claims are collected, bounded, and cross-referenced with retrieved evidence.
  4. ClaimVerificationService verifies claims (distinguishing search failure from lack of proof).
  5. Critic evaluates both candidates, contradictions, and verified evidence.
  6. Bounded debate loop: if material issues remain and round budget permits, executors revise.
  7. Judge performs final synthesis and adjudication with strictly grounded evidence.
  8. A validated QueryResponse is returned with full transparent trace.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from app.agents.critic import CriticAgent
from app.agents.executor_a import ExecutorAAgent
from app.agents.executor_b import ExecutorBAgent
from app.agents.judge import JudgeAgent
from app.agents.planner import PlannerAgent
from app.config import Settings, get_settings
from app.llm.base import (
    BaseLLMClient,
    LLMConfig,
    LLMError,
    LLMValidationError,
    build_llm_client,
    sanitize_error_message,
)
from app.models.schemas import (
    AgentRole,
    AgentTraceEntry,
    ClaimStatus,
    ClaimVerificationReport,
    ClaimVerificationResult,
    CritiqueSeverity,
    DebateRound,
    DebateStatus,
    EvidenceItem,
    ExecutorOutput,
    QueryResponse,
    RetrievalStatus,
    TaskPlan,
    VerificationReportStatus,
)
from app.orchestrator.similarity import compute_answer_similarity
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.service import EvidenceRetrievalService
from app.retrieval.targeted_retrieval import retrieve_for_claims

logger = logging.getLogger(__name__)

# Resource bounding limits
MAX_CLAIMS_PER_ROUND = 5
MAX_EVIDENCE_ITEMS = 10
MAX_SEARCH_QUERY_LEN = 250
DEFAULT_MAX_LLM_CALLS = 20


class DebateOrchestrator:
    """
    Coordinates the multi-agent adversarial collaborative debate process.
    """

    def __init__(
        self,
        planner: PlannerAgent | None = None,
        executor_a: ExecutorAAgent | None = None,
        executor_b: ExecutorBAgent | None = None,
        critic: CriticAgent | None = None,
        judge: JudgeAgent | None = None,
        retrieval_service: EvidenceRetrievalService | None = None,
        claim_verifier: ClaimVerificationService | None = None,
        llm_client: BaseLLMClient | None = None,
        settings: Settings | None = None,
        max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    ) -> None:
        self._settings = settings or get_settings()

        def _resolve_client(role: AgentRole) -> BaseLLMClient:
            if llm_client is not None:
                return llm_client
            provider = self._settings.get_agent_provider(role)
            model_id = self._settings.get_agent_model_id(role)
            is_json_role = role in (AgentRole.PLANNER, AgentRole.CRITIC, AgentRole.JUDGE)
            role_max_tokens = self._settings.llm_max_tokens
            if role in (AgentRole.CRITIC, AgentRole.JUDGE):
                role_max_tokens = max(role_max_tokens, 2560)
            config = LLMConfig(
                model_id=model_id,
                temperature=self._settings.llm_temperature,
                max_tokens=role_max_tokens,
                timeout_seconds=self._settings.llm_timeout_seconds,
                max_retries=self._settings.llm_max_retries,
                json_mode=is_json_role,
            )
            return build_llm_client(provider, config)

        planner_client = _resolve_client(AgentRole.PLANNER)
        exec_a_client = _resolve_client(AgentRole.EXECUTOR_A)
        exec_b_client = _resolve_client(AgentRole.EXECUTOR_B)
        critic_client = _resolve_client(AgentRole.CRITIC)
        judge_client = _resolve_client(AgentRole.JUDGE)

        self._client = llm_client or planner_client

        # Initialize agents with shared or role-specific client
        self._planner = planner or PlannerAgent(client=planner_client)
        self._executor_a = executor_a or ExecutorAAgent(client=exec_a_client)
        self._executor_b = executor_b or ExecutorBAgent(client=exec_b_client)
        self._critic = critic or CriticAgent(client=critic_client)
        judge_cfg = JudgeAgent.default_config()
        if not llm_client:
            judge_cfg.max_tokens = max(judge_cfg.max_tokens, 2560)
        self._judge = judge or JudgeAgent(client=judge_client, config=judge_cfg)

        # Initialize retrieval & verification
        self._retrieval_service = retrieval_service or EvidenceRetrievalService()
        self._claim_verifier = claim_verifier or ClaimVerificationService(llm_client=critic_client)

        self._max_llm_calls = max_llm_calls
        self._llm_call_count = 0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def llm_call_count(self) -> int:
        """Current number of LLM invocations during orchestration."""
        return self._llm_call_count

    @property
    def planner(self) -> PlannerAgent:
        return self._planner

    @property
    def executor_a(self) -> ExecutorAAgent:
        return self._executor_a

    @property
    def executor_b(self) -> ExecutorBAgent:
        return self._executor_b

    @property
    def critic(self) -> CriticAgent:
        return self._critic

    @property
    def judge(self) -> JudgeAgent:
        return self._judge

    def _increment_call_count(self) -> bool:
        """Increment LLM call counter. Returns False if budget exceeded."""
        self._llm_call_count += 1
        if self._llm_call_count > self._max_llm_calls:
            self._logger.warning(
                "LLM call budget exceeded: %d > %d", self._llm_call_count, self._max_llm_calls
            )
            return False
        return True

    def _determine_effective_max_rounds(self, requested_rounds: int | None) -> int:
        """
        Enforce user max_rounds strictly bounded by server configuration.
        """
        server_max = self._settings.debate_max_rounds
        if requested_rounds is None:
            return server_max
        return max(1, min(requested_rounds, server_max, 10))

    def _collect_claims(
        self, plan: TaskPlan, cand_a: ExecutorOutput, cand_b: ExecutorOutput
    ) -> list[str]:
        """Collect and bound factual claims requiring verification."""
        candidates = []
        for claim in cand_a.claims_to_verify:
            c = claim.strip()
            if c and c not in candidates:
                candidates.append(c)
        for claim in cand_b.claims_to_verify:
            c = claim.strip()
            if c and c not in candidates:
                candidates.append(c)

        if not candidates and plan.evidence_needed:
            for item in plan.evidence_needed:
                c = item.strip()
                if c and c not in candidates:
                    candidates.append(c)

        return candidates[:MAX_CLAIMS_PER_ROUND]

    def _format_critique_for_executor(
        self,
        executor_id: str,
        critic_output: Any,
        verifications: list[ClaimVerificationResult] | None = None,
    ) -> str:
        """Format targeted, actionable critique instructions for an executor."""
        relevant_issues = [
            i for i in critic_output.issues
            if i.target in {executor_id, "both"}
        ]
        lines = [f"Overall Critique: {critic_output.overall_critique}\n"]

        if relevant_issues:
            lines.append("TARGETED ISSUES TO RESOLVE IN YOUR REVISION:")
            for issue in relevant_issues:
                claim_id_prefix = f"[{issue.claim_id}] " if getattr(issue, "claim_id", None) else ""
                lines.append(
                    f"- [{issue.severity.value.upper()}] [{issue.issue_id}] {claim_id_prefix}Affected Claim: {issue.affected_claim}\n"
                    f"  Flaw: {issue.explanation}\n"
                    f"  Recommended Correction: {issue.recommended_correction}"
                )

        if critic_output.contradictions:
            lines.append("\nCONTRADICTIONS WITH OPPOSING CANDIDATE:")
            for c in critic_output.contradictions:
                lines.append(f"- {c}")

        if verifications:
            supported = [v for v in verifications if v.status == ClaimStatus.SUPPORTED]
            if supported:
                lines.append("\nPRESERVE SUPPORTED CLAIMS (Do NOT retract or weaken these corroborated facts):")
                for s in supported:
                    lines.append(f"- [{s.claim_id}] {s.claim_text}")

            contradicted = [v for v in verifications if v.status == ClaimStatus.CONTRADICTED]
            if contradicted:
                lines.append("\nRETRACT OR CORRECT CONTRADICTED CLAIMS:")
                for c in contradicted:
                    lines.append(f"- [{c.claim_id}] {c.claim_text} (Contradicted by evidence: {c.explanation})")

            inconclusive = [
                v for v in verifications
                if v.status in {ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.NOT_CHECKED}
            ]
            if inconclusive:
                lines.append("\nEXPLICIT UNCERTAINTIES (Must be qualified, not asserted as absolute fact):")
                for inc in inconclusive:
                    lines.append(f"- [{inc.claim_id}] {inc.claim_text} ({inc.explanation})")

        return "\n".join(lines)

    def _has_material_issues(self, critic_output: Any) -> bool:
        """
        Check if the Critic identified material unresolved issues.
        Material issues are HIGH or MEDIUM severity issues or direct contradictions.
        """
        high_med_issues = [
            i for i in critic_output.issues
            if i.severity in {CritiqueSeverity.HIGH, CritiqueSeverity.MEDIUM}
        ]
        return bool(high_med_issues or critic_output.contradictions or critic_output.unsupported_claims)

    @classmethod
    def _enforce_grounding_in_answer(
        cls,
        text: str,
        verifications: list[ClaimVerificationResult] | None,
        unresolved_uncertainties: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """
        Enforce evidence grounding in answer text (Judge synthesis or fallback candidate).

        Prevents unsupported or contradicted factual claims (both quantitative and qualitative)
        from being presented as established facts. If an unsupported claim is asserted in the text,
        it is explicitly qualified adjacent to the assertion (or omitted) and recorded in uncertainties.

        Preserves the distinction between INSUFFICIENT_EVIDENCE and CONTRADICTED. Missing evidence
        is never treated as proof of falsity.
        """
        uncertainties = list(unresolved_uncertainties or [])
        if not text or not verifications:
            return text, uncertainties

        stopwords = {
            "the", "and", "that", "this", "with", "from", "for", "have", "been", "will",
            "would", "could", "should", "were", "they", "their", "which", "about", "there",
            "what", "when", "where", "more", "most", "some", "such", "only", "into", "also",
            "than", "then", "very", "much", "both", "each", "other", "after", "over"
        }

        number_pattern = re.compile(
            r"\b\d+(?:[\.,]\d+)?(?:\s*(?:%|percent|qubits?|ms|s|kg|g|km|m|billion|million|trillion|\$|€|£))?",
            re.IGNORECASE,
        )

        qualification_markers = (
            "[unverified", "[inconclusive", "[contradicted", "unconfirmed",
            "not corroborated", "disputed", "remains uncertain", "insufficient evidence",
            "lack of empirical", "according to unverified", "it is claimed", "is hypothesized",
            "unverified quantitative claim", "unverified claim"
        )

        existing_uncertainties_lower = " ".join(uncertainties).lower()
        modified_text = text

        for v in verifications:
            if v.status == ClaimStatus.SUPPORTED:
                continue

            claim_text = v.claim_text.strip()
            if not claim_text:
                continue

            is_contradicted = (v.status == ClaimStatus.CONTRADICTED)
            status_desc = "contradicted by evidence" if is_contradicted else "inconclusive / unverified by evidence"
            adj_tag = (
                f" [Contradicted by evidence: {v.explanation}]"
                if is_contradicted
                else f" [Unverified / Inconclusive: {v.explanation}]"
            )

            # 1. Identify quantitative figures in claim
            quant_matches = [m.group(0).strip() for m in number_pattern.finditer(claim_text)]
            significant_quant = [
                q for q in quant_matches
                if len(q) > 1 or any(c in q for c in "%$€£.") or not q.isdigit() or (q.isdigit() and len(q) >= 3)
            ]

            # 2. Extract qualitative content tokens (named entities, subjects, actions)
            raw_tokens = re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", claim_text)
            content_tokens = [t.lower() for t in raw_tokens if t.lower() not in stopwords]

            # Split text into paragraphs to preserve overall structure
            paragraphs = modified_text.split("\n\n")
            new_paragraphs = []
            claim_flagged = False

            for para in paragraphs:
                if not para.strip():
                    new_paragraphs.append(para)
                    continue

                # Don't modify metadata banners or disclosure sections
                if para.startswith("[UNADJUDICATED") or para.startswith("[UNVERIFIED") or para.startswith("[CONTRADICTED"):
                    new_paragraphs.append(para)
                    continue

                # Split paragraph into sentences preserving bracketed tags and punctuation
                sentence_parts = re.split(r"(?:(?<=[.!?])|(?<=[\]\)]))\s+", para)
                new_sentences = []

                for s in sentence_parts:
                    s_stripped = s.strip()
                    if not s_stripped:
                        new_sentences.append(s)
                        continue

                    s_lower = s_stripped.lower()

                    # If this assertion already has this specific claim tag or standalone bracket tag, skip
                    if adj_tag.strip().lower() in s_lower or (s_stripped.startswith("[") and s_stripped.endswith("]")):
                        new_sentences.append(s)
                        continue

                    # Check match:
                    quant_in_s = [q for q in significant_quant if q.lower() in s_lower]
                    s_tokens = set(re.findall(r"\b[a-zA-Z0-9\-\.]{3,}\b", s_lower))
                    token_overlap = [t for t in content_tokens if t in s_tokens]
                    token_ratio = len(token_overlap) / max(1, len(content_tokens))

                    claim_sub = claim_text[:min(40, len(claim_text))].lower()
                    phrase_match = len(claim_sub) >= 15 and claim_sub in s_lower

                    is_match = False
                    if quant_in_s and (token_overlap or len(quant_in_s) >= 2):
                        is_match = True
                    elif phrase_match:
                        is_match = True
                    elif len(content_tokens) >= 3 and (token_ratio >= 0.50 or len(token_overlap) >= 3):
                        is_match = True
                    elif len(content_tokens) < 3 and len(token_overlap) >= 2:
                        is_match = True

                    if is_match:
                        claim_flagged = True
                        if s.endswith("."):
                            qualified_sentence = s[:-1].rstrip() + adj_tag + "."
                        else:
                            qualified_sentence = s.rstrip() + adj_tag
                        new_sentences.append(qualified_sentence)
                    else:
                        new_sentences.append(s)

                new_paragraphs.append(" ".join(new_sentences))

            modified_text = "\n\n".join(new_paragraphs)

            # Record prominent caveat in unresolved_uncertainties if flagged or present
            quant_in_answer = any(q.lower() in modified_text.lower() for q in significant_quant)
            claim_in_answer = claim_flagged or quant_in_answer or (claim_text[:50].lower() in modified_text.lower())

            if claim_in_answer:
                already_disclosed = (
                    v.claim_id.lower() in existing_uncertainties_lower
                    or (significant_quant and any(q.lower() in existing_uncertainties_lower for q in significant_quant))
                )
                if not already_disclosed:
                    caveat_type = "Unverified quantitative claim" if significant_quant else "Unverified claim"
                    caveat = (
                        f"{caveat_type} [{v.claim_id}]: '{claim_text}' is {status_desc} "
                        f"({v.explanation}). It must not be treated as a verified empirical finding."
                    )
                    uncertainties.append(caveat)
                    existing_uncertainties_lower += " " + caveat.lower()

        return modified_text, uncertainties

    @classmethod
    def _format_unadjudicated_fallback(
        cls,
        candidate_answer: str,
        failure_reason: str,
        verification_report: ClaimVerificationReport | None = None,
    ) -> str:
        """
        Format an unadjudicated fallback answer with clear disclosure of unverified and inconclusive claims.
        Guarantees that raw candidate outputs are never presented as verified syntheses.
        """
        zero_sources = not verification_report or verification_report.evidence_count == 0
        all_unverified = bool(
            verification_report
            and verification_report.results
            and all(
                r.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.NOT_CHECKED)
                for r in verification_report.results
            )
        )

        evidence_warning = ""
        if zero_sources or all_unverified:
            evidence_warning = (
                "\nEvidence Notice: ZERO external evidence sources were verified. "
                "All factual assertions, technical metrics, dates, and quantitative estimates in this fallback candidate "
                "originate solely from unverified candidate model generation and must NOT be treated as established empirical facts."
            )

        banner = (
            "[UNADJUDICATED FALLBACK CANDIDATE - NOT VERIFIED]\n"
            f"Notice: Adjudication was not completed ({failure_reason}). "
            "The following candidate answer has NOT been adjudicated, verified, or synthesized by the Judge."
            f"{evidence_warning}"
        )

        grounded_candidate = candidate_answer
        if verification_report and verification_report.results:
            grounded_candidate, _ = cls._enforce_grounding_in_answer(
                text=candidate_answer,
                verifications=verification_report.results,
                unresolved_uncertainties=[],
            )

        if zero_sources or all_unverified:
            candidate_header = "[CANDIDATE OUTPUT (UNCORROBORATED MODEL GENERATION)]:\n"
            grounded_candidate = f"{candidate_header}{grounded_candidate}"

        inconclusive_notes: list[str] = []
        if verification_report and verification_report.results:
            inconclusive = [
                r for r in verification_report.results
                if r.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.NOT_CHECKED)
            ]
            if inconclusive:
                inconclusive_notes.append("\n\n[UNVERIFIED / INCONCLUSIVE CLAIMS DISCLOSURE]:")
                for r in inconclusive:
                    inconclusive_notes.append(f"- [{r.claim_id}] {r.claim_text} ({r.status.value}): {r.explanation}")

            contradicted = [
                r for r in verification_report.results
                if r.status == ClaimStatus.CONTRADICTED
            ]
            if contradicted:
                inconclusive_notes.append("\n\n[CONTRADICTED CLAIMS DISCLOSURE]:")
                for r in contradicted:
                    inconclusive_notes.append(f"- [{r.claim_id}] {r.claim_text} (CONTRADICTED): {r.explanation}")

        inconclusive_text = "\n".join(inconclusive_notes) if inconclusive_notes else ""
        return f"{banner}\n\n{grounded_candidate}{inconclusive_text}"

    @classmethod
    def _guard_unsupported_quantitative_claims(
        cls,
        judge_output: Any,
        verifications: list[ClaimVerificationResult] | None,
    ) -> None:
        """
        Deterministic guard ensuring unsupported or contradicted quantitative/numerical claims
        are not presented as verified facts in the Judge's synthesis.
        """
        if not verifications or not getattr(judge_output, "final_answer", None):
            return

        grounded_answer, updated_uncertainties = cls._enforce_grounding_in_answer(
            text=judge_output.final_answer,
            verifications=verifications,
            unresolved_uncertainties=judge_output.unresolved_uncertainties,
        )
        judge_output.final_answer = grounded_answer
        judge_output.unresolved_uncertainties = updated_uncertainties

    async def run_debate(
        self,
        query: str,
        max_rounds: int | None = None,
    ) -> QueryResponse:
        """
        Execute the full multi-agent debate workflow for the provided query.

        Parameters
        ----------
        query:
            The raw user query or topic.
        max_rounds:
            Optional request-level override for maximum debate rounds.

        Returns
        -------
        QueryResponse
            Validated structured response with debate rounds, trace, sources, and final verdict.
        """
        clean_query = query.strip()
        start_time = time.monotonic()
        self._llm_call_count = 0
        effective_max_rounds = self._determine_effective_max_rounds(max_rounds)

        trace: list[AgentTraceEntry] = []
        debate_rounds: list[DebateRound] = []
        surfaced_sources: dict[str, EvidenceItem] = {}  # Deduplicated by URL

        self._logger.info(
            "Initiating debate | query=%r max_rounds=%d (effective=%d) provider=%s",
            clean_query,
            max_rounds or self._settings.debate_max_rounds,
            effective_max_rounds,
            self._client.provider_name(),
        )

        # ──────────────────────────────────────────────────────────────────────────
        # Step 1: Planner
        # ──────────────────────────────────────────────────────────────────────────
        if not self._increment_call_count():
            return self._build_budget_error_response(clean_query, trace)

        planner_warning: str | None = None
        try:
            plan, _ = await self._planner.plan(clean_query)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.PLANNER,
                    round_number=0,
                    output=(
                        f"Core Question: {plan.core_question}\n"
                        f"Subtasks: {', '.join(plan.subtasks)}\n"
                        f"Constraints: {', '.join(plan.constraints) if plan.constraints else 'None'}"
                    ),
                )
            )
        except LLMValidationError as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.warning("Planner agent output validation failed: %s. Using fallback plan.", sanitized)
            planner_warning = (
                f"Degraded planning: Automated planning failed ({sanitized}); "
                "execution proceeded with a minimal fallback plan derived directly from the user query."
            )
            plan = TaskPlan(
                core_question=clean_query,
                subtasks=[f"Analyze and address: {clean_query}"],
                constraints=[],
                expected_format="structured_synthesis",
                evidence_needed=[clean_query],
            )
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.PLANNER,
                    round_number=0,
                    output=(
                        f"[WARNING: Automated planning failed ({sanitized}). Using minimal fallback plan derived directly from user query.]\n"
                        f"Core Question: {plan.core_question}\n"
                        f"Subtasks: {', '.join(plan.subtasks)}\n"
                        f"Constraints: None (fallback plan)"
                    ),
                )
            )
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.error("Planner agent failed: %s", sanitized, exc_info=True)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.PLANNER,
                    round_number=0,
                    output=f"Planning failed: {sanitized}",
                )
            )
            return QueryResponse(
                query=clean_query,
                status=DebateStatus.ERROR,
                rounds_completed=0,
                final_answer=None,
                subtasks=[],
                debate_rounds=[],
                trace=trace,
                sources=[],
                error_message=f"Planner agent failure: {sanitized}",
            )

        # ──────────────────────────────────────────────────────────────────────────
        # Step 2: Initial Candidates (Round 1)
        # ──────────────────────────────────────────────────────────────────────────
        cand_a: ExecutorOutput | None = None
        cand_b: ExecutorOutput | None = None

        if not self._increment_call_count():
            return self._build_budget_error_response(clean_query, trace, subtasks=plan.subtasks)

        try:
            cand_a, _ = await self._executor_a.execute(clean_query, plan)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.EXECUTOR_A,
                    round_number=1,
                    output=cand_a.answer,
                )
            )
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.error("Executor A failed on round 1: %s", sanitized, exc_info=True)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.EXECUTOR_A,
                    round_number=1,
                    output=f"Initial generation failed: {sanitized}",
                )
            )

        if not self._increment_call_count():
            return self._build_budget_error_response(clean_query, trace, subtasks=plan.subtasks)

        try:
            cand_b, _ = await self._executor_b.execute(clean_query, plan)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.EXECUTOR_B,
                    round_number=1,
                    output=cand_b.answer,
                )
            )
        except Exception as exc:
            sanitized = sanitize_error_message(str(exc))
            self._logger.error("Executor B failed on round 1: %s", sanitized, exc_info=True)
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.EXECUTOR_B,
                    round_number=1,
                    output=f"Initial generation failed: {sanitized}",
                )
            )

        if cand_a is None and cand_b is None:
            return QueryResponse(
                query=clean_query,
                status=DebateStatus.ERROR,
                rounds_completed=0,
                final_answer=None,
                subtasks=plan.subtasks,
                debate_rounds=[],
                trace=trace,
                sources=[],
                error_message="Both Executor A and Executor B failed to generate initial candidates.",
            )

        # If one executor failed, instantiate a fallback based on the working executor
        if cand_a is None and cand_b is not None:
            cand_a = ExecutorOutput(
                executor_id="executor_a",
                approach_summary="[FAILED - Single Candidate Mode: Executor A initial generation failed]",
                answer=f"[Notice: Executor A generation failed; no independent candidate produced. Single candidate provided by Executor B.]\n\n{cand_b.answer}",
                key_assumptions=cand_b.key_assumptions,
                claims_to_verify=cand_b.claims_to_verify,
            )
        elif cand_b is None and cand_a is not None:
            cand_b = ExecutorOutput(
                executor_id="executor_b",
                approach_summary="[FAILED - Single Candidate Mode: Executor B initial generation failed]",
                answer=f"[Notice: Executor B generation failed; no independent candidate produced. Single candidate provided by Executor A.]\n\n{cand_a.answer}",
                key_assumptions=cand_a.key_assumptions,
                claims_to_verify=cand_a.claims_to_verify,
            )

        assert cand_a is not None and cand_b is not None

        # ──────────────────────────────────────────────────────────────────────────
        # Step 3: Evidence Retrieval and Claim Verification
        # ──────────────────────────────────────────────────────────────────────────
        claims_to_verify = self._collect_claims(plan, cand_a, cand_b)
        evidence_items: list[EvidenceItem] = []
        retrieval_status = RetrievalStatus.SUCCESS
        claim_evidence_map: dict[str, list[EvidenceItem]] = {}
        claim_status_map: dict[str, RetrievalStatus] = {}
        verifications: list[ClaimVerificationResult] = []
        verification_report: ClaimVerificationReport | None = None

        if claims_to_verify:
            try:
                retrieval_status, evidence_items, claim_evidence_map, claim_status_map = await retrieve_for_claims(
                    claims=claims_to_verify,
                    retrieval_service=self._retrieval_service,
                    max_results_per_claim=min(2, self._settings.search_max_results),
                    max_total_sources=MAX_EVIDENCE_ITEMS,
                    concurrency_limit=3,
                )
                for item in evidence_items:
                    surfaced_sources[item.url] = item
            except Exception as exc:
                sanitized = sanitize_error_message(str(exc))
                self._logger.warning("Targeted retrieval call failed: %s", sanitized)
                retrieval_status = RetrievalStatus.ERROR
                claim_evidence_map = {}
                claim_status_map = {}

            # Verify claims with ClaimVerificationService
            if self._increment_call_count():
                try:
                    verification_report = await self._claim_verifier.verify_claims(
                        claims=claims_to_verify,
                        evidence=evidence_items,
                        retrieval_status=retrieval_status,
                        claim_evidence_map=claim_evidence_map,
                        claim_status_map=claim_status_map,
                    )
                    verifications = verification_report.results
                except Exception as exc:
                    sanitized = sanitize_error_message(str(exc))
                    self._logger.warning("Claim verification failed: %s", sanitized)
                    # Build a failed report so the frontend can display why verification failed
                    verification_report = ClaimVerificationReport(
                        report_status=VerificationReportStatus.FAILED,
                        results=[
                            ClaimVerificationResult(
                                claim_id=f"CLAIM-{i}",
                                claim_text=c,
                                status=ClaimStatus.NOT_CHECKED,
                                explanation="Verification service encountered an internal error.",
                                evidence=[],
                                limitations=f"Verification failed: {sanitized}",
                            )
                            for i, c in enumerate(claims_to_verify, 1)
                        ],
                        summary=f"Claim verification failed: {sanitized}",
                        evidence_count=len(evidence_items),
                        claims_checked=0,
                        claims_not_checked=len(claims_to_verify),
                    )
        else:
            self._logger.debug("No factual claims collected for retrieval.")
            # Still produce a NO_CLAIMS report so the frontend always has a report object
            verification_report = ClaimVerificationReport(
                report_status=VerificationReportStatus.NO_CLAIMS,
                results=[],
                summary="No factual claims were collected from the executor candidates for verification.",
                evidence_count=0,
                claims_checked=0,
                claims_not_checked=0,
            )

        # ──────────────────────────────────────────────────────────────────────────
        # Step 4: Bounded Debate Loop (Rounds 1 .. effective_max_rounds)
        # ──────────────────────────────────────────────────────────────────────────
        current_round = 1
        converged = False
        critic_output: Any = None

        while current_round <= effective_max_rounds:
            # 1. Critique
            if not self._increment_call_count():
                break

            try:
                critic_output, _ = await self._critic.critique(
                    query=clean_query,
                    plan=plan,
                    executor_a=cand_a,
                    executor_b=cand_b,
                    verifications=verifications,
                )
            except Exception as exc:
                sanitized = sanitize_error_message(str(exc))
                self._logger.warning("Critic failed in round %d: %s", current_round, sanitized)
                trace.append(
                    AgentTraceEntry(
                        agent=AgentRole.CRITIC,
                        round_number=current_round,
                        output=f"Critique generation failed: {sanitized}",
                        evidence=[],
                        unresolved_issues=[],
                    )
                )
                break

            # Record Critic trace
            unresolved = [i.explanation for i in critic_output.issues]
            trace.append(
                AgentTraceEntry(
                    agent=AgentRole.CRITIC,
                    round_number=current_round,
                    output=critic_output.overall_critique,
                    evidence=list(surfaced_sources.values()),
                    unresolved_issues=unresolved,
                )
            )

            # 2. Similarity and Convergence Evaluation
            sim_score = compute_answer_similarity(cand_a.answer, cand_b.answer)
            has_issues = self._has_material_issues(critic_output)

            single_candidate_mode = (
                "[FAILED - Single Candidate Mode" in (cand_a.approach_summary or "")
                or "[FAILED - Single Candidate Mode" in (cand_b.approach_summary or "")
            )

            # Converged if NO material issues remain, or similarity threshold met without critical flaws
            # (Disallow similarity-based early convergence if running in single-candidate fallback mode)
            if not has_issues or (
                not single_candidate_mode
                and sim_score >= self._settings.debate_similarity_threshold
                and not any(i.severity == CritiqueSeverity.HIGH for i in critic_output.issues)
            ):
                converged = True

            addressed_round_issues = list(set(
                getattr(cand_a, "addressed_issues", []) + getattr(cand_b, "addressed_issues", [])
            ))
            a_notes = getattr(cand_a, "revision_notes", None) or cand_a.approach_summary
            b_notes = getattr(cand_b, "revision_notes", None) or cand_b.approach_summary

            debate_rounds.append(
                DebateRound(
                    round_number=current_round,
                    executor_a_answer=cand_a.answer,
                    executor_b_answer=cand_b.answer,
                    critique=critic_output.overall_critique,
                    similarity_score=sim_score,
                    converged=converged,
                    issues=critic_output.issues if critic_output else [],
                    addressed_issues=addressed_round_issues,
                    executor_a_notes=a_notes,
                    executor_b_notes=b_notes,
                )
            )

            # 3. Check Stopping Conditions
            if converged:
                self._logger.info(
                    "Debate reached early convergence in round %d (similarity=%.3f, has_issues=%s, single_cand=%s)",
                    current_round,
                    sim_score,
                    has_issues,
                    single_candidate_mode,
                )
                break

            if current_round >= effective_max_rounds:
                self._logger.info("Debate reached maximum rounds budget (%d)", effective_max_rounds)
                break

            # 4. Revisions for Next Round
            next_round = current_round + 1
            prev_a_answer = cand_a.answer
            prev_b_answer = cand_b.answer

            # Executor A Revision
            if self._increment_call_count():
                critique_a = self._format_critique_for_executor("executor_a", critic_output, verifications)
                try:
                    revised_a, _ = await self._executor_a.revise(
                        query=clean_query,
                        plan=plan,
                        prior_output=cand_a,
                        critique_findings=critique_a,
                        evidence=list(surfaced_sources.values()),
                    )
                    cand_a = revised_a
                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.EXECUTOR_A,
                            round_number=next_round,
                            output=cand_a.answer,
                        )
                    )
                except Exception as exc:
                    sanitized = sanitize_error_message(str(exc))
                    self._logger.warning("Executor A revision failed in round %d: %s", next_round, sanitized)
                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.EXECUTOR_A,
                            round_number=next_round,
                            output=f"[Revision failed: {sanitized}. Preserving prior candidate.]",
                        )
                    )

            # Executor B Revision
            if self._increment_call_count():
                critique_b = self._format_critique_for_executor("executor_b", critic_output, verifications)
                try:
                    revised_b, _ = await self._executor_b.revise(
                        query=clean_query,
                        plan=plan,
                        prior_output=cand_b,
                        critique_findings=critique_b,
                        evidence=list(surfaced_sources.values()),
                    )
                    cand_b = revised_b
                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.EXECUTOR_B,
                            round_number=next_round,
                            output=cand_b.answer,
                        )
                    )
                except Exception as exc:
                    sanitized = sanitize_error_message(str(exc))
                    self._logger.warning("Executor B revision failed in round %d: %s", next_round, sanitized)
                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.EXECUTOR_B,
                            round_number=next_round,
                            output=f"[Revision failed: {sanitized}. Preserving prior candidate.]",
                        )
                    )

            # Avoid repeated identical revisions
            if cand_a.answer == prev_a_answer and cand_b.answer == prev_b_answer:
                self._logger.info("Executor answers unchanged after revision; stopping debate loop.")
                current_round = next_round
                break

            current_round = next_round

        rounds_completed = len(debate_rounds)

        # ──────────────────────────────────────────────────────────────────────────
        # Step 5: Judge Adjudication & Synthesis
        # ──────────────────────────────────────────────────────────────────────────
        final_answer: str | None = None
        status = DebateStatus.COMPLETED
        verdict_summary: str | None = None
        unresolved_uncertainties: list[str] = []
        pipeline_error: str | None = None

        if critic_output is None:
            # Fallback if critic never succeeded
            failure_reason = "Critic evaluation failed before completing any rounds; Judge adjudication was skipped"
            final_answer = self._format_unadjudicated_fallback(cand_a.answer, failure_reason, verification_report)
            status = DebateStatus.ERROR
            verdict_summary = "Critique step failed; adjudication was skipped and unadjudicated fallback candidate used without Judge review."
            unresolved_uncertainties = [
                "Unverified candidate: Critic did not complete critique; verification-grounded adjudication was not performed."
            ]
            if verification_report and verification_report.results:
                for r in verification_report.results:
                    if r.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.NOT_CHECKED):
                        unresolved_uncertainties.append(
                            f"Inconclusive claim [{r.claim_id}]: '{r.claim_text}' - {r.explanation}"
                        )
                    elif r.status == ClaimStatus.CONTRADICTED:
                        unresolved_uncertainties.append(
                            f"Contradicted claim [{r.claim_id}]: '{r.claim_text}' - {r.explanation}"
                        )
            pipeline_error = "Critic agent failed during debate; returned unadjudicated fallback candidate without critique."
        else:
            if self._increment_call_count():
                try:
                    judge_output, _ = await self._judge.judge(
                        query=clean_query,
                        plan=plan,
                        executor_a=cand_a,
                        executor_b=cand_b,
                        critic=critic_output,
                        evidence=list(surfaced_sources.values()),
                        verifications=verifications,
                    )

                    # Grounding check: enforce that cited URLs only refer to known surfaced sources
                    valid_urls = set(surfaced_sources.keys())
                    sanitized_citations = []
                    for cit in judge_output.evidence_citations:
                        urls = re.findall(r"https?://[^\s)\]]+", cit)
                        if any(u not in valid_urls for u in urls):
                            continue
                        sanitized_citations.append(cit)

                    # Grounding check: enforce that unsupported quantitative claims are disclosed in uncertainties
                    self._guard_unsupported_quantitative_claims(judge_output, verifications)

                    final_answer = judge_output.final_answer
                    verdict_summary = judge_output.verdict_summary or judge_output.adjudication_rationale
                    unresolved_uncertainties = judge_output.unresolved_uncertainties

                    # C7: Do not label answer fully evidence-grounded merely because a warning was appended
                    if any(v.status != ClaimStatus.SUPPORTED for v in (verifications or [])):
                        if verdict_summary and any(phrase in verdict_summary.lower() for phrase in ["fully evidence-grounded", "fully grounded", "fully verified"]):
                            verdict_summary = re.sub(
                                r"\bfully\s+(?:evidence-)?(?:grounded|verified)\b",
                                "partially grounded (with unverified/inconclusive claims)",
                                verdict_summary,
                                flags=re.IGNORECASE,
                            )

                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.JUDGE,
                            round_number=rounds_completed,
                            output=judge_output.final_answer,
                        )
                    )
                except Exception as exc:
                    sanitized = sanitize_error_message(str(exc))
                    self._logger.error("Judge adjudication failed: %s", sanitized, exc_info=True)
                    trace.append(
                        AgentTraceEntry(
                            agent=AgentRole.JUDGE,
                            round_number=rounds_completed,
                            output=f"Judge failed: {sanitized}. Returning unadjudicated fallback candidate.",
                        )
                    )
                    failure_reason = f"Judge adjudication failed: {sanitized}"
                    final_answer = self._format_unadjudicated_fallback(cand_a.answer, failure_reason, verification_report)
                    status = DebateStatus.ERROR if rounds_completed == 0 else DebateStatus.MAX_ROUNDS
                    verdict_summary = f"Judge adjudication failed ({sanitized}); using unadjudicated fallback candidate."
                    unresolved_uncertainties = [
                        "Adjudication fallback: Final answer synthesized directly from prior executor candidate without full Judge review."
                    ]
                    if verification_report and verification_report.results:
                        for r in verification_report.results:
                            if r.status in (ClaimStatus.INSUFFICIENT_EVIDENCE, ClaimStatus.NOT_CHECKED):
                                unresolved_uncertainties.append(
                                    f"Inconclusive claim [{r.claim_id}]: '{r.claim_text}' - {r.explanation}"
                                )
                            elif r.status == ClaimStatus.CONTRADICTED:
                                unresolved_uncertainties.append(
                                    f"Contradicted claim [{r.claim_id}]: '{r.claim_text}' - {r.explanation}"
                                )
                    pipeline_error = f"Judge adjudication failed: {sanitized}"
            else:
                failure_reason = "Adjudication skipped due to LLM call budget limits"
                final_answer = self._format_unadjudicated_fallback(cand_a.answer, failure_reason, verification_report)
                status = DebateStatus.ERROR if rounds_completed == 0 else DebateStatus.MAX_ROUNDS
                verdict_summary = "Adjudication skipped due to LLM call budget limits; unadjudicated fallback candidate returned."
                unresolved_uncertainties = ["Debate ended early due to LLM call budget."]
                pipeline_error = "Adjudication skipped due to LLM call budget."

        elapsed = time.monotonic() - start_time
        self._logger.info(
            "Debate concluded | rounds=%d status=%s elapsed=%.2fs llm_calls=%d sources=%d",
            rounds_completed,
            status.value,
            elapsed,
            self._llm_call_count,
            len(surfaced_sources),
        )

        if planner_warning and planner_warning not in unresolved_uncertainties:
            unresolved_uncertainties.insert(0, planner_warning)

        return QueryResponse(
            query=clean_query,
            status=status,
            rounds_completed=rounds_completed,
            final_answer=final_answer,
            subtasks=plan.subtasks,
            debate_rounds=debate_rounds,
            trace=trace,
            sources=list(surfaced_sources.values()),
            verification_report=verification_report,
            verdict_summary=verdict_summary,
            unresolved_uncertainties=unresolved_uncertainties,
            error_message=pipeline_error,
        )

    def _build_budget_error_response(
        self,
        query: str,
        trace: list[AgentTraceEntry],
        subtasks: list[str] | None = None,
    ) -> QueryResponse:
        """Construct response when LLM call budget is exceeded prematurely."""
        return QueryResponse(
            query=query,
            status=DebateStatus.ERROR,
            rounds_completed=0,
            final_answer=None,
            subtasks=subtasks or [],
            debate_rounds=[],
            trace=trace,
            sources=[],
            error_message=f"Debate terminated: maximum LLM call budget of {self._max_llm_calls} exceeded.",
        )

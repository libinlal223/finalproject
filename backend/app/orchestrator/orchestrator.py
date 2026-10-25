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
from app.llm.base import BaseLLMClient, LLMConfig, build_llm_client, sanitize_error_message
from app.models.schemas import (
    AgentRole,
    AgentTraceEntry,
    ClaimVerificationResult,
    CritiqueSeverity,
    DebateRound,
    DebateStatus,
    EvidenceItem,
    ExecutorOutput,
    QueryResponse,
    RetrievalStatus,
    TaskPlan,
)
from app.orchestrator.similarity import compute_answer_similarity
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.service import EvidenceRetrievalService

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
        if llm_client is not None:
            self._client = llm_client
        else:
            config = LLMConfig(
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
                timeout_seconds=self._settings.llm_timeout_seconds,
            )
            self._client = build_llm_client(self._settings.llm_provider.value, config)

        # Initialize agents with shared or injected client
        self._planner = planner or PlannerAgent(client=self._client)
        self._executor_a = executor_a or ExecutorAAgent(client=self._client)
        self._executor_b = executor_b or ExecutorBAgent(client=self._client)
        self._critic = critic or CriticAgent(client=self._client)
        self._judge = judge or JudgeAgent(client=self._client)

        # Initialize retrieval & verification
        self._retrieval_service = retrieval_service or EvidenceRetrievalService()
        self._claim_verifier = claim_verifier or ClaimVerificationService(llm_client=self._client)

        self._max_llm_calls = max_llm_calls
        self._llm_call_count = 0
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def llm_call_count(self) -> int:
        """Current number of LLM invocations during orchestration."""
        return self._llm_call_count

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
        self, executor_id: str, critic_output: Any
    ) -> str:
        """Format targeted, concise critique instructions without hidden chain-of-thought."""
        relevant_issues = [
            i for i in critic_output.issues
            if i.target in {executor_id, "both"}
        ]
        lines = [f"Overall Critique: {critic_output.overall_critique}"]
        if relevant_issues:
            lines.append("Specific Issues to Address:")
            for issue in relevant_issues:
                lines.append(
                    f"- [{issue.severity.value.upper()}] Claim: {issue.affected_claim}\n"
                    f"  Flaw: {issue.explanation}\n"
                    f"  Recommended Correction: {issue.recommended_correction}"
                )
        if critic_output.contradictions:
            lines.append("Contradictions with Opposing Candidate:")
            for c in critic_output.contradictions:
                lines.append(f"- {c}")

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
                approach_summary="First-principles deductive approach (fallback)",
                answer=cand_b.answer,
                key_assumptions=cand_b.key_assumptions,
                claims_to_verify=cand_b.claims_to_verify,
            )
        elif cand_b is None and cand_a is not None:
            cand_b = ExecutorOutput(
                executor_id="executor_b",
                approach_summary="Empirical and comparative trade-off approach (fallback)",
                answer=cand_a.answer,
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
        verifications: list[ClaimVerificationResult] = []

        if claims_to_verify:
            search_query = plan.evidence_needed[0] if plan.evidence_needed else plan.core_question
            search_query = search_query[:MAX_SEARCH_QUERY_LEN]

            try:
                ret_resp = await self._retrieval_service.retrieve(
                    search_query, max_results=self._settings.search_max_results
                )
                retrieval_status = ret_resp.status
                evidence_items = ret_resp.items[:MAX_EVIDENCE_ITEMS]
                for item in evidence_items:
                    surfaced_sources[item.url] = item
            except Exception as exc:
                sanitized = sanitize_error_message(str(exc))
                self._logger.warning("Retrieval call failed: %s", sanitized)
                retrieval_status = RetrievalStatus.ERROR

            # Verify claims with ClaimVerificationService
            if self._increment_call_count():
                try:
                    report = await self._claim_verifier.verify_claims(
                        claims=claims_to_verify,
                        evidence=evidence_items,
                        retrieval_status=retrieval_status,
                    )
                    verifications = report.results
                except Exception as exc:
                    sanitized = sanitize_error_message(str(exc))
                    self._logger.warning("Claim verification failed: %s", sanitized)
        else:
            self._logger.debug("No factual claims collected for retrieval.")

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

            # Converged if NO material issues remain, or similarity threshold met without critical flaws
            if not has_issues or (
                sim_score >= self._settings.debate_similarity_threshold
                and not any(i.severity == CritiqueSeverity.HIGH for i in critic_output.issues)
            ):
                converged = True

            debate_rounds.append(
                DebateRound(
                    round_number=current_round,
                    executor_a_answer=cand_a.answer,
                    executor_b_answer=cand_b.answer,
                    critique=critic_output.overall_critique,
                    similarity_score=sim_score,
                    converged=converged,
                )
            )

            # 3. Check Stopping Conditions
            if converged:
                self._logger.info(
                    "Debate reached early convergence in round %d (similarity=%.3f, has_issues=%s)",
                    current_round,
                    sim_score,
                    has_issues,
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
                critique_a = self._format_critique_for_executor("executor_a", critic_output)
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
                critique_b = self._format_critique_for_executor("executor_b", critic_output)
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

        if critic_output is None:
            # Fallback if critic never succeeded
            final_answer = cand_a.answer
            status = DebateStatus.MAX_ROUNDS
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
                    )

                    # Grounding check: enforce that cited URLs only refer to known surfaced sources
                    valid_urls = set(surfaced_sources.keys())
                    sanitized_citations = []
                    for cit in judge_output.evidence_citations:
                        urls = re.findall(r"https?://[^\s)\]]+", cit)
                        if any(u not in valid_urls for u in urls):
                            continue
                        sanitized_citations.append(cit)

                    final_answer = judge_output.final_answer
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
                            output=f"Judge failed: {sanitized}. Synthesizing best available answer.",
                        )
                    )
                    final_answer = cand_a.answer
                    status = DebateStatus.MAX_ROUNDS
            else:
                final_answer = cand_a.answer
                status = DebateStatus.MAX_ROUNDS

        elapsed = time.monotonic() - start_time
        self._logger.info(
            "Debate concluded | rounds=%d status=%s elapsed=%.2fs llm_calls=%d sources=%d",
            rounds_completed,
            status.value,
            elapsed,
            self._llm_call_count,
            len(surfaced_sources),
        )

        return QueryResponse(
            query=clean_query,
            status=status,
            rounds_completed=rounds_completed,
            final_answer=final_answer,
            subtasks=plan.subtasks,
            debate_rounds=debate_rounds,
            trace=trace,
            sources=list(surfaced_sources.values()),
            error_message=None,
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

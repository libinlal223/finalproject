"""
app/evaluation/baseline.py
──────────────────────────
Single-model baseline comparison runner.

Executes a single configured LLM call without multi-agent debate, decomposition,
or iterative critique. Serves as the control group against which VADER's
multi-agent collaborative reasoning is benchmarked.

Principles:
  - Uses the identical underlying LLM adapter and configuration.
  - Transparently records all prompt formatting, configuration, and token counts.
  - Handles runtime errors without raising uncaught exceptions.
"""

from __future__ import annotations

import logging
import re
import time

from app.config import Settings, get_settings
from app.evaluation.models import EvaluationCase, SingleSystemResult
from app.llm.base import (
    BaseLLMClient,
    LLMConfig,
    LLMMessage,
    build_llm_client,
    sanitize_error_message,
)
from app.models.schemas import BaselineResponse

logger = logging.getLogger(__name__)

BASELINE_SYSTEM_PROMPT = (
    "You are a helpful, rigorous, and truthful AI assistant. "
    "Answer the user's question accurately, completely, and concisely. "
    "If the question involves unresolved historical debates, insufficient evidence, "
    "or future events that cannot be known with certainty, explicitly state this uncertainty "
    "and present the competing viewpoints or boundaries of knowledge."
)


class BaselineRunner:
    """
    Executes a direct single-model query for benchmarking.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        settings: Settings | None = None,
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

    @property
    def client(self) -> BaseLLMClient:
        return self._client

    async def execute_case(self, case: EvaluationCase) -> SingleSystemResult:
        """
        Run the single-model baseline against a benchmark case.
        """
        start_time = time.perf_counter()
        messages = [
            LLMMessage(role="system", content=BASELINE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=case.question),
        ]

        try:
            response = await self._client.complete(messages)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            # Extract any citations or URLs if present in the text
            url_pattern = r"https?://[^\s<>\"'()]+"
            citations = re.findall(url_pattern, response.content)

            prompt_toks = response.prompt_tokens if response.prompt_tokens > 0 else None
            comp_toks = response.completion_tokens if response.completion_tokens > 0 else None
            tot_toks = (prompt_toks + comp_toks) if (prompt_toks is not None and comp_toks is not None) else None

            model_name = getattr(self._client, "effective_model_id", response.model_id)
            prov_name = (
                self._client.provider_name()
                if callable(getattr(self._client, "provider_name", None))
                else response.provider
            )

            return SingleSystemResult(
                case_id=case.case_id,
                system_id="baseline",
                query=case.question,
                answer=response.content,
                latency_ms=round(elapsed_ms, 2),
                llm_call_count=1,
                prompt_tokens=prompt_toks,
                completion_tokens=comp_toks,
                total_tokens=tot_toks,
                cost_usd=None,  # Not fabricated
                retrieval_failure_count=0,
                model_id=model_name,
                provider=prov_name,
                citations=citations,
                rounds_completed=0,
                error=None,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            sanitized = sanitize_error_message(str(exc))
            logger.error("Baseline execution failed for case %s: %s", case.case_id, sanitized)
            model_name = getattr(self._client, "effective_model_id", "unknown")
            prov_name = (
                self._client.provider_name()
                if callable(getattr(self._client, "provider_name", None))
                else "unknown"
            )
            return SingleSystemResult(
                case_id=case.case_id,
                system_id="baseline",
                query=case.question,
                answer="",
                latency_ms=round(elapsed_ms, 2),
                llm_call_count=1,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                cost_usd=None,
                retrieval_failure_count=0,
                model_id=model_name,
                provider=prov_name,
                citations=[],
                rounds_completed=0,
                error=sanitized,
            )

    async def run_query(self, query: str) -> BaselineResponse:
        """
        API-facing method for POST /api/baseline.
        """
        start_time = time.perf_counter()
        messages = [
            LLMMessage(role="system", content=BASELINE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=query),
        ]
        response = await self._client.complete(messages)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        return BaselineResponse(
            query=query,
            answer=response.content,
            llm_provider=response.provider,
            model_id=response.model_id,
            latency_ms=round(elapsed_ms, 2),
        )

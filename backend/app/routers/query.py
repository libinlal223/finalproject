"""
app/routers/query.py
────────────────────
Multi-agent debate endpoint: POST /api/query
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.config import Settings, get_settings
from app.llm.base import BaseLLMClient, LLMConfig, build_llm_client
from app.models.schemas import QueryRequest, QueryResponse
from app.orchestrator.orchestrator import DebateOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Run multi-agent debate",
    description=(
        "Submits a user question to the VADER multi-agent system. "
        "The Planner decomposes the problem, Executors generate distinct candidate answers, "
        "evidence is retrieved and verified, the Critic evaluates flaws and contradictions, "
        "revisions are bounded by max_rounds, and the Judge adjudicates the final answer."
    ),
    tags=["debate"],
)
async def submit_query(
    request: QueryRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> QueryResponse:
    llm_client: BaseLLMClient | None = None
    if request.llm_provider:
        provider_name = request.llm_provider.value
        model_override = (
            settings.llm_model_id
            or (settings.groq_model_id if provider_name == "groq" else "")
            or (settings.gemini_model_id if provider_name == "gemini" else "")
        )
        config = LLMConfig(
            model_id=model_override,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout_seconds=settings.llm_timeout_seconds,
        )
        llm_client = build_llm_client(provider_name, config)

    orchestrator = DebateOrchestrator(
        llm_client=llm_client,
        settings=settings,
    )
    return await orchestrator.run_debate(
        query=request.query,
        max_rounds=request.max_rounds,
    )

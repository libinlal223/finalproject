"""
app/routers/baseline.py
───────────────────────
Single-model baseline endpoint: POST /api/baseline
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, status

from app.config import Settings, get_settings
from app.evaluation.baseline import BaselineRunner
from app.llm.base import LLMConfig, build_llm_client
from app.models.schemas import BaselineRequest, BaselineResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/baseline",
    response_model=BaselineResponse,
    status_code=status.HTTP_200_OK,
    summary="Run single-model baseline",
    description=(
        "Submits a question to a single configured LLM call without multi-agent "
        "debate or iterative critique, providing a control baseline."
    ),
    tags=["baseline"],
)
async def submit_baseline(
    request: BaselineRequest,
    settings: Annotated[Settings, Depends(get_settings)],
) -> BaselineResponse:
    """Execute single-model baseline query."""
    provider_name = (
        request.llm_provider.value
        if request.llm_provider
        else settings.llm_provider.value
    )
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
    runner = BaselineRunner(llm_client=llm_client, settings=settings)
    return await runner.run_query(request.query)

"""
app/routers/health.py
──────────────────────
Health-check endpoint: GET /api/health

The health endpoint:
  - Always returns 200 when the service is reachable.
  - Reports the active LLM provider and whether its key is configured.
  - Does NOT make any LLM API calls (missing keys do not break health checks).
  - Is used by tests, load balancers and deployment pipelines.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthResponse

router = APIRouter()

# Application version — bump this manually or wire to pyproject.toml / git tag
_APP_VERSION = "0.1.0"


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns service status, runtime environment, active LLM provider and "
        "whether an API key is configured.  Always returns 200 when the service "
        "is reachable — a missing API key is reported in the body, not as an error."
    ),
    tags=["meta"],
)
async def health_check() -> HealthResponse:
    """Return service health information."""
    settings = get_settings()

    return HealthResponse(
        status="ok",
        version=_APP_VERSION,
        environment=settings.app_env.value,
        llm_provider=settings.llm_provider.value,
        llm_key_present=settings.active_key_present(),
    )

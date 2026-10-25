"""
tests/conftest.py
──────────────────
Shared pytest fixtures for the VADER backend test suite.

All fixtures that interact with FastAPI use the async HTTPX test client.
All LLM interactions are intercepted by the mock adapter — no real API
keys are required to run this suite.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

# ─── Environment setup ────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _set_test_environment(monkeypatch):
    """
    Force test-safe environment variables before every test.

    This ensures:
      - LLM_PROVIDER=mock    → no real API calls are attempted
      - APP_ENV=test          → debug mode, predictable behaviour
      - No real API keys are needed
    """
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("APP_DEBUG", "true")
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "http://localhost:5173")

    # Clear the settings singleton cache so each test sees fresh env vars
    from app.config import get_settings
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ─── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture
async def client():
    """
    Async HTTPX test client wired to the FastAPI app.

    Uses ASGITransport so no real network connection is made.
    """
    from app.main import create_app
    app = create_app()

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac

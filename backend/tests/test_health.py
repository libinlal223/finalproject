"""
tests/test_health.py
──────────────────────
Tests for GET /api/health.

Acceptance criteria verified here:
  ✓ Endpoint returns 200 with correct Content-Type
  ✓ Response body matches HealthResponse schema
  ✓ Missing API keys do NOT prevent a successful health response
  ✓ Provider name reported matches the active config
  ✓ llm_key_present = True when provider = mock (no key needed)
  ✓ llm_key_present = False when a real provider has no key set
"""

from __future__ import annotations


class TestHealthEndpoint:
    """GET /api/health — basic contract tests."""

    async def test_health_returns_200(self, client):
        """Health endpoint must return HTTP 200."""
        response = await client.get("/api/health")
        assert response.status_code == 200, response.text

    async def test_health_content_type_is_json(self, client):
        """Response must be JSON."""
        response = await client.get("/api/health")
        assert "application/json" in response.headers["content-type"]

    async def test_health_response_schema(self, client):
        """Response body must include all required HealthResponse fields."""
        response = await client.get("/api/health")
        body = response.json()

        assert body["status"] == "ok"
        assert "version" in body
        assert "environment" in body
        assert "timestamp" in body
        assert "llm_provider" in body
        assert "llm_key_present" in body

    async def test_health_environment_is_test(self, client):
        """In the test environment, environment field must be 'test'."""
        response = await client.get("/api/health")
        assert response.json()["environment"] == "test"

    async def test_health_mock_provider_key_present_true(self, client):
        """Mock provider never needs a key — llm_key_present must be True."""
        response = await client.get("/api/health")
        body = response.json()
        assert body["llm_provider"] == "mock"
        assert body["llm_key_present"] is True

    async def test_health_groq_without_key_reports_missing(self, monkeypatch):
        """
        When Groq is active but no key is set, llm_key_present must be False.
        Health must still return 200 (missing key ≠ service unavailable).
        """
        monkeypatch.setenv("LLM_PROVIDER", "groq")
        monkeypatch.setenv("GROQ_API_KEY", "")

        from app.config import get_settings
        get_settings.cache_clear()

        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            response = await ac.get("/api/health")

        assert response.status_code == 200
        body = response.json()
        assert body["llm_provider"] == "groq"
        assert body["llm_key_present"] is False

    async def test_health_gemini_without_key_reports_missing(self, monkeypatch):
        """Same as above but for the Gemini provider."""
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("GEMINI_API_KEY", "")

        from app.config import get_settings
        get_settings.cache_clear()

        from httpx import ASGITransport, AsyncClient

        from app.main import create_app

        app = create_app()
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as ac:
            response = await ac.get("/api/health")

        assert response.status_code == 200
        body = response.json()
        assert body["llm_provider"] == "gemini"
        assert body["llm_key_present"] is False

    async def test_health_not_found_returns_404(self, client):
        """Unknown paths should return 404."""
        response = await client.get("/api/nonexistent")
        assert response.status_code == 404

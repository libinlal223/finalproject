"""
app/retrieval/ddg_provider.py
─────────────────────────────
DuckDuckGo search provider implementation for VADER.

Architecture:
  - Uses the duckduckgo_search library without requiring an external API key.
  - Runs the synchronous DDGS call in a background thread via asyncio.to_thread
    to ensure the FastAPI async event loop is never blocked.
  - Enforces explicit timeouts via asyncio.wait_for.
  - Validates all returned links with is_valid_url and canonicalizes them to
    eliminate tracking tokens and duplicate results.
  - Truncates snippets to prevent memory bloat or overly large downstream prompts.

Limitations & Availability:
  - DuckDuckGo public search endpoints are subject to IP-based rate limiting (HTTP 429)
    under continuous or bursty requests.
  - Availability depends on external network connectivity and DuckDuckGo endpoint stability.
  - When unavailable or rate-limited, the provider returns a structured RetrievalResponse
    with status=RATE_LIMITED or ERROR, allowing callers to handle the degradation gracefully.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

from app.llm.base import sanitize_error_message
from app.models.schemas import EvidenceItem, RetrievalResponse, RetrievalStatus
from app.retrieval.base import BaseSearchProvider
from app.retrieval.url_normalizer import canonicalize_url, is_valid_url

logger = logging.getLogger(__name__)

# Max snippet length to keep downstream prompts bounded
_MAX_SNIPPET_LENGTH = 1000


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """
    Asynchronous DuckDuckGo search provider.
    """

    def provider_name(self) -> str:
        return "ddg"

    def _sync_search(self, query: str, max_results: int) -> list[dict]:
        """Perform the blocking DuckDuckGo query inside a worker thread."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = ddgs.text(
                    keywords=query,
                    max_results=max_results,
                    safesearch="moderate" if self._config.safe_search else "off",
                )
                return list(results) if results else []
        except Exception:
            raise

    async def search(
        self, query: str, max_results: int | None = None
    ) -> RetrievalResponse:
        t_start = time.monotonic()
        clean_query = query.strip()
        limit = max_results or self._config.max_results

        if not clean_query:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.EMPTY,
                items=[],
                total_results=0,
                latency_ms=0.0,
            )

        try:
            from duckduckgo_search.exceptions import (
                DuckDuckGoSearchException,
                RatelimitException,
                TimeoutException,
            )
        except ImportError:
            # Fallback if package structure differs
            RatelimitException = type("RatelimitException", (Exception,), {})
            TimeoutException = type("TimeoutException", (Exception,), {})
            DuckDuckGoSearchException = Exception

        raw_results: list[dict] = []
        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(self._sync_search, clean_query, limit * 2),
                timeout=float(self._config.timeout_seconds),
            )
        except RatelimitException as exc:
            latency_ms = (time.monotonic() - t_start) * 1000
            sanitized = sanitize_error_message(str(exc))
            self._logger.warning("DuckDuckGo rate limit hit: %s", sanitized)
            return RetrievalResponse(
                query=clean_query,
                status=RetrievalStatus.RATE_LIMITED,
                items=[],
                error_message=f"Search provider rate limit exceeded: {sanitized}",
                latency_ms=latency_ms,
            )
        except (TimeoutException, TimeoutError):
            latency_ms = (time.monotonic() - t_start) * 1000
            self._logger.warning("DuckDuckGo search timed out after %ds", self._config.timeout_seconds)
            return RetrievalResponse(
                query=clean_query,
                status=RetrievalStatus.TIMEOUT,
                items=[],
                error_message=f"Search request timed out after {self._config.timeout_seconds}s.",
                latency_ms=latency_ms,
            )
        except (DuckDuckGoSearchException, Exception) as exc:
            latency_ms = (time.monotonic() - t_start) * 1000
            sanitized = sanitize_error_message(str(exc))
            self._logger.error("DuckDuckGo search failed: %s", sanitized)
            return RetrievalResponse(
                query=clean_query,
                status=RetrievalStatus.ERROR,
                items=[],
                error_message=f"Search provider failure: {sanitized}",
                latency_ms=latency_ms,
            )

        # Process, validate, and deduplicate returned items
        seen_urls: set[str] = set()
        items: list[EvidenceItem] = []

        for record in raw_results:
            if not isinstance(record, dict):
                continue

            raw_url = record.get("href") or record.get("link") or record.get("url")
            if not raw_url or not is_valid_url(raw_url):
                continue

            canonical = canonicalize_url(raw_url)
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)

            title = str(record.get("title") or "Untitled Source").strip()
            snippet = str(record.get("body") or record.get("snippet") or "").strip()
            if len(snippet) > _MAX_SNIPPET_LENGTH:
                snippet = snippet[:_MAX_SNIPPET_LENGTH] + "..."

            domain = urlparse(canonical).netloc.lower()

            items.append(
                EvidenceItem(
                    title=title,
                    url=canonical,
                    snippet=snippet,
                    domain=domain,
                    source="duckduckgo",
                )
            )

            if len(items) >= limit:
                break

        latency_ms = (time.monotonic() - t_start) * 1000

        if not items:
            return RetrievalResponse(
                query=clean_query,
                status=RetrievalStatus.EMPTY,
                items=[],
                total_results=0,
                latency_ms=latency_ms,
            )

        return RetrievalResponse(
            query=clean_query,
            status=RetrievalStatus.SUCCESS,
            items=items,
            total_results=len(items),
            latency_ms=latency_ms,
        )

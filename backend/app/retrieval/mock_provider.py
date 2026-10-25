"""
app/retrieval/mock_provider.py
──────────────────────────────
Deterministic offline mock search provider for unit testing and local development.
"""

from __future__ import annotations

import asyncio
import time
from urllib.parse import urlparse

from app.models.schemas import EvidenceItem, RetrievalResponse, RetrievalStatus
from app.retrieval.base import BaseSearchProvider
from app.retrieval.url_normalizer import canonicalize_url, is_valid_url


class MockSearchProvider(BaseSearchProvider):
    """
    Mock search provider producing predictable evidence items without internet access.
    """

    def provider_name(self) -> str:
        return "mock"

    @staticmethod
    def sample_evidence(query: str, count: int = 3) -> list[EvidenceItem]:
        """Generate deterministic sample evidence items for a given query."""
        results = []
        for i in range(1, count + 1):
            url = f"https://example.org/docs/evidence-{i}"
            results.append(
                EvidenceItem(
                    title=f"Empirical Study #{i} on {query}",
                    url=url,
                    snippet=f"Document #{i} confirms empirical findings and key trade-offs related to {query}.",
                    domain="example.org",
                    source="mock",
                    score=round(1.0 - (i * 0.1), 2),
                )
            )
        return results

    async def search(
        self, query: str, max_results: int | None = None
    ) -> RetrievalResponse:
        t_start = time.monotonic()
        limit = max_results or self._config.max_results
        extra = self._config.extra or {}

        # 1. Simulate configurable errors and conditions
        simulate = extra.get("simulate")
        if simulate == "rate_limit":
            latency_ms = (time.monotonic() - t_start) * 1000
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.RATE_LIMITED,
                items=[],
                error_message="Mock search rate limit exceeded.",
                latency_ms=latency_ms,
            )
        if simulate == "timeout":
            latency_ms = (time.monotonic() - t_start) * 1000
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.TIMEOUT,
                items=[],
                error_message="Mock search request timed out.",
                latency_ms=latency_ms,
            )
        if simulate == "error":
            latency_ms = (time.monotonic() - t_start) * 1000
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.ERROR,
                items=[],
                error_message="Mock search provider internal failure.",
                latency_ms=latency_ms,
            )
        if simulate == "empty":
            latency_ms = (time.monotonic() - t_start) * 1000
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.EMPTY,
                items=[],
                total_results=0,
                latency_ms=latency_ms,
            )

        # 2. Simulate latency
        delay_ms = float(extra.get("simulate_latency_ms", 0.0))
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)

        # 3. Gather raw items (from mock_items or sample generation)
        raw_items: list[EvidenceItem] = extra.get(
            "mock_items", self.sample_evidence(query, count=limit)
        )

        # 4. Validate and deduplicate by canonical URL
        seen_urls: set[str] = set()
        valid_items: list[EvidenceItem] = []

        for item in raw_items:
            raw_url = item.url
            if not is_valid_url(raw_url):
                continue
            canonical = canonicalize_url(raw_url)
            if canonical in seen_urls:
                continue
            seen_urls.add(canonical)

            domain = item.domain or urlparse(canonical).netloc.lower()
            valid_items.append(
                EvidenceItem(
                    title=item.title.strip() or "Untitled Source",
                    url=canonical,
                    snippet=item.snippet.strip(),
                    domain=domain,
                    source=item.source or "mock",
                    score=item.score,
                )
            )
            if len(valid_items) >= limit:
                break

        latency_ms = (time.monotonic() - t_start) * 1000

        if not valid_items:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.EMPTY,
                items=[],
                total_results=0,
                latency_ms=latency_ms,
            )

        return RetrievalResponse(
            query=query,
            status=RetrievalStatus.SUCCESS,
            items=valid_items,
            total_results=len(valid_items),
            latency_ms=latency_ms,
        )

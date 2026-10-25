"""
app/retrieval/service.py
────────────────────────
Evidence retrieval service orchestrating search providers.
"""

from __future__ import annotations

import logging

from app.config import get_settings
from app.models.schemas import RetrievalResponse, RetrievalStatus
from app.retrieval.base import BaseSearchProvider, SearchConfig
from app.retrieval.ddg_provider import DuckDuckGoSearchProvider
from app.retrieval.mock_provider import MockSearchProvider

logger = logging.getLogger(__name__)


def build_search_provider(
    provider_name: str, config: SearchConfig | None = None
) -> BaseSearchProvider:
    """
    Factory function to instantiate the requested search provider.

    Parameters
    ----------
    provider_name:
        'ddg' (DuckDuckGo) or 'mock'.
    config:
        Optional SearchConfig instance.
    """
    name = provider_name.lower().strip()
    if name in {"ddg", "duckduckgo"}:
        return DuckDuckGoSearchProvider(config)
    if name == "mock":
        return MockSearchProvider(config)

    raise ValueError(
        f"Unknown search provider: {provider_name!r}. "
        "Supported providers: 'ddg', 'mock'."
    )


class EvidenceRetrievalService:
    """
    High-level evidence retrieval service used by the Critic and debate orchestrators.
    """

    def __init__(
        self,
        provider: BaseSearchProvider | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        settings = get_settings()
        self._config = config or SearchConfig(
            max_results=settings.search_max_results,
            timeout_seconds=settings.search_timeout_seconds,
        )

        if provider is not None:
            self._provider = provider
        else:
            self._provider = build_search_provider(
                settings.search_provider, self._config
            )

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def provider(self) -> BaseSearchProvider:
        return self._provider

    async def retrieve(
        self, query: str, max_results: int | None = None
    ) -> RetrievalResponse:
        """
        Execute an evidence search for the given query.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Optional override for the number of results to return.

        Returns
        -------
        RetrievalResponse
            Validated structured response with status, items, and metadata.
        """
        clean_query = query.strip()
        if not clean_query:
            return RetrievalResponse(
                query=query,
                status=RetrievalStatus.EMPTY,
                items=[],
                total_results=0,
                latency_ms=0.0,
            )

        self._logger.debug(
            "Executing evidence retrieval | provider=%s query=%r",
            self._provider.provider_name(),
            clean_query,
        )

        return await self._provider.search(clean_query, max_results=max_results)

"""
app/retrieval/base.py
────────────────────
Abstract search provider interface and configuration for evidence retrieval.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from app.models.schemas import RetrievalResponse

logger = logging.getLogger(__name__)


@dataclass
class SearchConfig:
    """Configuration parameters for a search provider."""

    max_results: int = 5
    timeout_seconds: int = 10
    safe_search: bool = True
    extra: dict[str, Any] | None = None


class BaseSearchProvider(ABC):
    """
    Abstract interface for search providers (e.g. DuckDuckGo, Mock).
    """

    def __init__(self, config: SearchConfig | None = None) -> None:
        self._config = config or SearchConfig()
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @property
    def config(self) -> SearchConfig:
        return self._config

    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider's machine name (e.g. 'ddg', 'mock')."""
        ...

    @abstractmethod
    async def search(
        self, query: str, max_results: int | None = None
    ) -> RetrievalResponse:
        """
        Execute an asynchronous search query and return a validated RetrievalResponse.
        """
        ...

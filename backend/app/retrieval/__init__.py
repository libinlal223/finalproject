"""
app/retrieval
─────────────
Evidence retrieval and claim verification layer for VADER.
"""

from app.retrieval.base import BaseSearchProvider, SearchConfig
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.ddg_provider import DuckDuckGoSearchProvider
from app.retrieval.mock_provider import MockSearchProvider
from app.retrieval.service import EvidenceRetrievalService, build_search_provider
from app.retrieval.url_normalizer import canonicalize_url, is_valid_url

__all__ = [
    "BaseSearchProvider",
    "SearchConfig",
    "DuckDuckGoSearchProvider",
    "MockSearchProvider",
    "EvidenceRetrievalService",
    "ClaimVerificationService",
    "build_search_provider",
    "is_valid_url",
    "canonicalize_url",
]

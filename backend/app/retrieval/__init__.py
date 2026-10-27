"""
app/retrieval
─────────────
Evidence retrieval and claim verification layer for VADER.
"""

from app.retrieval.base import BaseSearchProvider, SearchConfig
from app.retrieval.claim_verifier import ClaimVerificationService
from app.retrieval.ddg_provider import DuckDuckGoSearchProvider
from app.retrieval.passage_extractor import extract_passage_for_claim, is_safe_external_url
from app.retrieval.service import EvidenceRetrievalService, build_search_provider
from app.retrieval.targeted_retrieval import derive_claim_query, retrieve_for_claims
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
    "derive_claim_query",
    "retrieve_for_claims",
    "extract_passage_for_claim",
    "is_safe_external_url",
]

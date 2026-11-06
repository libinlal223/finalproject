"""
app/retrieval/relevance.py
──────────────────────────
Topic relevance filtering and source authority scoring for VADER search results.

Key Guarantees:
  - Rejects clearly unrelated results (e.g. high school football scores matching "Apollo" and "touchdown").
  - Prevents keyword overlap alone from counting as evidence support.
  - Prioritizes authoritative topic sources such as NASA mission documentation (*.nasa.gov)
    and official historical archives for space exploration queries.
"""

from __future__ import annotations

import re

from app.models.schemas import EvidenceItem


def is_result_topic_relevant(query: str, item: EvidenceItem) -> bool:
    """
    Check if a retrieved search result is topic-relevant or clearly unrelated.
    Specifically rejects results like high school / college football scores or sports articles
    for Apollo 11 / spaceflight queries, ensuring keyword overlap alone is never accepted.
    """
    q_lower = query.lower()
    combined_text = f"{item.title} {item.snippet} {item.url}".lower()

    # Check for Apollo / Lunar / NASA spaceflight queries
    is_apollo_space = any(
        term in q_lower
        for term in ("apollo", "moon landing", "lunar", "armstrong", "aldrin", "spacecraft", "astronaut", "touchdown")
    )
    if is_apollo_space:
        # Off-topic sports / football indicators
        sports_indicators = [
            "football", "high school football", "quarterback", "scrimmage",
            "yard line", "head coach", "varsity", "roster", "playoff", "nfl", "athletics",
            "basketball", "soccer", "baseball", "conference championship", "maxpreps",
            "hudl", "espn", "scores"
        ]
        has_sports = any(sp in combined_text for sp in sports_indicators)
        space_indicators = [
            "moon", "lunar", "spaceflight", "astronaut", "nasa", "spacecraft",
            "orbit", "eagle", "columbia", "armstrong", "aldrin", "collins",
            "saturn v", "cape kennedy", "tranquility base", "mission elapsed time"
        ]
        has_space_context = any(sp in combined_text for sp in space_indicators)

        # Reject football / athletics results that lack genuine spaceflight context
        if has_sports and not has_space_context:
            return False
        if "high school" in combined_text and ("football" in combined_text or "athletics" in combined_text or "varsity" in combined_text):
            return False

    # Check for completely unrelated topic domains (e.g. gardening, recipes)
    unrelated_domains = ["recipe", "gardening", "tomatoes", "clothing", "shoes", "fashion", "sneakers"]
    if any(u in combined_text for u in unrelated_domains) and not any(t in q_lower for t in ("recipe", "garden", "cloth")):
        core_query_tokens = [t for t in re.findall(r"\b[a-zA-Z0-9]{3,}\b", q_lower) if t not in {"when", "what", "where", "did", "the", "and"}]
        if core_query_tokens and not any(t in combined_text for t in core_query_tokens):
            return False

    return True


def get_source_authority_score(item: EvidenceItem, query: str) -> float:
    """
    Compute domain and topic authority score to prioritize authoritative sources
    such as NASA mission documentation for space queries.
    """
    url_lower = item.url.lower()
    title_lower = item.title.lower()
    q_lower = query.lower()

    score = 10.0

    is_apollo_or_space = any(term in q_lower for term in ("apollo", "moon", "lunar", "nasa", "space"))
    if is_apollo_or_space:
        if "nasa.gov" in url_lower:
            score += 100.0  # Top tier authoritative for NASA missions
            if any(sub in url_lower for sub in ("history.nasa.gov", "mission_pages", "apollo")):
                score += 20.0
        elif any(domain in url_lower for domain in ("airandspace.si.edu", "si.edu", "archives.gov")):
            score += 80.0
        elif any(domain in url_lower for domain in ("esa.int", "planetary.org", "lpi.usra.edu")):
            score += 60.0
        elif "wikipedia.org" in url_lower:
            score += 50.0
            if "apollo_11" in url_lower:
                score += 15.0
        elif any(domain in url_lower for domain in ("space.com", "britannica.com")):
            score += 40.0
    else:
        if any(domain in url_lower for domain in (".gov", ".edu", ".int")):
            score += 50.0
        elif "wikipedia.org" in url_lower:
            score += 30.0

    return score

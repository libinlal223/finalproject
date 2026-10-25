"""
app/orchestrator/similarity.py
──────────────────────────────
Lightweight TF-IDF cosine similarity computation between executor candidate answers.

Used strictly as an optional convergence hint / similarity signal,
NOT as a proxy for factual correctness.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_answer_similarity(text_a: str, text_b: str) -> float:
    """
    Compute normalized cosine similarity in [0.0, 1.0] between two texts using TF-IDF.

    Returns 0.0 if either text is empty or contains no extractable words.
    """
    clean_a = text_a.strip()
    clean_b = text_b.strip()

    if not clean_a or not clean_b:
        return 0.0

    if clean_a.lower() == clean_b.lower():
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform([clean_a, clean_b])
        sim_val = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
        return max(0.0, min(1.0, round(sim_val, 4)))
    except Exception as exc:
        logger.debug("TF-IDF similarity calculation failed: %s", exc)
        # Fallback to simple Jaccard word set similarity if TF-IDF fails
        words_a = set(clean_a.lower().split())
        words_b = set(clean_b.lower().split())
        union = words_a | words_b
        if not union:
            return 0.0
        return round(len(words_a & words_b) / len(union), 4)

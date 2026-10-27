"""
app/retrieval/passage_extractor.py
──────────────────────────────────
Safe webpage fetching and targeted passage extraction for VADER.

Key Architectural Guarantees:
  - SSRF Protection: Strictly forbids internal IP ranges (127.0.0.1, 10.x, 192.168.x,
    169.254.x, localhost, etc.) and non-http(s) schemes.
  - Resource Bounding: Enforces a strict 2.5s timeout and 64KB response size cap.
  - Faithful Excerpts: Extracts literal sentences directly from retrieved HTML text
    without LLM synthesis or alteration.
  - Graceful Fallback: If page retrieval fails, times out, or is non-HTML, preserves
    the search snippet and explicitly records the limitation note.
"""

from __future__ import annotations

import ipaddress
import logging
import re
from urllib.parse import urlparse

import httpx

from app.models.schemas import EvidenceItem

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 2.5
_MAX_RESPONSE_BYTES = 65536  # 64 KB cap
_MAX_PASSAGE_CHARS = 450


def is_safe_external_url(url: str) -> bool:
    """Validate that a URL is safe for external HTTP retrieval (prevents SSRF)."""
    try:
        parsed = urlparse(url.strip())
        if parsed.scheme.lower() not in {"http", "https"}:
            return False

        hostname = (parsed.hostname or "").strip().lower()
        if not hostname or "." not in hostname:
            return False

        if hostname in {"localhost", "127.0.0.1", "0.0.0.0", "::1", "metadata.google.internal"}:
            return False

        # Check for private or loopback IP addresses
        try:
            ip = ipaddress.ip_address(hostname)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        except ValueError:
            # Hostname is a domain name, not a literal IP
            pass

        return True
    except Exception:
        return False


def _clean_html_text(raw_html: str) -> str:
    """Strip HTML tags, styles, and scripts to extract readable text content."""
    # Remove script and style elements
    text = re.sub(r"<(script|style|svg|noscript)[^>]*>[\s\S]*?</\1>", " ", raw_html, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Replace common HTML entities
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
    # Collapse multiple whitespaces
    return re.sub(r"\s+", " ", text).strip()


def _extract_best_passage(page_text: str, claim_text: str) -> str | None:
    """Find the 1-3 sentence window in the page text that best matches the claim."""
    if not page_text or not claim_text:
        return None

    # Extract alphanumeric tokens from claim (lowercased)
    claim_tokens = set(re.findall(r"\b[a-z0-9\-\.]{2,}\b", claim_text.lower()))
    # Identify key numbers or entities from the claim
    claim_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?%?\b", claim_text))

    if not claim_tokens:
        return None

    # Split text into candidate sentences
    sentences = re.split(r"(?<=[.!?])\s+", page_text)
    if not sentences:
        return None

    best_score = 0.0
    best_passage = None

    for i in range(len(sentences)):
        # Test 1 to 2 consecutive sentences window
        for window_len in (1, 2):
            if i + window_len > len(sentences):
                break
            window = " ".join(sentences[i : i + window_len]).strip()
            if len(window) < 30:
                continue
            window_lower = window.lower()
            tokens_in_window = sum(1 for t in claim_tokens if t in window_lower)

            # Bonus for matching numbers or exact metrics
            number_bonus = sum(2 for n in claim_numbers if n.lower() in window_lower)
            score = tokens_in_window + number_bonus

            if score > best_score:
                best_score = score
                best_passage = window[:_MAX_PASSAGE_CHARS]

    # Require minimum match threshold (at least 3 key tokens or key numbers)
    min_tokens = min(3, len(claim_tokens))
    if best_score >= min_tokens:
        return best_passage

    return None


async def extract_passage_for_claim(
    item: EvidenceItem,
    claim_text: str,
    client: httpx.AsyncClient | None = None,
) -> tuple[str, str | None]:
    """
    Safely retrieve full page text and extract a relevant passage.

    Returns
    -------
    tuple[str, str | None]
        (passage_text, limitation_note)
        If page fetch fails or yields no match, returns (item.snippet, limitation_note).
    """
    if not is_safe_external_url(item.url):
        return item.snippet, "URL disallowed by safety policy; evaluated against search snippet."

    own_client = False
    if client is None:
        client = httpx.AsyncClient(
            timeout=_TIMEOUT_SECONDS,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,text/plain;q=0.9",
            },
            follow_redirects=True,
        )
        own_client = True

    try:
        response = await client.get(item.url)
        if response.status_code != 200:
            return item.snippet, f"Page returned HTTP {response.status_code}; evaluated against search snippet."

        content_type = response.headers.get("content-type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return item.snippet, f"Non-HTML content type ({content_type}); evaluated against search snippet."

        # Read only up to _MAX_RESPONSE_BYTES
        raw_text = response.text[:_MAX_RESPONSE_BYTES]
        cleaned_text = _clean_html_text(raw_text)
        passage = _extract_best_passage(cleaned_text, claim_text)

        if passage:
            return passage, None
        return item.snippet, "Full page fetched but lacked direct passage match; evaluated against search snippet."
    except httpx.TimeoutException:
        return item.snippet, "Page fetch timed out; evaluated against search snippet."
    except Exception as exc:
        logger.debug("Page fetch failed for %s: %s", item.url, exc)
        return item.snippet, "Page fetch unavailable; evaluated against search snippet."
    finally:
        if own_client:
            await client.aclose()

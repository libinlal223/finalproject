"""
app/retrieval/url_normalizer.py
───────────────────────────────
URL validation, canonicalization, and deduplication for evidence retrieval.

Rules:
  - Allowed schemes: http, https only (disallows javascript:, data:, file:, ftp:).
  - Normalizes scheme and host to lowercase.
  - Strips standard default ports (80 for http, 443 for https).
  - Trims trailing slash from path (unless path is empty or root).
  - Strips tracking / analytics query parameters (utm_*, fbclid, gclid, ref, etc.).
  - Sorts remaining legitimate query parameters for deterministic matching.
  - Removes URL fragments (#...).
"""

from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)

# Query parameter keys typically used for tracking, session states, or referrals
_TRACKING_PARAMS: set[str] = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "fbclid",
    "gclid",
    "gclsrc",
    "dclid",
    "zanpid",
    "msclkid",
    "ref",
    "source",
    "campaign",
}


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid, well-formed, and uses an allowed HTTP/HTTPS scheme.
    """
    if not url or not isinstance(url, str):
        return False

    clean_url = url.strip()
    if len(clean_url) < 10 or len(clean_url) > 2048:
        return False

    try:
        parsed = urlparse(clean_url)
    except Exception:
        return False

    # Disallow dangerous or unsupported schemes
    if parsed.scheme.lower() not in {"http", "https"}:
        return False

    # Must have a valid network location (hostname)
    netloc = parsed.netloc.strip()
    if not netloc or "." not in netloc:
        return False

    # Disallow control characters or whitespace in host/path
    if any(c.isspace() or ord(c) < 32 for c in clean_url):
        return False

    return True


def canonicalize_url(url: str) -> str:
    """
    Convert a URL into a normalized canonical representation for deduplication.

    Returns the canonical URL string, or the stripped original if parsing fails.
    """
    if not is_valid_url(url):
        return url.strip()

    try:
        parsed = urlparse(url.strip())
        scheme = parsed.scheme.lower()
        netloc = parsed.netloc.lower()

        # Remove default ports
        if (scheme == "http" and netloc.endswith(":80")) or (
            scheme == "https" and netloc.endswith(":443")
        ):
            netloc = netloc.rsplit(":", 1)[0]

        # Normalize path: remove redundant trailing slash if path length > 1
        path = parsed.path or "/"
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]

        # Filter out tracking query parameters and sort remaining
        query_pairs = parse_qsl(parsed.query, keep_blank_values=False)
        cleaned_pairs = [
            (k, v) for k, v in query_pairs if k.lower() not in _TRACKING_PARAMS
        ]
        cleaned_pairs.sort(key=lambda x: x[0])
        query = urlencode(cleaned_pairs)

        # Drop fragment
        return urlunparse((scheme, netloc, path, parsed.params, query, ""))
    except Exception as exc:
        logger.debug("Failed to canonicalize URL '%s': %s", url, exc)
        return url.strip()

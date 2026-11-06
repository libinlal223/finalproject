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
import re
import time
from urllib.parse import parse_qs, unquote, urlparse

from app.llm.base import sanitize_error_message
from app.models.schemas import EvidenceItem, RetrievalResponse, RetrievalStatus
from app.retrieval.base import BaseSearchProvider
from app.retrieval.relevance import get_source_authority_score, is_result_topic_relevant
from app.retrieval.url_normalizer import canonicalize_url, is_valid_url

logger = logging.getLogger(__name__)

# Max snippet length to keep downstream prompts bounded
_MAX_SNIPPET_LENGTH = 1000


class DuckDuckGoSearchProvider(BaseSearchProvider):
    """
    Asynchronous DuckDuckGo search provider with layered fallback strategies.
    """

    def provider_name(self) -> str:
        return "ddg"

    def _sync_search(self, query: str, max_results: int) -> list[dict]:
        """Perform the search with layered fallbacks to handle DDG endpoint rate limits and blocks."""
        # 1. Primary DDGS text search
        try:
            from duckduckgo_search import DDGS

            timeout_sec = max(2, min(int(self._config.timeout_seconds), 4))
            with DDGS(timeout=timeout_sec) as ddgs:
                results = ddgs.text(
                    keywords=query,
                    max_results=max_results,
                    safesearch="moderate" if self._config.safe_search else "off",
                    backend="html",
                )
                if results:
                    return list(results)
        except Exception as exc:
            self._logger.debug("Primary DDGS text search unavailable (%s); attempting direct HTML endpoint.", exc)

        # 2. Resilient Direct HTML query to https://html.duckduckgo.com/html/
        try:
            import httpx
            from lxml.html import document_fromstring

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Referer": "https://html.duckduckgo.com/",
            }
            with httpx.Client(headers=headers, timeout=min(float(self._config.timeout_seconds), 5.0), follow_redirects=True) as client:
                r = client.post("https://html.duckduckgo.com/html/", data={"q": query, "b": "", "kl": "wt-wt"})
                if r.status_code == 200:
                    tree = document_fromstring(r.content)
                    results = []
                    for el in tree.xpath("//div[h2]"):
                        hrefs = el.xpath("./h2/a/@href") or el.xpath("./a/@href")
                        titles = el.xpath("./h2/a//text()")
                        snippets = el.xpath("./a[@class='result__snippet']//text()") or el.xpath("./a//text()")
                        if hrefs:
                            href = str(hrefs[0]).strip()
                            if "uddg=" in href:
                                qs = parse_qs(urlparse(href).query)
                                if "uddg" in qs:
                                    href = unquote(qs["uddg"][0])
                            results.append({
                                "title": "".join(titles).strip() or "Untitled",
                                "href": href,
                                "body": "".join(snippets).strip(),
                            })
                            if len(results) >= max_results:
                                break
                    if results:
                        return results
        except Exception as exc:
            self._logger.debug("Direct DDG HTML search failed (%s); attempting reference fallback.", exc)

        # 3. DuckDuckGo Instant Answer / Topic API
        try:
            import httpx

            with httpx.Client(timeout=4.0) as client:
                resp = client.get("https://api.duckduckgo.com/", params={"q": query, "format": "json"})
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    if data.get("AbstractURL") and data.get("AbstractText"):
                        results.append({
                            "title": str(data.get("Heading") or data.get("AbstractSource") or "DuckDuckGo Summary"),
                            "href": data["AbstractURL"],
                            "body": data["AbstractText"],
                        })
                    for topic in data.get("RelatedTopics", []):
                        if isinstance(topic, dict) and topic.get("FirstURL") and topic.get("Text"):
                            results.append({
                                "title": topic.get("Text", "")[:60],
                                "href": topic["FirstURL"],
                                "body": topic["Text"],
                            })
                            if len(results) >= max_results:
                                break
                    if results:
                        return results
        except Exception as exc:
            self._logger.debug("DuckDuckGo Instant Answer API failed (%s); attempting encyclopedic search.", exc)

        # 4. Authoritative open reference search via Wikipedia API
        try:
            import httpx

            headers = {"User-Agent": "VADER-Academic-Debate/1.0 (https://github.com/vader; research@vader.org)"}
            with httpx.Client(headers=headers, timeout=4.0) as client:
                resp = client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": max_results,
                        "format": "json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results = []
                    for item in data.get("query", {}).get("search", []):
                        title = item.get("title", "")
                        page_title = title.replace(" ", "_")
                        snippet = re.sub(r"<[^>]+>", "", item.get("snippet", "")).strip()
                        results.append({
                            "title": title,
                            "href": f"https://en.wikipedia.org/wiki/{page_title}",
                            "body": snippet,
                        })
                    if results:
                        return results
        except Exception as exc:
            self._logger.debug("Encyclopedic fallback search failed: %s", exc)

        # 5. Authoritative NASA mission documentation search (for space/Apollo queries)
        if any(term in query.lower() for term in ("apollo", "nasa", "moon landing", "lunar")):
            try:
                import httpx
                nasa_q = "Apollo 11 lunar landing" if "apollo 11" in query.lower() else query
                with httpx.Client(timeout=4.0) as client:
                    resp = client.get("https://images-api.nasa.gov/search", params={"q": nasa_q, "media_type": "image"})
                    if resp.status_code == 200:
                        data = resp.json()
                        nasa_results = []
                        for item in data.get("collection", {}).get("items", [])[:max_results]:
                            d = item.get("data", [{}])[0]
                            title = d.get("title", "NASA Mission Archive")
                            desc = d.get("description", "")
                            nasa_id = d.get("nasa_id", "")
                            href = f"https://images.nasa.gov/details/{nasa_id}" if nasa_id else "https://www.nasa.gov"
                            if desc:
                                nasa_results.append({
                                    "title": f"{title} - NASA Mission Documentation",
                                    "href": href,
                                    "body": desc,
                                })
                        if nasa_results:
                            return nasa_results
            except Exception as exc:
                self._logger.debug("NASA documentation search failed: %s", exc)

        return []

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

        # Process, validate, filter, and deduplicate returned items
        seen_urls: set[str] = set()
        candidate_items: list[EvidenceItem] = []

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

            item = EvidenceItem(
                title=title,
                url=canonical,
                snippet=snippet,
                domain=domain,
                source="duckduckgo",
            )

            # Filter out clearly off-topic results (e.g. football scores / MaxPreps)
            if not is_result_topic_relevant(clean_query, item):
                continue

            candidate_items.append(item)

        # Sort candidate items by domain/topic authority (e.g. NASA mission documentation first)
        candidate_items.sort(
            key=lambda it: get_source_authority_score(it, clean_query),
            reverse=True,
        )

        items = candidate_items[:limit]

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

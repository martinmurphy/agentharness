"""A keyless web search tool, backed by DuckDuckGo's HTML endpoint.

State-free, built on stdlib ``urllib`` + ``html.parser`` (no new dependency).
Keyless so search works with zero setup, consistent with the rest of the
harness — but it scrapes DuckDuckGo's HTML results page, so it is unofficial and
best-effort: it can break if the markup changes and may be rate-limited or
blocked on some IPs. Adding a keyed backend (Brave/Tavily) later is a new backend
function plus a selector; ``SearchResult`` is the shared contract (see
docs/future-work.md).

``web_search`` finds links; the ``web_fetch`` tool reads a page — the model
searches, then fetches whichever results are worth reading.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

from agentharness.tools.registry import Tool

_ENDPOINT = "https://html.duckduckgo.com/html/"
_TIMEOUT_SECONDS = 30
_MAX_BYTES = 500_000  # results pages are larger than a typical fetch
_USER_AGENT = "agentharness/0.1"
_DEFAULT_COUNT = 5
_MAX_COUNT = 10


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


class _ResultParser(HTMLParser):
    """Extract DuckDuckGo HTML results.

    Result anchors carry class ``result__a`` (title text + redirect href);
    snippet anchors carry class ``result__snippet``. We pair them in document
    order: each ``result__a`` opens a new result, and the next ``result__snippet``
    text attaches to it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._pending: dict[str, str] | None = None  # {title, url} awaiting snippet
        self._mode: str | None = None  # "title" | "snippet" | None
        self._buf: list[str] = []

    @staticmethod
    def _classes(attrs: list[tuple[str, str | None]]) -> set[str]:
        for name, value in attrs:
            if name == "class" and value:
                return set(value.split())
        return set()

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        classes = self._classes(attrs)
        if "result__a" in classes:
            self._flush_title()
            href = next((v for n, v in attrs if n == "href" and v), "")
            self._pending = {"title": "", "url": _real_url(href)}
            self._mode = "title"
            self._buf = []
        elif "result__snippet" in classes and self._pending is not None:
            self._mode = "snippet"
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode is not None:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._mode is None:
            return
        text = "".join(self._buf).strip()
        if self._mode == "title" and self._pending is not None:
            self._pending["title"] = text
        elif self._mode == "snippet" and self._pending is not None:
            self.results.append(
                SearchResult(
                    title=self._pending["title"], url=self._pending["url"], snippet=text
                )
            )
            self._pending = None
        self._mode = None
        self._buf = []

    def _flush_title(self) -> None:
        # A result with a title but no snippet still counts.
        if self._pending is not None and self._pending["title"]:
            self.results.append(
                SearchResult(title=self._pending["title"], url=self._pending["url"], snippet="")
            )
        self._pending = None


def _real_url(href: str) -> str:
    """Recover the target URL from a DuckDuckGo redirect href.

    DDG hrefs look like ``//duckduckgo.com/l/?uddg=<url-encoded>&rut=...``; the
    real URL is the ``uddg`` query param. Falls back to the raw href (prefixing
    ``https:`` for protocol-relative links) if there is no ``uddg`` param.
    """
    if not href:
        return ""
    parsed = urllib.parse.urlparse(href)
    uddg = urllib.parse.parse_qs(parsed.query).get("uddg")
    if uddg:
        return uddg[0]
    if href.startswith("//"):
        return "https:" + href
    return href


def _duckduckgo_search(query: str, count: int) -> list[SearchResult]:
    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    request = urllib.request.Request(
        _ENDPOINT,
        data=data,
        method="POST",
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp:
            raw = resp.read(_MAX_BYTES)
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise ValueError(f"search request failed: {reason}") from exc
    except TimeoutError as exc:
        raise ValueError(f"search timed out after {_TIMEOUT_SECONDS}s") from exc

    parser = _ResultParser()
    parser.feed(raw.decode("utf-8", errors="replace"))
    parser._flush_title()  # emit a trailing title-only result, if any
    return parser.results[:count]


def _format(query: str, results: list[SearchResult]) -> str:
    if not results:
        return f"No results found for {query!r}."
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title} — {r.url}")
        if r.snippet:
            lines.append(f"   {r.snippet}")
    return "\n".join(lines)


def _search(args: dict) -> str:
    query = args.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("'query' is required and must be a non-empty string")
    count = args.get("count", _DEFAULT_COUNT)
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = _DEFAULT_COUNT
    count = max(1, min(count, _MAX_COUNT))
    results = _duckduckgo_search(query.strip(), count)
    return _format(query.strip(), results)


def web_search_tool() -> Tool:
    return Tool(
        name="web_search",
        description=(
            "Search the web (via DuckDuckGo) and return result titles, URLs, and "
            "snippets. Use it to discover pages, then read a chosen result with the "
            "web_fetch tool. Keyless and best-effort — results may occasionally be "
            "unavailable."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "count": {
                    "type": "integer",
                    "description": f"Number of results (1-{_MAX_COUNT}, default {_DEFAULT_COUNT}).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        handler=_search,
    )

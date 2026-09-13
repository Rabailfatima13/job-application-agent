"""`web_search` - the project's only outward network call (FR-2).

The tool returns *structured search results and nothing else*. It never
summarises, never calls a model, and never decides what matters: that is the
research agent's job. Keeping the split here is what lets Week 7 expose this
same tool over MCP without dragging an LLM along with it.

Two providers, both plain HTTPS GETs over the `httpx` already in
requirements: SerpAPI (the configured default) and Brave. They differ only in
URL, auth header and response shape, so each is a few lines rather than a
subclass hierarchy.

The HTTP call itself is injected (`request_fn`), so tests exercise the real
query-building and parsing without a key or a network.
"""

import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..config import Settings
from . import Tool

SERPAPI = "serpapi"
BRAVE = "brave"

SERPAPI_URL = "https://serpapi.com/search.json"
BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

# Bounded on purpose: a short role-relevant brief needs a handful of results,
# and every extra one costs latency, tokens and search quota (NFR-1, NFR-2).
DEFAULT_MAX_RESULTS = 5
# Provider snippets are already short; this stops a pathological one from
# bloating the summarisation prompt.
MAX_SNIPPET_CHARS = 500

RequestFn = Callable[[str, dict, dict], dict]

# httpx logs every request at INFO, and the log line contains the full URL.
# SerpAPI takes its credential as a query parameter, so that one line prints the
# API key wherever the host sends INFO logs - which, when this package runs as
# an MCP server, means the server's stderr and therefore the client's logs. A
# live run leaked a real key exactly this way. Nothing here needs httpx's
# request log, so it is silenced at import; anything logged at WARNING or above
# still comes through.
logging.getLogger("httpx").setLevel(logging.WARNING)


class SearchError(RuntimeError):
    """Raised when the search provider could not be reached or understood.

    A tool-level failure, not a pipeline-level one - the research agent
    catches it and degrades to a brief that says research was unavailable.
    """


class SearchResult(BaseModel):
    """One search hit, trimmed to what the research agent actually uses.

    Lives with the tool rather than in models/: it is an intermediate the
    agent consumes, not one of the proposal's structured outputs. What reaches
    the user is `CompanyBrief.sources`.
    """

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = Field(default="", description="Provider-supplied extract.")


def _http_get(url: str, params: dict, headers: dict) -> dict:
    import httpx  # imported lazily so offline tests never touch it

    try:
        response = httpx.get(url, params=params, headers=headers, timeout=45.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        # Deliberately not `{exc}`: httpx puts the full request URL in its
        # message, and for SerpAPI the key travels as a query parameter - so
        # the raw message carries the credential into error responses, MCP
        # tool errors and the trace log. Only the status code is reported, and
        # `from None` keeps the URL-bearing original out of the traceback.
        raise SearchError(
            f"Web search request failed with HTTP {exc.response.status_code}."
        ) from None
    except httpx.HTTPError as exc:
        raise SearchError(
            f"Web search request failed ({type(exc).__name__})."
        ) from None
    except ValueError as exc:
        # response.json() raises json.JSONDecodeError (a ValueError, not an
        # httpx.HTTPError) when a 200 response body isn't valid JSON - e.g. a
        # provider serving an HTML rate-limit or maintenance page. Uncaught,
        # this would crash the whole pipeline run instead of degrading to a
        # brief that says research was unavailable, same as any other search
        # failure.
        raise SearchError(
            f"Web search response was not valid JSON ({type(exc).__name__})."
        ) from None


def _serpapi_request(
    query: str, api_key: str, max_results: int
) -> tuple[str, dict, dict]:
    params = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max_results,
    }
    return SERPAPI_URL, params, {}


def _brave_request(query: str, api_key: str, max_results: int) -> tuple[str, dict, dict]:
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    return BRAVE_URL, {"q": query, "count": max_results}, headers


def _parse_serpapi(payload: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("link", ""),
            snippet=(item.get("snippet") or "")[:MAX_SNIPPET_CHARS],
        )
        for item in payload.get("organic_results") or []
        if item.get("link")
    ]


def _parse_brave(payload: dict[str, Any]) -> list[SearchResult]:
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=(item.get("description") or "")[:MAX_SNIPPET_CHARS],
        )
        for item in (payload.get("web") or {}).get("results") or []
        if item.get("url")
    ]


_PROVIDERS = {
    SERPAPI: (_serpapi_request, _parse_serpapi),
    BRAVE: (_brave_request, _parse_brave),
}


class WebSearchClient:
    """Runs a query against the configured provider and returns results.

    The API key comes from configuration (`SEARCH_API_KEY`) and is never
    written down here; a missing key fails loudly at search time rather than
    at import time, so the rest of the system still runs without one.
    """

    def __init__(
        self,
        provider: str = SERPAPI,
        api_key: str | None = None,
        request_fn: RequestFn = _http_get,
        max_results: int = DEFAULT_MAX_RESULTS,
    ) -> None:
        if provider not in _PROVIDERS:
            raise ValueError(
                f"Unknown search provider '{provider}' "
                f"(expected one of {sorted(_PROVIDERS)})."
            )
        self.provider = provider
        self.max_results = max_results
        self._api_key = api_key
        self._request_fn = request_fn

    @classmethod
    def from_settings(cls, settings: Settings, **kwargs) -> "WebSearchClient":
        return cls(
            provider=settings.search_provider,
            api_key=settings.search_api_key,
            **kwargs,
        )

    def search(self, query: str, max_results: int | None = None) -> list[SearchResult]:
        query = query.strip()
        if not query:
            raise SearchError("Refusing to run an empty search query.")
        if not self._api_key:
            raise SearchError(
                f"SEARCH_API_KEY is not set for provider '{self.provider}' "
                "(see .env.example)."
            )

        limit = max_results or self.max_results
        build_request, parse = _PROVIDERS[self.provider]
        url, params, headers = build_request(query, self._api_key, limit)

        payload = self._request_fn(url, params, headers)
        if not isinstance(payload, dict):
            raise SearchError(f"{self.provider} returned an unexpected response.")
        return parse(payload)[:limit]


def build_web_search_tool(client: WebSearchClient) -> Tool:
    """The registry entry. Returns `SearchResult`s; summarising is not its job.

    No tracing here: `BaseAgent.call_tool` already records one event per tool
    call, and tracing again inside the tool would double-count it.
    """
    return Tool(
        name="web_search",
        description=(
            "Search the web for facts about a company or role. Returns a short "
            "list of {title, url, snippet} results."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": f"How many results to return (default "
                    f"{DEFAULT_MAX_RESULTS}).",
                },
            },
            "required": ["query"],
        },
        run=client.search,
    )

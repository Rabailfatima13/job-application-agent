"""The web_search tool (FR-2).

The HTTP call is injected, so these tests exercise real query building and real
response parsing for both providers without a key or a network connection.
"""

import pytest

from job_agent.tools import SearchError, WebSearchClient, build_web_search_tool
from job_agent.tools.web_search import BRAVE_URL, SERPAPI_URL, SearchResult

SERPAPI_PAYLOAD = {
    "organic_results": [
        {
            "title": "Arbisoft - Software Development Company",
            "link": "https://arbisoft.com",
            "snippet": "Arbisoft builds data platforms and machine learning systems.",
        },
        {
            "title": "Arbisoft on LinkedIn",
            "link": "https://linkedin.com/company/arbisoft",
            "snippet": "Engineering teams in Lahore working with Python and Django.",
        },
        {"title": "No link here", "snippet": "dropped - unusable without a url"},
    ]
}

BRAVE_PAYLOAD = {
    "web": {
        "results": [
            {
                "title": "Arbisoft",
                "url": "https://arbisoft.com",
                "description": "Builds data platforms and ML systems.",
            }
        ]
    }
}


class RecordingRequest:
    """A stand-in for the HTTP layer that records what it was asked for."""

    def __init__(self, payload=None, error: Exception | None = None) -> None:
        self.payload = payload if payload is not None else {}
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, url, params, headers) -> dict:
        self.calls.append({"url": url, "params": params, "headers": headers})
        if self.error is not None:
            raise self.error
        return self.payload


def test_serpapi_query_is_built_from_the_configuration():
    request = RecordingRequest(SERPAPI_PAYLOAD)
    client = WebSearchClient("serpapi", api_key="test-key", request_fn=request)

    client.search("Arbisoft Junior AI Engineer", max_results=3)

    (call,) = request.calls
    assert call["url"] == SERPAPI_URL
    assert call["params"]["q"] == "Arbisoft Junior AI Engineer"
    assert call["params"]["num"] == 3
    assert call["params"]["api_key"] == "test-key"


def test_serpapi_results_are_parsed_into_structured_hits():
    client = WebSearchClient(
        "serpapi", api_key="k", request_fn=RecordingRequest(SERPAPI_PAYLOAD)
    )

    results = client.search("Arbisoft")

    assert all(isinstance(r, SearchResult) for r in results)
    assert results[0].title == "Arbisoft - Software Development Company"
    assert results[0].url == "https://arbisoft.com"
    assert "machine learning" in results[0].snippet
    # A hit without a usable URL cannot be cited, so it is dropped.
    assert len(results) == 2


def test_brave_uses_its_own_url_auth_header_and_response_shape():
    request = RecordingRequest(BRAVE_PAYLOAD)
    client = WebSearchClient("brave", api_key="brave-key", request_fn=request)

    results = client.search("Arbisoft", max_results=2)

    (call,) = request.calls
    assert call["url"] == BRAVE_URL
    assert call["headers"]["X-Subscription-Token"] == "brave-key"
    assert call["params"]["count"] == 2
    assert results[0].url == "https://arbisoft.com"


def test_results_are_bounded_by_the_requested_limit():
    client = WebSearchClient(
        "serpapi", api_key="k", request_fn=RecordingRequest(SERPAPI_PAYLOAD)
    )

    assert len(client.search("Arbisoft", max_results=1)) == 1


def test_empty_results_are_returned_as_an_empty_list_not_an_error():
    client = WebSearchClient(
        "serpapi", api_key="k", request_fn=RecordingRequest({"organic_results": []})
    )

    assert client.search("obscure company") == []


def test_a_provider_error_surfaces_as_a_search_error():
    request = RecordingRequest(error=SearchError("Web search request failed: 503"))
    client = WebSearchClient("serpapi", api_key="k", request_fn=request)

    with pytest.raises(SearchError):
        client.search("Arbisoft")


def test_a_failed_request_never_reports_the_api_key():
    # httpx puts the whole request URL in its message, and SerpAPI carries the
    # key as a query parameter - so the raw message would leak the credential
    # into error responses, MCP tool errors and the trace log.
    import httpx

    from job_agent.tools.web_search import _http_get

    request = httpx.Request(
        "GET",
        "https://serpapi.com/search.json",
        params={"engine": "google", "q": "x", "api_key": "SUPER-SECRET-KEY"},
    )
    response = httpx.Response(401, request=request)

    def raise_status(*_args, **_kwargs):
        response.raise_for_status()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "get", raise_status)
        with pytest.raises(SearchError) as exc:
            _http_get(
                "https://serpapi.com/search.json",
                {"api_key": "SUPER-SECRET-KEY"},
                {},
            )

    message = str(exc.value)
    assert "SUPER-SECRET-KEY" not in message
    assert "api_key" not in message
    assert "401" in message  # the useful part survives


def test_httpx_request_logging_is_silenced_so_urls_never_reach_logs():
    # A live MCP run printed a real SerpAPI key: httpx logs every request URL at
    # INFO, and SerpAPI carries the credential as a query parameter, so the log
    # line contained the key. The MCP SDK enables INFO logging, so it went
    # straight to the server's stderr.
    import logging

    import job_agent.tools.web_search  # noqa: F401  (import applies the setting)

    assert logging.getLogger("httpx").level >= logging.WARNING


def test_the_httpx_logger_stays_quiet_even_when_the_root_logger_is_verbose(caplog):
    import logging

    import httpx

    from job_agent.tools.web_search import _http_get

    request = httpx.Request(
        "GET",
        "https://serpapi.com/search.json",
        params={"api_key": "SUPER-SECRET-KEY"},
    )
    response = httpx.Response(401, request=request)

    with caplog.at_level(logging.DEBUG):  # as verbose as a host could get
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(httpx, "get", lambda *a, **k: response.raise_for_status())
            with pytest.raises(SearchError):
                _http_get(
                    "https://serpapi.com/search.json",
                    {"api_key": "SUPER-SECRET-KEY"},
                    {},
                )

    assert "SUPER-SECRET-KEY" not in caplog.text


def test_a_failed_search_writes_no_credential_into_the_trace_log(tmp_path):
    # Tool failures are recorded as trace events, and the collector mirrors
    # them to a file on disk. Whatever the search error says ends up there, so
    # it must not say the key.
    import httpx

    from job_agent.observability import TraceCollector, traced_tool_call
    from job_agent.tools.web_search import _http_get

    log_path = tmp_path / "tool_calls.log"
    collector = TraceCollector(log_path=log_path)
    request = httpx.Request(
        "GET",
        "https://serpapi.com/search.json",
        params={"api_key": "SUPER-SECRET-KEY"},
    )
    response = httpx.Response(401, request=request)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "get", lambda *a, **k: response.raise_for_status())
        with pytest.raises(SearchError):
            with traced_tool_call(collector, "research", "web_search", {"q": "x"}):
                _http_get(
                    "https://serpapi.com/search.json",
                    {"api_key": "SUPER-SECRET-KEY"},
                    {},
                )

    written = log_path.read_text(encoding="utf-8")
    assert "SUPER-SECRET-KEY" not in written
    assert "api_key" not in written
    assert '"status": "error"' in written  # the failure is still recorded


def test_a_transport_failure_also_reports_nothing_sensitive():
    import httpx

    from job_agent.tools.web_search import _http_get

    def boom(*_args, **_kwargs):
        raise httpx.ConnectError("failed to connect to https://serpapi.com?api_key=SECRET")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "get", boom)
        with pytest.raises(SearchError) as exc:
            _http_get("https://serpapi.com/search.json", {"api_key": "SECRET"}, {})

    assert "SECRET" not in str(exc.value)
    assert "ConnectError" in str(exc.value)


def test_a_non_json_200_response_is_reported_as_a_search_error_not_a_crash():
    # A provider can return HTTP 200 with a non-JSON body - an HTML rate-limit
    # or maintenance page is a realistic case. response.json() raises
    # json.JSONDecodeError there, which is a ValueError, not an httpx.HTTPError -
    # uncaught, that would crash the whole pipeline run instead of degrading
    # to a brief that says research was unavailable, the same as any other
    # search failure.
    import httpx

    from job_agent.tools.web_search import _http_get

    def html_response(*_args, **_kwargs):
        return httpx.Response(
            200,
            text="<html>rate limited</html>",
            request=httpx.Request("GET", "https://serpapi.com/search.json"),
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(httpx, "get", html_response)
        with pytest.raises(SearchError) as exc:
            _http_get("https://serpapi.com/search.json", {"api_key": "SECRET"}, {})

    assert "SECRET" not in str(exc.value)
    assert "JSON" in str(exc.value)


def test_an_unexpected_payload_is_rejected():
    client = WebSearchClient(
        "serpapi", api_key="k", request_fn=RecordingRequest(payload="<html>")
    )

    with pytest.raises(SearchError):
        client.search("Arbisoft")


def test_a_missing_api_key_fails_at_search_time_with_a_clear_message():
    client = WebSearchClient("serpapi", api_key=None, request_fn=RecordingRequest())

    with pytest.raises(SearchError, match="SEARCH_API_KEY"):
        client.search("Arbisoft")


def test_an_empty_query_is_refused_before_any_request_is_made():
    request = RecordingRequest(SERPAPI_PAYLOAD)
    client = WebSearchClient("serpapi", api_key="k", request_fn=request)

    with pytest.raises(SearchError):
        client.search("   ")
    assert request.calls == []


def test_an_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="Unknown search provider"):
        WebSearchClient("google", api_key="k")


def test_the_client_can_be_built_from_settings(settings):
    client = WebSearchClient.from_settings(settings, request_fn=RecordingRequest())

    assert client.provider == "serpapi"  # from configuration, not hard-coded


def test_the_tool_wraps_the_client_and_returns_search_results():
    client = WebSearchClient(
        "serpapi", api_key="k", request_fn=RecordingRequest(SERPAPI_PAYLOAD)
    )
    tool = build_web_search_tool(client)

    results = tool.run(query="Arbisoft")

    assert tool.name == "web_search"
    assert tool.input_schema["required"] == ["query"]
    assert all(isinstance(r, SearchResult) for r in results)

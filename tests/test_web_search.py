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

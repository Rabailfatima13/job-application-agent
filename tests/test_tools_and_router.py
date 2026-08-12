"""Tool registry and model routing - the two seams Week 6/7 work plugs into."""

import pytest

from job_agent.config import HEAVY, LIGHT
from job_agent.tools import (
    Tool,
    ToolRegistry,
    WebSearchClient,
    build_default_registry,
)


def _tool(name="web_search"):
    return Tool(
        name=name,
        description="Search the web.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        run=lambda query: f"results for {query}",
    )


def test_registry_exposes_and_runs_registered_tools():
    registry = ToolRegistry([_tool()])
    assert registry.names() == ["web_search"]
    assert registry.get("web_search").run(query="Arbisoft") == "results for Arbisoft"


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([_tool()])
    with pytest.raises(ValueError):
        registry.register(_tool())


def test_default_registry_exposes_the_parsing_tools(router):
    # Agents discover tools here, not by importing tool modules directly.
    registry = build_default_registry(router)
    assert set(registry.names()) == {"parse_jd", "parse_cv"}
    for tool in registry.all():
        assert tool.input_schema["required"] == ["source"]


def test_the_registry_gains_web_search_when_a_search_client_is_supplied(router):
    client = WebSearchClient("serpapi", api_key="k", request_fn=lambda *_: {})
    registry = build_default_registry(router, search_client=client)

    assert set(registry.names()) == {"parse_jd", "parse_cv", "web_search"}


def test_parsing_routes_to_the_cheap_tier_and_scoring_to_the_heavy_one(router):
    assert router.tier_for("parse_cv") == LIGHT
    assert router.tier_for("score_fit") == HEAVY
    # An unrecognised step must not silently land on the weaker model.
    assert router.tier_for("some_future_step") == HEAVY


def test_clients_are_built_once_per_tier(router):
    assert router.for_step("parse_cv") is router.for_step("parse_jd")
    assert router.for_step("parse_cv") is not router.for_step("score_fit")

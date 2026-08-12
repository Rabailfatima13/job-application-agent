"""Tool definitions and the registry agents are given.

A tool is a plain callable plus a JSON-schema description (NFR-7: narrow,
well-named, independently testable). The schema shape is the one both
Anthropic and OpenAI tool-calling accept, so a tool written once works on
either provider - and the same definitions are what the MCP server exposes.

The registry is an object, not a module-level global, so each test builds its
own and nothing leaks between runs.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., Any]


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())


# Imported after Tool/ToolRegistry exist, since tool modules build on them.
from .document_io import load_document_text  # noqa: E402
from .parsing import build_parsing_tools  # noqa: E402
from .web_search import (  # noqa: E402
    SearchError,
    SearchResult,
    WebSearchClient,
    build_web_search_tool,
)


def build_default_registry(
    router, collector=None, search_client: WebSearchClient | None = None
) -> ToolRegistry:
    """Every tool the system currently has, ready for an agent.

    This is the single discovery point: agents (and, in Week 7, the MCP server)
    take a registry built here rather than importing tool modules directly, so
    a new tool is registered in exactly one place.

    `web_search` appears only when a search client is supplied, so a run
    without search credentials still gets a working parsing registry.
    """
    tools = build_parsing_tools(router, collector)
    if search_client is not None:
        tools.append(build_web_search_tool(search_client))
    return ToolRegistry(tools)


__all__ = [
    "SearchError",
    "SearchResult",
    "Tool",
    "ToolRegistry",
    "WebSearchClient",
    "build_default_registry",
    "build_parsing_tools",
    "build_web_search_tool",
    "load_document_text",
]

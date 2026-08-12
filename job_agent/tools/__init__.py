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

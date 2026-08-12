"""The single interface every model provider is adapted to.

`ChatResult` carries token counts because FR-12 requires per-call cost to be
traceable; a provider that cannot report them returns 0 rather than a guess.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A tool the model asked to run."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResult:
    """One normalized model turn, whatever the provider was."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = "end_turn"  # "end_turn" | "tool_use"
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@runtime_checkable
class ModelClient(Protocol):
    """What an agent is allowed to assume about a model.

    Implementations live in providers.py. Tests pass a stub with the same two
    members instead of calling a real API.
    """

    name: str

    def complete(
        self,
        messages: list[dict],
        system: str,
        tools: list | None = None,
        max_tokens: int = 1024,
    ) -> ChatResult:
        """Send one turn and return the normalized result."""
        ...

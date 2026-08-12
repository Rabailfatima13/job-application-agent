"""The common base every worker agent inherits (NFR-7).

It provides the three things all workers need and nothing else: a traced model
call routed to the right tier, a traced tool call, and a `run(context)`
contract. Worker subclasses are therefore just a prompt, a tool list, and an
output schema - which is what keeps them explainable.
"""

from abc import ABC, abstractmethod

from ..llm import ChatResult, ModelRouter
from ..memory.session import RunContext
from ..observability import TraceCollector, traced_tool_call
from ..tools import ToolRegistry


class BaseAgent(ABC):
    """A narrowly-scoped worker with a name, tools, and one job."""

    #: Human-readable agent name, used as the `agent` field on trace events.
    name: str = "agent"

    def __init__(
        self,
        router: ModelRouter,
        tools: ToolRegistry | None = None,
        collector: TraceCollector | None = None,
    ) -> None:
        self.router = router
        self.tools = tools or ToolRegistry()
        self.collector = collector

    @abstractmethod
    def run(self, context: RunContext):
        """Do this agent's one job, reading from and returning a structured
        result for `context`. Implementations must return a Pydantic model,
        never free text."""

    def call_model(
        self,
        step: str,
        messages: list[dict],
        system: str,
        max_tokens: int = 1024,
    ) -> ChatResult:
        """Run one model turn on whichever tier `step` is routed to, recording
        latency and token counts as a trace event."""
        tier = self.router.tier_for(step)
        client = self.router.client(tier)
        with traced_tool_call(
            self.collector,
            agent=self.name,
            name=f"model:{tier}:{step}",
            arguments={"model": client.name},
            kind="model",
        ) as outcome:
            result = client.complete(messages, system=system, max_tokens=max_tokens)
            outcome["result"] = result.text
            outcome["input_tokens"] = result.input_tokens
            outcome["output_tokens"] = result.output_tokens
        return result

    def call_tool(self, name: str, **arguments):
        """Run one of this agent's tools, recording it as a trace event.

        A tool this agent does not own is an error, not a fallback - workers
        are meant to have narrow, declared capabilities.
        """
        tool = self.tools.get(name)
        if tool is None:
            raise ValueError(f"{self.name} has no tool '{name}'.")
        with traced_tool_call(
            self.collector, agent=self.name, name=name, arguments=arguments
        ) as outcome:
            result = tool.run(**arguments)
            outcome["result"] = result
        return result

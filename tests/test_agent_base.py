"""BaseAgent gives workers traced model and tool calls - the behaviour every
Week 6 agent inherits rather than re-implements."""

import pytest

from job_agent.agents import BaseAgent
from job_agent.memory.session import RunContext
from job_agent.observability import TraceCollector
from job_agent.tools import Tool, ToolRegistry


class ParsingAgent(BaseAgent):
    """Smallest possible worker: one model call on a light-tier step."""

    name = "parsing"

    def run(self, context: RunContext):
        return self.call_model("parse_cv", [{"role": "user", "content": "cv"}], "sys")


def test_model_calls_are_routed_and_traced(router):
    collector = TraceCollector()
    agent = ParsingAgent(router, collector=collector)

    result = agent.run(RunContext(cv_text="cv", jd_text="jd"))

    assert result.text == "ok"
    (event,) = collector.events
    assert event.kind == "model"
    assert event.name == "model:light:parse_cv"  # cheap tier, as routed
    assert (event.input_tokens, event.output_tokens) == (10, 5)


def test_tool_calls_are_traced(router):
    collector = TraceCollector()
    tools = ToolRegistry(
        [
            Tool(
                name="web_search",
                description="Search.",
                input_schema={"type": "object", "properties": {}},
                run=lambda query: f"hits for {query}",
            )
        ]
    )
    agent = ParsingAgent(router, tools=tools, collector=collector)

    assert agent.call_tool("web_search", query="Arbisoft") == "hits for Arbisoft"
    assert collector.events[0].name == "web_search"


def test_an_agent_cannot_call_a_tool_it_does_not_own(router):
    agent = ParsingAgent(router)
    with pytest.raises(ValueError):
        agent.call_tool("draft_cover_letter")

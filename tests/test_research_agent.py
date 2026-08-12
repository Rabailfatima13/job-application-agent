"""Research agent (FR-2, US-2).

The search tool is stubbed and the model is the existing stub client, so the
whole agent runs offline. What is tested is the agent's own behaviour: query
construction, the search/summarise split, source preservation, the grounding
guard, and honest degradation when there is nothing to research.
"""

import json

import pytest

from job_agent.agents import ResearchAgent
from job_agent.agents.research import (
    NO_COMPANY_SUMMARY,
    build_query,
    build_summarisation_prompt,
    unsupported_numbers,
)
from job_agent.memory.session import RunContext
from job_agent.models import CompanyBrief, JobDescription
from job_agent.observability import TraceCollector
from job_agent.tools import SearchError, Tool, ToolRegistry
from job_agent.tools.web_search import SearchResult
from job_agent.validation import RetryExhaustedError

RESULTS = [
    SearchResult(
        title="Arbisoft - Software Development Company",
        url="https://arbisoft.com",
        snippet=(
            "Arbisoft builds data platforms, search systems and machine learning "
            "products for clients including Coursera."
        ),
    ),
    SearchResult(
        title="Arbisoft engineering blog",
        url="https://arbisoft.com/blog",
        snippet="Teams work in Python and Django, with a growing LLM practice.",
    ),
]

EXTRACTION = {
    "summary": (
        "Arbisoft builds data platforms and machine learning products for "
        "clients including Coursera. Its teams work mainly in Python and Django "
        "and it has a growing LLM practice."
    ),
    "facts": [
        {"text": "Builds data platforms and search systems.", "source_index": 0},
        {"text": "Works with clients including Coursera.", "source_index": 0},
        {"text": "Engineering teams use Python and Django.", "source_index": 1},
    ],
}


class FakeSearchTool:
    """A web_search stand-in with the real tool's name and signature."""

    def __init__(self, results=None, error: Exception | None = None) -> None:
        self.results = list(results or [])
        self.error = error
        self.calls: list[dict] = []

    def run(self, query, max_results=None):
        self.calls.append({"query": query, "max_results": max_results})
        if self.error is not None:
            raise self.error
        return list(self.results)

    def as_tool(self) -> Tool:
        return Tool(
            name="web_search",
            description="Search the web.",
            input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
            run=self.run,
        )


def build_agent(make_router, reply=None, results=None, error=None, collector=None):
    router = make_router(light_replies=[reply] if reply else None)
    tool = FakeSearchTool(results, error)
    agent = ResearchAgent(
        router, tools=ToolRegistry([tool.as_tool()]), collector=collector
    )
    return agent, tool, router


@pytest.fixture
def job() -> JobDescription:
    return JobDescription(
        role="Junior AI Engineer",
        company="Arbisoft",
        raw_text="...",
        requirements=[],
        responsibilities=["Build LLM-backed product features"],
        skills=["Python", "FastAPI", "pytest"],
    )


# --- Query construction ------------------------------------------------------


def test_the_query_combines_company_role_and_a_couple_of_technologies(job):
    assert build_query(job) == "Arbisoft Junior AI Engineer Python FastAPI"


def test_the_search_tool_is_called_with_a_bounded_result_count(make_router, job):
    agent, tool, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    agent.research(job, max_results=3)

    (call,) = tool.calls
    assert call["query"].startswith("Arbisoft Junior AI Engineer")
    assert call["max_results"] == 3


# --- Search -> summarise split -----------------------------------------------


def test_the_retrieved_results_are_what_the_model_summarises(make_router, job):
    agent, _, router = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    agent.research(job)

    prompt = router.client("light").calls[0]["messages"][0]["content"]
    assert "[0] Arbisoft - Software Development Company" in prompt
    assert "https://arbisoft.com/blog" in prompt
    # Role context is included so the brief stays role-relevant...
    assert "ROLE BEING APPLIED FOR: Junior AI Engineer" in prompt


def test_the_prompt_contains_only_retrieved_material(job):
    prompt = build_summarisation_prompt(job, RESULTS)

    assert prompt.count("SEARCH RESULTS:") == 1
    assert "Coursera" in prompt  # came from the snippet, not from the model


def test_summarisation_runs_on_the_cheap_tier_through_the_router(make_router, job):
    agent, _, router = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    agent.research(job)

    assert len(router.client("light").calls) == 1
    assert router.client("heavy").calls == []


# --- The brief ---------------------------------------------------------------


def test_the_brief_is_schema_valid_and_keeps_the_company_from_the_posting(
    make_router, job
):
    agent, _, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    brief = agent.research(job)

    assert isinstance(brief, CompanyBrief)
    assert CompanyBrief.model_validate(brief.model_dump()) == brief
    assert brief.company == "Arbisoft"
    assert len(brief.facts) == 3


def test_sources_are_preserved_for_every_cited_result(make_router, job):
    agent, _, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    brief = agent.research(job)

    assert [s.url for s in brief.sources] == [
        "https://arbisoft.com",
        "https://arbisoft.com/blog",
    ]
    assert brief.sources[0].title == "Arbisoft - Software Development Company"


def test_only_cited_results_become_sources(make_router, job):
    single_source = {**EXTRACTION, "facts": [EXTRACTION["facts"][0]]}
    agent, _, _ = build_agent(make_router, json.dumps(single_source), RESULTS)

    brief = agent.research(job)

    assert [s.url for s in brief.sources] == ["https://arbisoft.com"]


# --- Grounding ---------------------------------------------------------------


def test_a_fact_citing_no_real_result_is_dropped(make_router, job):
    collector = TraceCollector()
    invented = {
        **EXTRACTION,
        "facts": [
            *EXTRACTION["facts"],
            {"text": "Arbisoft was acquired by Google.", "source_index": 9},
        ],
    }
    agent, _, _ = build_agent(
        make_router, json.dumps(invented), RESULTS, collector=collector
    )

    brief = agent.research(job)

    assert not any("acquired" in fact for fact in brief.facts)
    guard = [e for e in collector.events if e.name == "research:grounding_guard"]
    assert guard and "uncited fact" in guard[0].result


def test_a_figure_absent_from_the_cited_result_is_dropped(make_router, job):
    collector = TraceCollector()
    invented = {
        **EXTRACTION,
        "facts": [
            *EXTRACTION["facts"],
            # Cites a real result, but the number was never in it.
            {"text": "Arbisoft has raised $250 million in funding.", "source_index": 0},
        ],
    }
    agent, _, _ = build_agent(
        make_router, json.dumps(invented), RESULTS, collector=collector
    )

    brief = agent.research(job)

    assert not any("250" in fact for fact in brief.facts)
    guard = [e for e in collector.events if e.name == "research:grounding_guard"]
    assert guard and "unsupported figure" in guard[0].result


def test_an_invented_figure_in_the_summary_is_stripped(make_router, job):
    fabricated = {
        **EXTRACTION,
        "summary": (
            "Arbisoft builds data platforms. It employs 4000 engineers worldwide."
        ),
    }
    agent, _, _ = build_agent(make_router, json.dumps(fabricated), RESULTS)

    brief = agent.research(job)

    assert "4000" not in brief.summary
    assert "Arbisoft builds data platforms." in brief.summary


def test_the_number_guard_ignores_formatting():
    source = "Founded in 2007, the company works with 17 clients."
    assert unsupported_numbers("Founded in 2007.", source) == []
    assert unsupported_numbers("Works with 17 clients.", source) == []
    assert unsupported_numbers("Raised $250 million.", source) == ["250"]


# --- Degrading honestly ------------------------------------------------------


def test_an_unnamed_company_is_not_searched_for_or_guessed(make_router, job):
    unnamed = job.model_copy(update={"company": "Unknown"})
    agent, tool, router = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    brief = agent.research(unnamed)

    assert tool.calls == []  # no "Unknown company" search
    assert router.client("light").calls == []  # and no model call either
    assert brief.summary == NO_COMPANY_SUMMARY
    assert brief.facts == [] and brief.sources == []
    assert CompanyBrief.model_validate(brief.model_dump()) == brief


def test_an_empty_result_set_produces_an_honest_brief(make_router, job):
    agent, _, router = build_agent(make_router, json.dumps(EXTRACTION), results=[])

    brief = agent.research(job)

    assert brief.facts == [] and brief.sources == []
    assert "No search results" in brief.summary
    assert router.client("light").calls == []  # nothing to summarise


def test_a_search_failure_degrades_instead_of_breaking_the_run(make_router, job):
    agent, _, _ = build_agent(
        make_router, json.dumps(EXTRACTION), error=SearchError("provider down")
    )

    brief = agent.research(job)

    assert "unavailable" in brief.summary
    assert "provider down" in brief.summary
    assert brief.sources == []


# --- Tracing, retry, context -------------------------------------------------


def test_the_search_call_and_the_model_call_are_both_traced(make_router, job):
    collector = TraceCollector()
    agent, _, _ = build_agent(
        make_router, json.dumps(EXTRACTION), RESULTS, collector=collector
    )

    agent.research(job)

    search, model = collector.events[0], collector.events[1]
    assert (search.name, search.kind, search.agent) == ("web_search", "tool", "research")
    assert search.duration_ms >= 0
    assert model.name == "model:light:summarise"
    assert (model.input_tokens, model.output_tokens) == (10, 5)


def test_malformed_output_is_retried_through_the_existing_guard(make_router, job):
    router = make_router(light_replies=["here's what I found...", json.dumps(EXTRACTION)])
    agent = ResearchAgent(router, tools=ToolRegistry([FakeSearchTool(RESULTS).as_tool()]))

    brief = agent.research(job)

    assert len(brief.facts) == 3
    stub = router.client("light")
    assert len(stub.calls) == 2
    assert "rejected by the schema validator" in stub.calls[1]["messages"][0]["content"]


def test_persistently_malformed_output_never_becomes_a_brief(make_router, job):
    router = make_router(light_replies=["nope", "nope", "nope"])
    agent = ResearchAgent(router, tools=ToolRegistry([FakeSearchTool(RESULTS).as_tool()]))

    with pytest.raises(RetryExhaustedError):
        agent.research(job)


def test_run_researches_from_the_run_context(make_router, job):
    context = RunContext(cv_text="cv", jd_text=job.raw_text)
    context.job = job
    agent, _, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    brief = agent.run(context)

    assert brief.company == "Arbisoft"
    assert context.warnings == []


def test_run_warns_when_research_produced_nothing(make_router, job):
    context = RunContext(cv_text="cv", jd_text=job.raw_text)
    context.job = job.model_copy(update={"company": "Unknown"})
    agent, _, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    agent.run(context)

    assert context.warnings == [NO_COMPANY_SUMMARY]


def test_run_refuses_to_research_an_unparsed_posting(make_router):
    agent, _, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    with pytest.raises(ValueError, match="parsed job description"):
        agent.run(RunContext(cv_text="cv", jd_text="jd"))

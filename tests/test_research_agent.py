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
    mentions_company,
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


def test_the_query_targets_the_company_not_the_job(job):
    # A live run searched "Arbisoft Junior AI Engineer Python FastAPI" and got
    # back job boards and strangers' profiles. The query is now about the
    # employer; role relevance is applied in the summariser instead.
    query = build_query(job)

    assert query == '"Arbisoft" company overview'
    assert "Junior AI Engineer" not in query
    assert "FastAPI" not in query


def test_the_company_name_is_quoted_so_it_stays_the_primary_target():
    job = JobDescription(
        role="Data Engineer", company="Acme Data Systems", raw_text="...", skills=["Go"]
    )

    query = build_query(job)

    assert query.startswith('"Acme Data Systems"')
    assert "Data Engineer" not in query.removeprefix('"Acme Data Systems"')


def test_the_query_holds_no_employer_specific_logic():
    for company in ("Arbisoft", "Netflix", "A Very Small Studio"):
        job = JobDescription(role="Engineer", company=company, raw_text="...")
        assert build_query(job) == f'"{company}" company overview'


# --- Relevance guard ---------------------------------------------------------


def test_results_that_never_mention_the_company_are_discarded(make_router, job):
    collector = TraceCollector()
    noise = [
        SearchResult(
            title="Laraib Arjamand - Software Engineer | Python | Django",
            url="https://example.com/profile",
            snippet="Software engineer working with Python and Django.",
        ),
        SearchResult(
            title="Senior Python AI Engineer (FastAPI) - Hybrid London",
            url="https://example.com/job",
            snippet="A vacancy at another employer paying $35/hr.",
        ),
    ]
    agent, _, _ = build_agent(
        make_router, json.dumps(EXTRACTION), [*RESULTS, *noise], collector=collector
    )

    brief = agent.research(job)

    # Only the two genuinely-about-Arbisoft results could be cited.
    assert {s.url for s in brief.sources} <= {r.url for r in RESULTS}
    guard = [e for e in collector.events if e.name == "research:relevance_guard"]
    assert guard and guard[0].arguments == {"kept": 2, "dropped": 2}


def test_irrelevant_results_never_reach_the_summariser(make_router, job):
    off_topic = SearchResult(
        title="Some other company's careers page",
        url="https://example.com/other",
        snippet="A totally unrelated employer hiring engineers.",
    )
    agent, _, router = build_agent(
        make_router, json.dumps(EXTRACTION), [RESULTS[0], off_topic]
    )

    agent.research(job)

    prompt = router.client("light").calls[0]["messages"][0]["content"]
    assert "totally unrelated employer" not in prompt


def test_a_result_set_with_nothing_about_the_company_degrades_honestly(
    make_router, job
):
    off_topic = [
        SearchResult(title="Unrelated", url="https://example.com", snippet="Nothing.")
    ]
    agent, _, router = build_agent(make_router, json.dumps(EXTRACTION), off_topic)

    brief = agent.research(job)

    assert brief.facts == [] and brief.sources == []
    assert "nothing that mentions the company" in brief.summary
    assert router.client("light").calls == []  # nothing worth summarising


def test_a_multi_word_company_matches_on_its_leading_word():
    long_name = SearchResult(
        title="Arbisoft engineering blog",
        url="https://arbisoft.com/blog",
        snippet="Engineering practice.",
    )

    assert mentions_company(long_name, "Arbisoft Pvt Ltd")
    assert not mentions_company(long_name, "Netflix")


def test_an_empty_company_name_matches_nothing():
    result = SearchResult(title="Anything", url="https://example.com", snippet="text")

    assert not mentions_company(result, "   ")


def test_a_very_short_company_name_is_not_matched_loosely():
    # A two-letter head would match almost anything; require a real word.
    result = SearchResult(title="Sonic boom", url="https://example.com", snippet="")

    assert not mentions_company(result, "So Ltd")


def test_the_search_tool_is_called_with_a_bounded_result_count(make_router, job):
    agent, tool, _ = build_agent(make_router, json.dumps(EXTRACTION), RESULTS)

    agent.research(job, max_results=3)

    (call,) = tool.calls
    assert "Arbisoft" in call["query"]
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

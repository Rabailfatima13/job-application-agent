"""Supervisor and the Week 6 end-to-end pipeline.

Everything is stubbed: the model is the existing `StubModelClient`, web search
is a fake tool. No test needs an API key, a network, or a real database beyond
tmp_path. What is verified is the pipeline's behaviour - order, state hand-off,
persistence, failure handling - not the components, which have their own tests.
"""

import json
from datetime import datetime
from itertools import count

import pytest

from job_agent.agents import ResearchAgent, ScoringAgent, Supervisor, build_supervisor
from job_agent.agents.supervisor import (
    PIPELINE_NODES,
    RESEARCH_FAILED_WARNING,
    new_application_id,
)
from job_agent.memory import InMemoryApplicationTracker, SQLiteApplicationTracker
from job_agent.models import (
    ApplicationRecord,
    ApplicationStatus,
    CompanyBrief,
    FitReport,
    JobDescription,
    ParsedCV,
)
from job_agent.observability import TraceCollector
from job_agent.tools import SearchError, Tool, ToolRegistry, build_parsing_tools
from job_agent.tools.web_search import SearchResult
from job_agent.validation import RetryExhaustedError

# --- Canned model replies, one per pipeline step -----------------------------

CV_EXTRACTION = {
    "candidate_name": "Mahnoor Rauf",
    "evidence": [
        {
            "text": (
                "Built a Task Management REST API with FastAPI, SQLAlchemy and "
                "SQLite, with Pydantic validation and pytest coverage."
            ),
            "section": "Experience",
            "skills": ["FastAPI", "pytest"],
        },
        {
            "text": (
                "Built a research agent with tool calling, session memory and "
                "pre/post-tool hooks."
            ),
            "section": "Experience",
            "skills": ["tool calling"],
        },
    ],
    "skills": ["Python", "FastAPI", "pytest", "SQLite"],
}

JD_EXTRACTION = {
    "role": "Junior AI Engineer",
    "company": "Arbisoft",
    "location": "Lahore, hybrid",
    "requirements": [
        {"text": "Strong Python", "must_have": True},
        {"text": "Experience building REST APIs", "must_have": True},
        {"text": "Testing discipline (pytest)", "must_have": True},
        {"text": "Kubernetes / cloud deployment experience", "must_have": False},
    ],
    "responsibilities": ["Help build LLM-backed product features"],
    "skills": ["Python", "FastAPI", "pytest"],
}

RESEARCH_EXTRACTION = {
    "summary": "Arbisoft builds data platforms and machine learning products.",
    "facts": [{"text": "Builds data platforms.", "source_index": 0}],
}

SCORING_JUDGEMENT = {
    "judgements": [
        {"requirement_index": 0, "met": True, "evidence_index": 0},
        {"requirement_index": 1, "met": True, "evidence_index": 0},
        {"requirement_index": 2, "met": True, "evidence_index": 0},
        {"requirement_index": 3, "met": False, "evidence_index": None},
    ],
    "recommended_emphasis": [0, 1],
}

# 3 must-haves met (3.0) out of 3.0 + 0.5 nice-to-have = 0.857
EXPECTED_FIT = 0.857

SEARCH_RESULTS = [
    SearchResult(
        title="Arbisoft",
        url="https://arbisoft.com",
        snippet="Arbisoft builds data platforms and machine learning products.",
    )
]


def fake_search_tool(results=SEARCH_RESULTS, error: Exception | None = None) -> Tool:
    def run(query, max_results=None):
        if error is not None:
            raise error
        return list(results)

    return Tool(
        name="web_search",
        description="Search the web.",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
        run=run,
    )


@pytest.fixture
def pipeline(make_router):
    """Build a supervisor with every dependency injected and controllable.

    The id counter is shared across supervisors built by one test, so two runs
    against the same tracker get distinct ids just as real uuids would.
    """
    ids = count(1)

    def _build(
        tracker=None,
        jd_extraction=None,
        research_reply=None,
        scoring_reply=None,
        search_error=None,
        collector=None,
    ):
        collector = collector or TraceCollector()
        router = make_router(
            # parse_cv, parse_jd and the research summary all run light...
            light_replies=[
                json.dumps(CV_EXTRACTION),
                json.dumps(jd_extraction or JD_EXTRACTION),
                research_reply or json.dumps(RESEARCH_EXTRACTION),
            ],
            # ...and scoring runs heavy.
            heavy_replies=[scoring_reply or json.dumps(SCORING_JUDGEMENT)],
        )
        registry = ToolRegistry(
            [
                *build_parsing_tools(router, collector),
                fake_search_tool(error=search_error),
            ]
        )
        return Supervisor(
            tools=registry,
            research=ResearchAgent(router, tools=registry, collector=collector),
            scoring=ScoringAgent(router, collector=collector),
            tracker=tracker if tracker is not None else InMemoryApplicationTracker(),
            collector=collector,
            id_factory=lambda: f"app-{next(ids)}",
            clock=lambda: datetime(2026, 8, 12, 9, 0),
        )

    return _build


# --- Happy path --------------------------------------------------------------


def test_the_pipeline_produces_every_stage_output(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()
    supervisor = pipeline(tracker=tracker)

    context = supervisor.run(sample_cv_text, sample_jd_text)

    assert isinstance(context.parsed_cv, ParsedCV)
    assert isinstance(context.job, JobDescription)
    assert isinstance(context.brief, CompanyBrief)
    assert isinstance(context.fit_report, FitReport)
    assert isinstance(context.application, ApplicationRecord)
    assert context.warnings == []
    # ...and the application really was persisted, not just returned.
    assert tracker.get(context.application.application_id) is not None


def test_each_stage_output_carries_the_expected_content(
    pipeline, sample_cv_text, sample_jd_text
):
    context = pipeline().run(sample_cv_text, sample_jd_text)

    assert context.parsed_cv.candidate_name == "Mahnoor Rauf"
    assert (context.job.role, context.job.company) == ("Junior AI Engineer", "Arbisoft")
    assert context.brief.sources[0].url == "https://arbisoft.com"
    assert context.fit_report.overall_fit == EXPECTED_FIT
    assert context.fit_report.gaps == ["Kubernetes / cloud deployment experience"]
    assert context.application.fit_score == EXPECTED_FIT


def test_the_application_record_is_built_from_the_validated_fit_report(
    pipeline, sample_cv_text, sample_jd_text
):
    context = pipeline().run(sample_cv_text, sample_jd_text)
    record = context.application

    assert record.role == context.fit_report.role
    assert record.company == context.fit_report.company
    assert record.fit_score == context.fit_report.overall_fit
    assert record.created_at == datetime(2026, 8, 12, 9, 0)


# --- Human in the loop -------------------------------------------------------


def test_every_application_starts_as_a_draft_and_is_never_submitted(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()

    pipeline(tracker=tracker).run(sample_cv_text, sample_jd_text)

    (record,) = tracker.list()
    assert record.status is ApplicationStatus.draft
    assert tracker.list(status=ApplicationStatus.submitted) == []


# --- Persistence and multiple applications -----------------------------------


def test_the_application_survives_in_a_real_sqlite_tracker(
    pipeline, tmp_path, sample_cv_text, sample_jd_text
):
    db_path = tmp_path / "applications.db"
    supervisor = pipeline(tracker=SQLiteApplicationTracker(db_path))

    context = supervisor.run(sample_cv_text, sample_jd_text)

    # A brand new tracker instance, as a later process would open it.
    reopened = SQLiteApplicationTracker(db_path).get(context.application.application_id)
    assert reopened is not None
    assert reopened.role == "Junior AI Engineer"
    assert reopened.fit_score == EXPECTED_FIT
    assert reopened.status is ApplicationStatus.draft


def test_one_cv_against_two_roles_produces_two_tracked_applications(
    pipeline, tmp_path, sample_cv_text, sample_jd_text
):
    # The proposal's "one CV, many jobs" requirement (FR-13), including a
    # second role at the *same* company - so the id cannot be role/company.
    db_path = tmp_path / "applications.db"
    tracker = SQLiteApplicationTracker(db_path)
    second_jd = {
        **JD_EXTRACTION,
        "role": "Data Engineer",
        "requirements": JD_EXTRACTION["requirements"][:2],
    }

    first = pipeline(tracker=tracker).run(sample_cv_text, sample_jd_text)
    second = pipeline(tracker=tracker, jd_extraction=second_jd).run(
        sample_cv_text, sample_jd_text
    )

    assert first.application.application_id != second.application.application_id
    assert first.application.role == "Junior AI Engineer"
    assert second.application.role == "Data Engineer"
    assert second.application.company == first.application.company
    # The second posting has only the two met must-haves, so it scores 1.0.
    assert second.application.fit_score == 1.0

    persisted = SQLiteApplicationTracker(db_path).list()
    assert len(persisted) == 2
    assert {r.role for r in persisted} == {"Junior AI Engineer", "Data Engineer"}


def test_application_ids_are_unique_and_not_derived_from_the_role():
    ids = {new_application_id() for _ in range(100)}
    assert len(ids) == 100


# --- Research degradation ----------------------------------------------------


def test_a_search_failure_does_not_stop_the_fit_analysis(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()
    supervisor = pipeline(tracker=tracker, search_error=SearchError("provider down"))

    context = supervisor.run(sample_cv_text, sample_jd_text)

    assert context.fit_report.overall_fit == EXPECTED_FIT
    assert tracker.get(context.application.application_id) is not None
    # The reason is preserved for the user rather than swallowed.
    assert any("unavailable" in warning for warning in context.warnings)
    assert context.brief.sources == []


def test_an_unresearchable_summary_still_yields_a_tracked_application(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()
    # The summariser never returns valid JSON, so research exhausts its retries.
    supervisor = pipeline(tracker=tracker, research_reply="not json")

    context = supervisor.run(sample_cv_text, sample_jd_text)

    assert context.brief is None
    assert RESEARCH_FAILED_WARNING in context.warnings
    assert context.fit_report is not None
    assert tracker.get(context.application.application_id) is not None


def test_a_posting_without_a_company_is_neither_searched_nor_guessed(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()
    anonymous = {**JD_EXTRACTION, "company": None}
    supervisor = pipeline(tracker=tracker, jd_extraction=anonymous)

    context = supervisor.run(sample_cv_text, sample_jd_text)

    assert context.job.company == "Unknown"
    assert context.brief.facts == [] and context.brief.sources == []
    assert context.warnings  # says why there is no research
    assert context.application.company == "Unknown"
    assert tracker.get(context.application.application_id) is not None


# --- Scoring failure ---------------------------------------------------------


def test_a_scoring_failure_writes_no_application_and_surfaces_clearly(
    pipeline, sample_cv_text, sample_jd_text
):
    tracker = InMemoryApplicationTracker()
    supervisor = pipeline(tracker=tracker, scoring_reply="they seem good")

    with pytest.raises(RetryExhaustedError):
        supervisor.run(sample_cv_text, sample_jd_text)

    # No invented score, no half-written record.
    assert tracker.list() == []


# --- Input handling ----------------------------------------------------------


def test_the_pipeline_accepts_raw_text_for_both_inputs(
    pipeline, sample_cv_text, sample_jd_text
):
    context = pipeline().run(sample_cv_text, sample_jd_text)

    assert context.parsed_cv.raw_text == sample_cv_text.strip()
    assert context.job.raw_text == sample_jd_text.strip()


def test_the_pipeline_accepts_file_paths_for_both_inputs(
    pipeline, fixtures_dir, sample_cv_text, sample_jd_text
):
    context = pipeline().run(
        fixtures_dir / "sample_cv.txt", fixtures_dir / "sample_jd.txt"
    )

    # The existing document loader read the files; no second input system.
    assert context.parsed_cv.raw_text == sample_cv_text.strip()
    assert context.job.raw_text == sample_jd_text.strip()
    assert context.application is not None


# --- Ordering, state, tracing ------------------------------------------------


def test_the_stages_run_in_the_intended_order(pipeline, sample_cv_text, sample_jd_text):
    collector = TraceCollector()

    pipeline(collector=collector).run(sample_cv_text, sample_jd_text)

    stages = [e.name for e in collector.events if e.name in PIPELINE_NODES]
    assert stages == list(PIPELINE_NODES)


def test_the_run_is_traced_down_to_the_individual_tool_and_model_calls(
    pipeline, sample_cv_text, sample_jd_text
):
    collector = TraceCollector()

    pipeline(collector=collector).run(sample_cv_text, sample_jd_text)

    names = [e.name for e in collector.events]
    assert "web_search" in names
    assert "model:light:parse_cv" in names
    assert "model:light:parse_jd" in names
    assert "model:light:summarise" in names
    assert "model:heavy:score_fit" in names

    totals = collector.totals()
    assert totals["errors"] == 0
    assert totals["input_tokens"] > 0  # Week 8 has real numbers to report
    assert totals["duration_ms"] >= 0


def test_state_written_by_one_node_is_visible_to_the_next(
    pipeline, sample_cv_text, sample_jd_text
):
    context = pipeline().run(sample_cv_text, sample_jd_text)

    # Scoring could only have produced these requirements from the parsed JD,
    # and this evidence from the parsed CV - so state survived the hand-off.
    scored = [m.requirement for m in context.fit_report.requirement_matches]
    assert scored == [r.text for r in context.job.requirements]
    evidence = [m.evidence for m in context.fit_report.requirement_matches if m.met]
    assert all(text in context.parsed_cv.raw_text for text in evidence)


# --- The graph itself --------------------------------------------------------


def test_the_graph_builds_with_the_expected_nodes(pipeline):
    graph = pipeline().graph

    assert set(PIPELINE_NODES) <= set(graph.get_graph().nodes)


def test_the_graph_runs_to_completion_on_the_happy_path(
    pipeline, sample_cv_text, sample_jd_text
):
    context = pipeline().run(sample_cv_text, sample_jd_text)

    # Reaching END means the final node ran and its state came back out.
    assert context.application is not None
    assert isinstance(context, type(context))


def test_a_node_failure_propagates_out_of_the_graph(
    pipeline, sample_cv_text, sample_jd_text
):
    supervisor = pipeline(jd_extraction=None, scoring_reply="nope")

    with pytest.raises(RetryExhaustedError):
        supervisor.run(sample_cv_text, sample_jd_text)


# --- Dependency injection ----------------------------------------------------


def test_the_supervisor_never_opens_a_database_of_its_own(
    pipeline, tmp_path, monkeypatch, sample_cv_text, sample_jd_text
):
    import job_agent.agents.supervisor as supervisor_module

    def fail(*args, **kwargs):  # pragma: no cover - only runs if the test fails
        raise AssertionError("Supervisor must not construct a tracker itself")

    monkeypatch.setattr(supervisor_module, "SQLiteApplicationTracker", fail)
    tracker = InMemoryApplicationTracker()

    context = pipeline(tracker=tracker).run(sample_cv_text, sample_jd_text)

    assert tracker.get(context.application.application_id) is not None
    assert not list(tmp_path.glob("*.db"))


def test_the_production_factory_wires_sqlite_and_web_search(settings):
    # build_supervisor is the only place that knows those choices; it needs no
    # network or API call to construct them.
    supervisor = build_supervisor(settings)

    assert isinstance(supervisor.tracker, SQLiteApplicationTracker)
    assert supervisor.tracker.db_path == settings.tracker_db_path
    assert set(supervisor.tools.names()) == {"parse_cv", "parse_jd", "web_search"}
    assert supervisor.collector is not None

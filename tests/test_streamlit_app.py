"""Streamlit demo surface (FR-15, NFR-6).

Driven with Streamlit's own `AppTest`, which runs the script in-process - no
browser, no server, and no provider call: `run_pipeline` is patched in every
test, so the suite consumes no tokens.

What is under test is the surface's contract: it calls the real pipeline entry
point with what the user typed, renders what comes back, degrades readably when
things are missing, and never puts a credential on screen.
"""

from datetime import datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from job_agent.app import ui as streamlit_app
from job_agent.config import ModelConfig, Settings
from job_agent.memory.session import RunContext
from job_agent.models import (
    ApplicationRecord,
    ApplicationStatus,
    CompanyBrief,
    CoverLetter,
    CVEvidence,
    FitReport,
    JobDescription,
    MatchLevel,
    ParsedCV,
    RequirementMatch,
    RoleRequirement,
    Source,
    TailoredCV,
)
from job_agent.observability import TraceCollector, TraceEvent

# AppTest resolves relative paths against this file, so be explicit.
APP = str(
    Path(__file__).resolve().parent.parent / "job_agent" / "app" / "streamlit_app.py"
)
CV_TEXT = "Built a Task Management REST API with FastAPI and SQLite."
JD_TEXT = "Junior AI Engineer at Arbisoft. Requirements: Strong Python."


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        # The provider names the adapters actually recognise - `tier_problem`
        # mirrors `llm.providers.build_client`, so a made-up name here would
        # (correctly) disable the Run button.
        heavy=ModelConfig(
            provider="anthropic", model="big-model", api_key="HEAVY-SECRET"
        ),
        light=ModelConfig(
            provider="openai_compatible",
            model="small-model",
            api_key="LIGHT-SECRET",
            base_url="https://example.invalid/v1",
        ),
        search_provider="serpapi",
        search_api_key="SEARCH-SECRET",
        tracker_db_path=tmp_path / "applications.db",
        tool_call_log_path=tmp_path / "tool_calls.log",
    )


def finished_context() -> RunContext:
    """A RunContext shaped exactly like a successful run's."""
    evidence = CVEvidence(text=CV_TEXT, section="Experience")
    context = RunContext(cv_text=CV_TEXT, jd_text=JD_TEXT)
    context.parsed_cv = ParsedCV(
        candidate_name="Mahnoor Rauf",
        raw_text=CV_TEXT,
        evidence=[evidence],
        skills=["Python"],
    )
    context.job = JobDescription(
        role="Junior AI Engineer",
        company="Arbisoft",
        raw_text=JD_TEXT,
        requirements=[
            RoleRequirement(text="Strong Python"),
            RoleRequirement(text="Kubernetes experience"),
        ],
    )
    context.brief = CompanyBrief(
        company="Arbisoft",
        summary="Arbisoft builds data platforms.",
        facts=["Builds data platforms."],
        sources=[Source(title="Arbisoft", url="https://arbisoft.com")],
    )
    context.fit_report = FitReport(
        role="Junior AI Engineer",
        company="Arbisoft",
        overall_fit=0.75,
        requirement_matches=[
            RequirementMatch(
                requirement="Strong Python",
                evidence=CV_TEXT,
                match_level=MatchLevel.match,
                met=True,
            ),
            # Every requirement gets a match, missing ones included - the
            # real scoring agent never produces a bare gap without one, so
            # the fixture shouldn't either (see `_resolve_matches`).
            RequirementMatch(
                requirement="Kubernetes experience",
                match_level=MatchLevel.missing,
                met=False,
                reason="the CV never mentions Kubernetes",
            ),
        ],
        gaps=["Kubernetes experience"],
        recommended_emphasis=[CV_TEXT],
    )
    context.tailored_cv = TailoredCV(
        role="Junior AI Engineer",
        company="Arbisoft",
        bullets=["Built a Task Management REST API with FastAPI."],
        omitted=["BS Computer Science"],
    )
    context.cover_letter = CoverLetter(
        role="Junior AI Engineer",
        company="Arbisoft",
        body="Dear Hiring Team,\n\nI built a REST API.\n\nSincerely,\nMahnoor Rauf",
    )
    context.application = ApplicationRecord(
        application_id="app-1",
        role="Junior AI Engineer",
        company="Arbisoft",
        fit_score=0.75,
        status=ApplicationStatus.draft,
        created_at=datetime(2026, 8, 14, 9, 0),
    )
    collector = TraceCollector()
    # Shaped like a real trace: the supervisor's stage spans *contain* the model
    # calls traced below them, so the two sets of durations overlap. Summing
    # both would report 8.1s for a run that took 7.2s - see `run_totals`.
    for name in ("parse_cv", "parse_jd", "research", "score_fit", "write_application",
                 "track_application"):
        collector.record(TraceEvent(agent="supervisor", name=name, duration_ms=1200.0))
    collector.record(
        TraceEvent(
            agent="parse_cv",
            name="model:light:parse_cv",
            kind="model",
            duration_ms=900.0,
            input_tokens=400,
            output_tokens=120,
        )
    )
    collector.record(
        TraceEvent(
            agent="writing",
            name="writing:grounding_guard",
            status="success",
            result=(
                "- Managed Kubernetes clusters.\n"
                "  -> the CV never mentions Kubernetes"
            ),
        )
    )
    collector.record(
        TraceEvent(agent="writing", name="validate:WritingDraft", status="error")
    )
    context.trace = collector
    return context


def rendered_text(app) -> str:
    """Everything the user would actually see.

    `str(app)` is a structural repr, not the rendered output - asserting
    against it silently passes (or fails) for the wrong reasons. This walks the
    element collections instead, including json blocks (which is how
    `st.write(dict)` renders) and metric labels.
    """
    parts: list[str] = []
    for name in (
        "title", "header", "subheader", "markdown", "caption",
        "success", "warning", "info", "error", "text", "code",
    ):
        for element in getattr(app, name, []):
            parts.append(str(getattr(element, "value", "")))
    for element in getattr(app, "metric", []):
        parts += [str(element.label), str(element.value)]
    for element in getattr(app, "json", []):
        parts.append(str(element.value))
    for element in getattr(app, "table", []):
        parts.append(str(getattr(element, "value", "")))
    for element in getattr(app, "download_button", []):
        parts.append(str(element.label))
    return "\n".join(parts)


class Upload:
    """The bits of Streamlit's UploadedFile the view actually touches."""

    def __init__(self, name: str, data: bytes = b"from the file") -> None:
        self.name = name
        self._data = b"%PDF-1.4 fake" if name.lower().endswith(".pdf") else data

    def getvalue(self) -> bytes:
        return self._data


def run_app(settings, pipeline=None, monkeypatch=None) -> AppTest:
    """Start the app with configuration and the pipeline both stubbed."""
    app = AppTest.from_file(APP, default_timeout=30)
    # Settings reach the app through the patched loader only - putting them in
    # session_state would dump them into AppTest's repr, which is where the
    # test would then "find" a leaked credential that the UI never rendered.
    monkeypatch.setattr(streamlit_app, "load_settings", lambda *a, **k: settings)
    if pipeline is not None:
        monkeypatch.setattr(streamlit_app, "run_pipeline", pipeline)
    return app


# --- import and startup ------------------------------------------------------


def test_the_app_imports_and_renders(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert not app.exception
    assert "Job Application Agent" in app.title[0].value


def test_the_sidebar_shows_models_but_never_a_credential(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "small-model" in rendered and "big-model" in rendered
    for secret in ("HEAVY-SECRET", "LIGHT-SECRET", "SEARCH-SECRET"):
        assert secret not in rendered


def test_a_missing_search_key_is_explained_without_blocking(tmp_path, monkeypatch):
    app = run_app(unconfigured(tmp_path), monkeypatch=monkeypatch).run()

    assert any("company research is skipped" in i.value for i in app.info)


# --- input handling ----------------------------------------------------------


def test_empty_input_is_refused_before_the_pipeline_runs(settings, monkeypatch):
    calls = []

    def pipeline(cv, jd, s):  # pragma: no cover - must never be reached
        calls.append((cv, jd))
        raise AssertionError("the pipeline must not run without inputs")

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.button[0].click().run()

    assert calls == []
    assert any("provide both" in e.value for e in app.error)
    assert not app.exception


def test_a_cv_without_a_job_description_is_refused(settings, monkeypatch):
    app = run_app(settings, pipeline=lambda *a: None, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.button[0].click().run()

    assert any("provide both" in e.value for e in app.error)


def test_the_pipeline_is_called_with_exactly_what_the_user_typed(
    settings, monkeypatch
):
    seen = {}

    def pipeline(cv, jd, s):
        seen["cv"], seen["jd"], seen["settings"] = cv, jd, s
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert seen["cv"] == CV_TEXT
    assert seen["jd"] == JD_TEXT
    assert seen["settings"] is settings


def test_surrounding_whitespace_is_trimmed(settings, monkeypatch):
    seen = {}

    def pipeline(cv, jd, s):
        seen["cv"] = cv
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(f"  {CV_TEXT}  ")
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert seen["cv"] == CV_TEXT


# --- rendering a successful run ----------------------------------------------


@pytest.fixture
def completed(settings, monkeypatch) -> AppTest:
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    return app.button[0].click().run()


def test_a_successful_run_renders_without_error(completed):
    assert not completed.exception
    assert any("Run complete" in s.value for s in completed.success)


def test_the_fit_score_and_report_are_shown(completed):
    rendered = rendered_text(completed)
    assert "75%" in rendered
    assert "Strong Python" in rendered
    assert "Match" in rendered  # the match-level label, not just a checkmark
    assert "Kubernetes experience" in rendered  # the gap
    assert CV_TEXT in rendered  # the evidence behind the match


def test_the_fit_report_is_grouped_into_four_sections(completed):
    rendered = rendered_text(completed)
    assert "Strengths / Matches" in rendered
    assert "Missing from CV" in rendered
    # Ordered strongest-first: a match appears before the missing section.
    assert rendered.index("Strengths / Matches") < rendered.index("Missing from CV")


def test_a_missing_requirement_uses_the_fixed_careful_wording_not_the_models_own(
    completed,
):
    # The fixture's missing requirement carries its own reason ("the CV never
    # mentions Kubernetes") - that text legitimately appears elsewhere on the
    # page (a real, unrelated grounding rejection in the Validation section),
    # so this checks the fit-report caption directly rather than whole-page
    # text: the caption immediately under the missing requirement must be the
    # fixed wording, not the model's own reason string.
    captions = [c.value for c in completed.caption]
    assert streamlit_app.MISSING_FROM_CV_CAPTION in captions
    assert "the CV never mentions Kubernetes" not in captions

    rendered = rendered_text(completed)
    assert "you don't have this skill" not in rendered.lower()


def test_a_section_with_no_requirements_at_that_level_is_not_rendered(completed):
    # This fixture has no partial or related requirements at all - those
    # headings must not appear as empty sections.
    rendered = rendered_text(completed)
    assert "Partial matches" not in rendered
    assert "Related evidence" not in rendered


def test_partial_and_related_sections_show_the_models_own_reason(settings, monkeypatch):
    # Unlike "missing", partial/related evidence is real - the model's own
    # explanation is informative and should reach the reader unchanged.
    context = finished_context()
    context.fit_report.requirement_matches.extend(
        [
            RequirementMatch(
                requirement="Data visualization",
                evidence="Basic knowledge on SQL and Python.",
                match_level=MatchLevel.partial,
                met=False,
                reason="the CV shows basic Python, not the visualization "
                "libraries the role asks for",
            ),
            RequirementMatch(
                requirement="Analytical and problem-solving skills",
                evidence="Data Structures, Calculus",
                match_level=MatchLevel.related,
                met=False,
                reason="coursework is related but does not itself demonstrate "
                "professional analytical experience",
            ),
        ]
    )

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    rendered = rendered_text(app)
    assert "Partial matches" in rendered
    assert "Related evidence" in rendered
    assert "not the visualization libraries the role asks for" in rendered
    assert "does not itself demonstrate professional analytical experience" in rendered


def test_the_pipeline_stages_are_shown_from_the_real_trace(completed):
    rendered = rendered_text(completed)
    for label in ("Parse CV", "Research the company", "Score candidate-role fit",
                  "Tailor CV + draft cover letter", "Save to the tracker"):
        assert label in rendered


def test_the_tailored_cv_and_cover_letter_are_shown_and_downloadable(completed):
    rendered = rendered_text(completed)
    assert "Built a Task Management REST API with FastAPI." in rendered
    assert "Dear Hiring Team," in rendered
    labels = [button.label for button in completed.download_button]
    assert "Download tailored CV" in labels
    assert "Download cover letter" in labels


def test_research_sources_are_shown(completed):
    rendered = rendered_text(completed)
    assert "Arbisoft builds data platforms." in rendered
    assert "https://arbisoft.com" in rendered


def test_grounding_decisions_are_explained_in_plain_words(completed):
    rendered = rendered_text(completed)
    assert "Draft rejected: claims not supported by the CV" in rendered
    assert "the CV never mentions Kubernetes" in rendered
    assert "regenerated" in rendered  # the schema retry is reported too


def test_the_tracked_application_and_its_status_are_shown(completed):
    rendered = rendered_text(completed)
    assert "app-1" in rendered
    assert "draft" in rendered
    assert "Nothing is submitted anywhere" in rendered


def test_a_clean_grounding_pass_says_so(settings, monkeypatch):
    """The happy path: a run whose writer never triggers a guard. Every other
    grounding test exercises a rejection - this is the one that proves the
    "nothing needed correcting" message actually renders, which is the more
    likely outcome on a real successful run."""
    context = finished_context()
    clean = TraceCollector()
    for event in context.trace.events:
        if event.name not in streamlit_app.GUARDS:
            clean.record(event)
    context.trace = clean

    app = run_app(settings, pipeline=lambda *a: context, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert any(
        "Every generated claim was supported by your CV" in s.value
        for s in app.success
    )


def test_the_tracker_table_shows_every_real_row(settings, monkeypatch):
    """The `st.table` branch: every other test's tracker starts empty, so a
    real tracked row has never actually been proven to render. A mocked
    `run_pipeline` never touches the real tracker DB - `context.application`
    is rendered directly from the canned context above the table, but the
    table itself only ever reflects what is really on disk - so this seeds
    two real rows rather than relying on the run itself to add one."""
    from job_agent.memory import SQLiteApplicationTracker

    tracker = SQLiteApplicationTracker(settings.tracker_db_path)
    tracker.add(
        ApplicationRecord(
            application_id="earlier-app",
            role="Backend Engineer",
            company="Prior Co",
            fit_score=0.5,
            status=ApplicationStatus.draft,
            created_at=datetime(2026, 8, 1, 9, 0),
        )
    )
    tracker.add(
        ApplicationRecord(
            application_id="later-app",
            role="Data Engineer",
            company="Later Co",
            fit_score=0.9,
            status=ApplicationStatus.draft,
            created_at=datetime(2026, 8, 10, 9, 0),
        )
    )

    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    rendered = rendered_text(app)
    assert "All tracked applications (2)" in rendered
    assert "Prior Co" in rendered
    assert "Later Co" in rendered


# --- degraded runs -----------------------------------------------------------


def test_a_run_without_documents_explains_itself(settings, monkeypatch):
    context = finished_context()
    context.tailored_cv = None
    context.cover_letter = None
    context.warnings = ["Tailored documents could not be produced."]

    app = run_app(settings, pipeline=lambda *a: context, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert not app.exception
    assert any("could not be produced" in w.value for w in app.warning)
    assert any("your original CV stands" in w.value for w in app.warning)


def test_a_run_without_research_still_renders(settings, monkeypatch):
    context = finished_context()
    context.brief = None

    app = run_app(settings, pipeline=lambda *a: context, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert not app.exception
    assert any("No company research" in i.value for i in app.info)


# --- failures ----------------------------------------------------------------


def test_a_pipeline_failure_is_reported_without_crashing(settings, monkeypatch):
    def explode(*_args):
        raise RuntimeError("provider unavailable")

    app = run_app(settings, pipeline=explode, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    assert not app.exception  # the app survives
    message = "\n".join(e.value for e in app.error)
    assert "could not be completed" in message
    assert "provider unavailable" in message
    assert "Traceback" not in message


def test_a_failure_never_shows_a_credential(settings, monkeypatch):
    def explode(*_args):
        raise RuntimeError(
            "401 from https://api.example/v1?api_key=SEARCH-SECRET (key LIGHT-SECRET)"
        )

    app = run_app(settings, pipeline=explode, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    rendered = rendered_text(app)
    for secret in ("SEARCH-SECRET", "LIGHT-SECRET", "HEAVY-SECRET"):
        assert secret not in rendered
    assert "***" in rendered  # redacted, not silently swallowed


# --- helpers -----------------------------------------------------------------


def test_redact_removes_every_configured_secret(settings):
    text = "keys: HEAVY-SECRET LIGHT-SECRET SEARCH-SECRET"

    assert streamlit_app.redact(text, settings) == "keys: *** *** ***"


def test_redact_leaves_ordinary_text_alone(settings):
    assert streamlit_app.redact("nothing to hide", settings) == "nothing to hide"


def test_safe_error_is_one_line_and_redacted(settings):
    message = streamlit_app.safe_error(RuntimeError("boom LIGHT-SECRET"), settings)

    assert message == "RuntimeError: boom ***"


def test_an_uploaded_text_file_wins_over_the_pasted_box(tmp_path):
    assert streamlit_app.read_upload(Upload("cv.txt"), "pasted", tmp_path) == (
        "from the file"
    )
    assert streamlit_app.read_upload(None, "  pasted  ", tmp_path) == "pasted"


def test_an_uploaded_pdf_is_handed_to_the_pipeline_as_a_path(tmp_path):
    source = streamlit_app.read_upload(Upload("cv.pdf"), "", tmp_path)

    # The UI does not parse documents - the pipeline's own loader does.
    assert source.endswith(".pdf")
    assert Path(source).read_bytes() == b"%PDF-1.4 fake"


# --- results survive a rerun -------------------------------------------------
#
# Streamlit re-executes the whole script on every interaction, and on that
# rerun the Run button reads False. Holding the finished RunContext in a local
# variable therefore threw away a minute of real model spend the moment the
# user touched anything - including the download buttons put there for them.


def test_results_survive_a_download_click(completed):
    assert "Dear Hiring Team," in rendered_text(completed)  # precondition

    after = completed.download_button[0].click().run()

    rendered = rendered_text(after)
    assert "Dear Hiring Team," in rendered
    assert "Strong Python" in rendered
    assert "75%" in rendered


def test_results_survive_a_second_download_click(completed):
    after = completed.download_button[1].click().run()

    assert "Built a Task Management REST API with FastAPI." in rendered_text(after)


def test_results_survive_editing_the_cv_box(completed):
    assert "Dear Hiring Team," in rendered_text(completed)  # precondition

    after = completed.text_area("cv_text").set_value(f"{CV_TEXT} And more.").run()

    assert "Dear Hiring Team," in rendered_text(after)


def test_results_survive_editing_the_job_description_box(completed):
    after = completed.text_area("jd_text").set_value(f"{JD_TEXT} And more.").run()

    assert "Dear Hiring Team," in rendered_text(after)


def test_nothing_is_rendered_before_the_first_run(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "Fit report" not in rendered
    assert "Run complete" not in rendered
    assert not app.download_button  # nothing to download yet


def test_a_second_run_replaces_the_first_result(settings, monkeypatch):
    second = finished_context()
    second.fit_report.overall_fit = 0.42
    second.cover_letter.body = "Dear Second Team,"
    contexts = iter([finished_context(), second])

    app = run_app(
        settings, pipeline=lambda *a: next(contexts), monkeypatch=monkeypatch
    ).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()
    assert "75%" in rendered_text(app)

    app.button[0].click().run()

    rendered = rendered_text(app)
    assert "42%" in rendered
    assert "Dear Second Team," in rendered
    # The old result is replaced, not stacked underneath the new one.
    assert "Dear Hiring Team," not in rendered


def test_a_failed_run_does_not_leave_the_previous_result_on_screen(
    settings, monkeypatch
):
    """The dangerous case: a stale fit report sitting under an error message
    reads as though it belongs to the run that just failed."""
    state = {"fail": False}

    def pipeline(*_args):
        if state["fail"]:
            raise RuntimeError("provider unavailable")
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()
    assert "Dear Hiring Team," in rendered_text(app)

    state["fail"] = True
    app.button[0].click().run()

    rendered = rendered_text(app)
    assert "could not be completed" in rendered
    assert "Dear Hiring Team," not in rendered
    assert "75%" not in rendered


def test_an_incomplete_click_keeps_the_previous_result(settings, monkeypatch):
    """Clearing the CV box and clicking Run does not start a run, so there is
    nothing to invalidate - destroying the last result would just punish a
    misclick."""
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("cv_text").set_value(CV_TEXT)
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button[0].click().run()

    app.text_area("cv_text").set_value("")
    app.button[0].click().run()

    rendered = rendered_text(app)
    assert "provide both" in rendered
    assert "Dear Hiring Team," in rendered


# --- latency is not double-counted -------------------------------------------


def test_nested_events_are_not_double_counted():
    """A stage span contains the model call traced inside it. Adding both
    reports roughly twice the time the run actually took."""
    collector = TraceCollector()
    collector.record(
        TraceEvent(
            agent="parse_cv",
            name="model:light:parse_cv",
            kind="model",
            duration_ms=900.0,
            input_tokens=400,
            output_tokens=120,
        )
    )
    collector.record(
        TraceEvent(agent="supervisor", name="parse_cv", duration_ms=1000.0)
    )
    context = RunContext(cv_text=CV_TEXT, jd_text=JD_TEXT, trace=collector)

    totals = streamlit_app.run_totals(context)

    assert collector.totals()["duration_ms"] == 1900.0  # the raw sum overcounts
    assert totals["elapsed_ms"] == 1000.0  # the stage span alone
    assert totals["call_ms"] == 900.0  # time spent inside the call
    assert totals["calls"] == 1  # the wrapper is not itself a call
    # Tokens were never double-counted: a stage wrapper carries zero, so these
    # come straight from the collector.
    assert totals["input_tokens"] == 400
    assert totals["output_tokens"] == 120


def test_run_totals_survives_a_trace_with_no_stage_events():
    context = RunContext(cv_text=CV_TEXT, jd_text=JD_TEXT, trace=TraceCollector())

    assert streamlit_app.run_totals(context) == {
        "elapsed_ms": 0,
        "call_ms": 0,
        "calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }


def test_the_displayed_total_time_is_the_elapsed_time(completed):
    """Six 1.2s stages containing a 0.9s model call: 7.2s elapsed, not 8.1s."""
    values = [element.value for element in completed.metric]

    assert "7.2s" in values
    assert "8.1s" not in values


def test_the_time_inside_model_and_tool_calls_is_reported_separately(completed):
    assert any("0.9s of that was spent" in c.value for c in completed.caption)


def test_the_displayed_token_counts_come_from_the_model_call(completed):
    assert any("400 in / 120 out" in element.value for element in completed.metric)


# --- temporary files ---------------------------------------------------------


def test_an_uploaded_pdf_is_deleted_once_the_run_is_over():
    with streamlit_app.prepared_sources(Upload("cv.pdf"), "", None, JD_TEXT) as (
        cv,
        jd,
    ):
        path = Path(cv)
        assert path.exists()  # the pipeline can still read it
        assert jd == JD_TEXT

    assert not path.exists()
    assert not path.parent.exists()


def test_two_uploaded_pdfs_do_not_overwrite_each_other():
    with streamlit_app.prepared_sources(
        Upload("cv.pdf"), "", Upload("jd.pdf"), ""
    ) as (cv, jd):
        assert cv != jd
        assert Path(cv).exists() and Path(jd).exists()


def test_a_temporary_pdf_is_removed_even_when_the_pipeline_fails(
    settings, monkeypatch
):
    seen: list[Path] = []

    def explode(cv, _jd, _settings):
        seen.append(Path(cv))
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(streamlit_app, "run_pipeline", explode)
    streamlit_app.st.session_state.clear()
    streamlit_app.start_run(Upload("cv.pdf"), "", None, JD_TEXT, settings)

    assert seen and seen[0].suffix == ".pdf"
    assert not seen[0].exists()


# --- configuration blocks the run --------------------------------------------


def unconfigured(tmp_path, **overrides) -> Settings:
    defaults = dict(
        heavy=ModelConfig(provider="anthropic", model="m", api_key="k"),
        light=ModelConfig(
            provider="openai_compatible", model="m", base_url="https://example.invalid"
        ),
        search_provider="serpapi",
        search_api_key=None,
        tracker_db_path=tmp_path / "a.db",
        tool_call_log_path=tmp_path / "t.log",
    )
    return Settings(**{**defaults, **overrides})


def test_run_is_enabled_when_both_tiers_are_configured(tmp_path, monkeypatch):
    app = run_app(unconfigured(tmp_path), monkeypatch=monkeypatch).run()

    assert app.button[0].disabled is False


def test_run_is_disabled_without_an_anthropic_key(tmp_path, monkeypatch):
    settings = unconfigured(
        tmp_path, heavy=ModelConfig(provider="anthropic", model="m", api_key=None)
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert app.button[0].disabled is True
    assert any("needs an API key" in e.value for e in app.error)


def test_run_is_disabled_without_a_base_url_for_the_light_tier(tmp_path, monkeypatch):
    settings = unconfigured(
        tmp_path,
        light=ModelConfig(provider="openai_compatible", model="m", base_url=None),
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert app.button[0].disabled is True
    assert any("needs a base URL" in e.value for e in app.error)


def test_a_keyless_local_endpoint_does_not_block_the_run(tmp_path, monkeypatch):
    """Ollama ignores the API key, and `OpenAICompatibleClient` supplies a
    placeholder for it - so a missing light key must not disable the button."""
    settings = unconfigured(
        tmp_path,
        light=ModelConfig(
            provider="openai_compatible",
            model="llama3",
            api_key=None,
            base_url="http://localhost:11434/v1",
        ),
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert app.button[0].disabled is False
    assert streamlit_app.missing_requirements(settings) == []


def test_a_missing_search_key_never_blocks_the_run(tmp_path, monkeypatch):
    app = run_app(unconfigured(tmp_path, search_api_key=None), monkeypatch=monkeypatch)

    assert app.run().button[0].disabled is False


def test_the_blocking_message_names_no_credential(settings, monkeypatch):
    blocked = Settings(
        heavy=ModelConfig(provider="anthropic", model="m", api_key=None),
        light=ModelConfig(provider="openai_compatible", model="m", base_url=None),
        search_provider="serpapi",
        search_api_key="SEARCH-SECRET",
        tracker_db_path=settings.tracker_db_path,
        tool_call_log_path=settings.tool_call_log_path,
    )

    app = run_app(blocked, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "SEARCH-SECRET" not in rendered
    assert len(app.error) == 2  # one per unconfigured tier


def test_an_unknown_provider_is_reported_rather_than_guessed(tmp_path):
    settings = unconfigured(
        tmp_path, heavy=ModelConfig(provider="mystery", model="m", api_key="k")
    )

    assert "mystery" in streamlit_app.missing_requirements(settings)[0]

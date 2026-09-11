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

from job_agent.agents.writing import assemble_full_cv
from job_agent.app import auth
from job_agent.app import ui as streamlit_app
from job_agent.config import ModelConfig, Settings
from job_agent.memory import BaselineCVStore, TailoredCVVersionStore
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
    User,
)
from job_agent.observability import TraceCollector, TraceEvent

TEST_USER = User(
    id=1, name="Test User", email="test@example.com", created_at=datetime(2026, 1, 1)
)

# AppTest resolves relative paths against this file, so be explicit.
APP = str(
    Path(__file__).resolve().parent.parent / "job_agent" / "app" / "streamlit_app.py"
)
CV_TEXT = "Built a Task Management REST API with FastAPI and SQLite."
JD_TEXT = "Junior AI Engineer at Arbisoft. Requirements: Strong Python."


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(
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
    # Most tests exercise the common case: a baseline CV is already on file
    # and the entry page only needs a job description. Tests for the
    # "no baseline yet" first-time flow build their own settings/DB instead.
    BaselineCVStore(s.tracker_db_path).save(TEST_USER.id, CV_TEXT)
    return s


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
    tailored_bullets = ["Built a Task Management REST API with FastAPI."]
    tailored_omitted = ["BS Computer Science"]
    context.tailored_cv = TailoredCV(
        role="Junior AI Engineer",
        company="Arbisoft",
        bullets=tailored_bullets,
        omitted=tailored_omitted,
        # The same deterministic assembly the real writing agent uses (see
        # agents/writing.py's assemble_full_cv) - a realistic complete CV,
        # not just the bullets, so this fixture matches production shape.
        full_text=assemble_full_cv(context.parsed_cv, context.job, tailored_bullets),
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


def run_app(settings, pipeline=None, monkeypatch=None, logged_in=True) -> AppTest:
    """Start the app with configuration and the pipeline both stubbed.

    `logged_in=True` (the default) pre-seeds a signed-in user in session
    state, so every test written before the login gate existed keeps
    exercising the pipeline UI unchanged, without each one needing to know
    about authentication. Tests that specifically cover the login/signup
    screen itself pass `logged_in=False`.
    """
    app = AppTest.from_file(APP, default_timeout=30)
    # Settings reach the app through the patched loader only - putting them in
    # session_state would dump them into AppTest's repr, which is where the
    # test would then "find" a leaked credential that the UI never rendered.
    monkeypatch.setattr(streamlit_app, "load_settings", lambda *a, **k: settings)
    if pipeline is not None:
        monkeypatch.setattr(streamlit_app, "run_pipeline", pipeline)
    if logged_in:
        app.session_state[auth.CURRENT_USER_KEY] = TEST_USER
    return app


# --- login / signup / logout -------------------------------------------------


def test_a_logged_out_visitor_sees_only_the_login_signup_screen(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()

    assert not app.exception
    assert "Sign in to continue." in rendered_text(app)
    # Nothing from the pipeline UI is reachable yet.
    assert not app.file_uploader
    assert "✦ Start Analysis" not in [b.label for b in app.button]


def test_signup_creates_an_account_and_signs_the_user_in(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()

    app.text_input(key="signup_name").set_value("Ada Lovelace")
    app.text_input(key="signup_email").set_value("ada@example.com")
    app.text_input(key="signup_password").set_value("correct-horse-1")
    app.text_input(key="signup_confirm").set_value("correct-horse-1")
    app.button(key="signup_submit").click().run()

    assert not app.exception
    rendered = rendered_text(app)
    assert "Signed in as" in rendered and "Ada Lovelace" in rendered
    assert "✦ Start Analysis" in [b.label for b in app.button]


def test_signup_with_mismatched_passwords_is_rejected(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()

    app.text_input(key="signup_name").set_value("Ada Lovelace")
    app.text_input(key="signup_email").set_value("ada@example.com")
    app.text_input(key="signup_password").set_value("correct-horse-1")
    app.text_input(key="signup_confirm").set_value("does-not-match")
    app.button(key="signup_submit").click().run()

    assert any("do not match" in e.value for e in app.error)
    assert "✦ Start Analysis" not in [b.label for b in app.button]


def test_signup_with_an_email_already_registered_is_rejected(settings, monkeypatch):
    from job_agent.memory import UserStore

    UserStore(settings.tracker_db_path).create(
        "Ada Lovelace", "ada@example.com", "correct-horse-1"
    )

    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()
    app.text_input(key="signup_name").set_value("Someone Else")
    app.text_input(key="signup_email").set_value("ada@example.com")
    app.text_input(key="signup_password").set_value("another-password")
    app.text_input(key="signup_confirm").set_value("another-password")
    app.button(key="signup_submit").click().run()

    assert any("already exists" in e.value for e in app.error)
    assert "✦ Start Analysis" not in [b.label for b in app.button]


def test_login_with_the_wrong_password_is_rejected(settings, monkeypatch):
    from job_agent.memory import UserStore

    UserStore(settings.tracker_db_path).create(
        "Ada Lovelace", "ada@example.com", "correct-horse-1"
    )

    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()
    app.text_input(key="login_email").set_value("ada@example.com")
    app.text_input(key="login_password").set_value("wrong-password")
    app.button(key="login_submit").click().run()

    assert any("Incorrect email or password" in e.value for e in app.error)
    assert "✦ Start Analysis" not in [b.label for b in app.button]


def test_login_with_correct_credentials_succeeds(settings, monkeypatch):
    from job_agent.memory import UserStore

    UserStore(settings.tracker_db_path).create(
        "Ada Lovelace", "ada@example.com", "correct-horse-1"
    )

    app = run_app(settings, monkeypatch=monkeypatch, logged_in=False).run()
    app.text_input(key="login_email").set_value("ada@example.com")
    app.text_input(key="login_password").set_value("correct-horse-1")
    app.button(key="login_submit").click().run()

    assert not app.exception
    rendered = rendered_text(app)
    assert "Signed in as" in rendered and "Ada Lovelace" in rendered
    assert "✦ Start Analysis" in [b.label for b in app.button]


def test_logging_out_returns_to_the_login_signup_screen(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch, logged_in=True).run()
    assert "✦ Start Analysis" in [b.label for b in app.button]  # precondition

    logout = [b for b in app.button if b.label == "Log out"][0]
    logout.click().run()

    assert not app.exception
    assert "Sign in to continue." in rendered_text(app)
    assert "✦ Start Analysis" not in [b.label for b in app.button]


# --- import and startup ------------------------------------------------------


def test_the_app_imports_and_renders(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert not app.exception
    assert "Job Application Agent" in app.title[0].value


def test_the_configuration_sidebar_is_hidden_from_the_end_user(settings, monkeypatch):
    """Product decision: the technical "Configuration" sidebar (provider/
    model names, which keys are set, the tracker db filename) is not shown
    to the end user - the sidebar now holds only the account panel (who is
    signed in, and the way out). This must not mean the settings themselves
    are gone: `render_sidebar` is untouched and still fully correct; nothing
    in `main()` calls it any more, exactly like `render_progress` (the
    hidden Pipeline section) before it."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    for label in (
        "Configuration",
        "small-model",
        "big-model",
        "Scoring/writing key",
        "Parsing key",
        "Search key",
    ):
        assert label not in rendered
    for secret in ("HEAVY-SECRET", "LIGHT-SECRET", "SEARCH-SECRET"):
        assert secret not in rendered
    # The account panel is still there - only Configuration is gone.
    assert f"Signed in as **{TEST_USER.name}**" in rendered


def test_a_missing_search_key_no_longer_shows_a_sidebar_notice(tmp_path, monkeypatch):
    """The explanatory notice ("no search key: company research is
    skipped") lived inside the now-hidden Configuration sidebar - it no
    longer appears anywhere, but the run itself still isn't blocked by a
    missing search key (see the entry-page tests for that guarantee,
    unaffected by this)."""
    app = run_app(unconfigured(tmp_path), monkeypatch=monkeypatch).run()

    assert not any("company research is skipped" in i.value for i in app.info)


# --- input handling ----------------------------------------------------------


def test_empty_input_is_refused_before_the_pipeline_runs(settings, monkeypatch):
    calls = []

    def pipeline(cv, jd, s, on_progress=None):  # pragma: no cover - must never be reached
        calls.append((cv, jd))
        raise AssertionError("the pipeline must not run without inputs")

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.button(key="start_analysis").click().run()

    assert calls == []
    assert any("provide both" in e.value for e in app.error)
    assert not app.exception


def test_a_baseline_cv_without_a_job_description_is_still_refused(settings, monkeypatch):
    """The `settings` fixture already has a baseline CV saved - a job
    description is still required before a run can start."""
    app = run_app(settings, pipeline=lambda *a: None, monkeypatch=monkeypatch).run()
    app.button(key="start_analysis").click().run()

    assert any("provide both" in e.value for e in app.error)


def test_the_pipeline_is_called_with_the_baseline_cv_and_what_the_user_typed(
    settings, monkeypatch
):
    seen = {}

    def pipeline(cv, jd, s, on_progress=None):
        seen["cv"], seen["jd"], seen["settings"] = cv, jd, s
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert seen["cv"] == CV_TEXT  # from the saved baseline, not typed
    assert seen["jd"] == JD_TEXT
    assert seen["settings"] is settings


def test_surrounding_whitespace_in_the_job_description_is_trimmed(settings, monkeypatch):
    seen = {}

    def pipeline(cv, jd, s, on_progress=None):
        seen["jd"] = jd
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(f"  {JD_TEXT}  ")
    app.button(key="start_analysis").click().run()

    assert seen["jd"] == JD_TEXT


# --- rendering a successful run ----------------------------------------------


@pytest.fixture
def completed(settings, monkeypatch) -> AppTest:
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    return app.button(key="start_analysis").click().run()


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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    rendered = rendered_text(app)
    assert "Partial matches" in rendered
    assert "Related evidence" in rendered
    assert "not the visualization libraries the role asks for" in rendered
    assert "does not itself demonstrate professional analytical experience" in rendered


def test_the_pipeline_section_is_hidden_from_the_results_page(completed):
    """Product decision: the technical pipeline breakdown (stage timings,
    call counts, tokens) is not shown to the end user. This must not mean
    the pipeline stopped recording it - see `test_nested_events_are_not_double_
    counted` and `test_run_totals_survives_a_trace_with_no_stage_events` below,
    which prove `run_totals`/the underlying trace are untouched; this test
    only proves the *page* no longer displays any of it."""
    rendered = rendered_text(completed)
    for label in (
        "Parse CV",
        "Research the company",
        "Score candidate-role fit",
        "Tailor CV + draft cover letter",
        "Save to the tracker",
        "Model + tool calls",
    ):
        assert label not in rendered
    # The fixture's own trace still has every stage - proof this is a display
    # choice, not something removed from the pipeline itself.
    stage_names = {name for name, _ in streamlit_app.STAGES}
    recorded = {e.name for e in finished_context().trace.events}
    assert stage_names <= recorded


def _pipeline_reporting_progress(stage_names):
    """A pipeline stub that actually calls `on_progress` one stage at a time,
    the same contract `run_pipeline`'s real LangGraph streaming honours -
    so a test can verify `render_processing`'s own row-update reaction to
    progress, independent of LangGraph's own streaming mechanics (already
    verified separately against the real, unmodified supervisor graph)."""

    def pipeline(cv, jd, s, on_progress=None):
        trace = TraceCollector()
        for name in stage_names:
            trace.record(TraceEvent(agent="test", name=name))
            if on_progress is not None:
                on_progress(trace)
        return finished_context()

    return pipeline


def test_the_processing_page_shows_every_stage_as_done_on_success(settings, monkeypatch):
    """The Loading/Processing page: friendlier labels than the results-page
    summary, but the same real stages - every row must show as complete once
    the pipeline actually reports each one finished."""
    stage_names = [name for name, _ in streamlit_app.STAGES]
    app = run_app(
        settings,
        pipeline=_pipeline_reporting_progress(stage_names),
        monkeypatch=monkeypatch,
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    rendered = rendered_text(app)
    for label in (
        "Preparing CV",
        "Reading job description",
        "Researching company",
        "Analyzing candidate-job fit",
        "Creating tailored CV and cover letter",
        "Saving results",
    ):
        assert f"✅ {label}" in rendered


def test_the_processing_page_omits_tailoring_when_it_is_skipped(settings, monkeypatch):
    """A stage this run will never execute must not appear at all - not as
    pending, not as done - since `settings.skip_tailoring` means the graph
    has no write_application node to ever complete."""
    from dataclasses import replace

    stage_names = [
        name for name, _ in streamlit_app.STAGES if name != "write_application"
    ]
    app = run_app(
        replace(settings, skip_tailoring=True),
        pipeline=_pipeline_reporting_progress(stage_names),
        monkeypatch=monkeypatch,
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    rendered = rendered_text(app)
    assert "Creating tailored CV and cover letter" not in rendered
    assert "✅ Preparing CV" in rendered
    assert "✅ Saving results" in rendered


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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert not app.exception
    assert any("could not be produced" in w.value for w in app.warning)
    assert any("your original CV stands" in w.value for w in app.warning)


def test_skipped_tailoring_is_labeled_as_skipped_not_as_a_grounding_failure(
    settings, monkeypatch
):
    from dataclasses import replace

    context = finished_context()
    context.tailored_cv = None
    context.cover_letter = None

    app = run_app(
        replace(settings, skip_tailoring=True),
        pipeline=lambda *a: context,
        monkeypatch=monkeypatch,
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert not app.exception
    assert any("Tailoring was skipped" in i.value for i in app.info)
    rendered = rendered_text(app)
    assert "grounding check could not verify" not in rendered
    assert "No cover letter was produced." not in rendered


def test_a_run_without_research_still_renders(settings, monkeypatch):
    context = finished_context()
    context.brief = None

    app = run_app(settings, pipeline=lambda *a: context, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert not app.exception
    assert any("No company research" in i.value for i in app.info)


# --- failures ----------------------------------------------------------------


def test_a_pipeline_failure_is_reported_without_crashing(settings, monkeypatch):
    def explode(*_args):
        raise RuntimeError("provider unavailable")

    app = run_app(settings, pipeline=explode, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

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


def test_results_survive_editing_the_job_description_box(completed):
    after = completed.text_area("jd_text").set_value(f"{JD_TEXT} And more.").run()

    assert "Dear Hiring Team," in rendered_text(after)


def test_start_a_new_analysis_clears_the_result_without_touching_the_inputs(completed):
    assert "Dear Hiring Team," in rendered_text(completed)  # precondition

    reset = [b for b in completed.button if "Start a new analysis" in b.label][0]
    after = reset.click().run()

    rendered = rendered_text(after)
    assert "Dear Hiring Team," not in rendered
    assert "75%" not in rendered
    # The entry form is untouched - still there, still holding what was typed.
    assert after.text_area("jd_text").value == JD_TEXT


def test_nothing_is_rendered_before_the_first_run(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "Fit report" not in rendered
    assert "Run complete" not in rendered
    assert not app.download_button  # nothing to download yet


def test_the_entry_page_explains_what_is_required(settings, monkeypatch):
    """The `settings` fixture already has a baseline CV saved, so the entry
    page only needs to ask for a job description - not a CV."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "Job description" in rendered and "required" in rendered
    assert "baseline CV" in rendered  # explains the baseline is reused
    assert "✦ Start Analysis" in [b.label for b in app.button]


def test_a_second_run_replaces_the_first_result(settings, monkeypatch):
    second = finished_context()
    second.fit_report.overall_fit = 0.42
    second.cover_letter.body = "Dear Second Team,"
    contexts = iter([finished_context(), second])

    app = run_app(
        settings, pipeline=lambda *a: next(contexts), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()
    assert "75%" in rendered_text(app)

    app.button(key="start_analysis").click().run()

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
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()
    assert "Dear Hiring Team," in rendered_text(app)

    state["fail"] = True
    app.button(key="start_analysis").click().run()

    rendered = rendered_text(app)
    assert "could not be completed" in rendered
    assert "Dear Hiring Team," not in rendered
    assert "75%" not in rendered


def test_an_incomplete_click_keeps_the_previous_result(settings, monkeypatch):
    """Clearing the job description and clicking Start Analysis does not
    start a run, so there is nothing to invalidate - destroying the last
    result would just punish a misclick."""
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    app.text_area("jd_text").set_value("")
    app.button(key="start_analysis").click().run()

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


def test_no_stage_timing_or_token_metrics_are_displayed(completed):
    """The three figures the hidden Pipeline section used to show (elapsed
    time, time-inside-calls, token counts) must not leak onto the page some
    other way. `run_totals` itself - the function that computes all three -
    is still exercised directly by `test_nested_events_are_not_double_counted`
    and `test_run_totals_survives_a_trace_with_no_stage_events` below."""
    metric_values = [element.value for element in completed.metric]
    assert "7.2s" not in metric_values
    assert "400 in / 120 out" not in metric_values
    assert not any("0.9s of that was spent" in c.value for c in completed.caption)


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

    def explode(cv, _jd, _settings, on_progress=None):
        seen.append(Path(cv))
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(streamlit_app, "run_pipeline", explode)
    streamlit_app.st.session_state.clear()
    streamlit_app.start_run(Upload("cv.pdf"), "", None, JD_TEXT, settings, TEST_USER)

    assert seen and seen[0].suffix == ".pdf"
    assert not seen[0].exists()


# --- configuration blocks the run --------------------------------------------


def unconfigured(tmp_path, *, with_baseline: bool = True, **overrides) -> Settings:
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
    s = Settings(**{**defaults, **overrides})
    if with_baseline:
        # Most callers are about model/provider configuration blocking the
        # run, not the baseline CV - seed one so it is never the reason the
        # button is disabled. Pass with_baseline=False to test the
        # no-baseline-yet state itself.
        BaselineCVStore(s.tracker_db_path).save(TEST_USER.id, CV_TEXT)
    return s


def test_run_is_enabled_when_both_tiers_are_configured(tmp_path, monkeypatch):
    app = run_app(unconfigured(tmp_path), monkeypatch=monkeypatch).run()

    assert app.button(key="start_analysis").disabled is False


def test_run_is_disabled_without_an_anthropic_key(tmp_path, monkeypatch):
    settings = unconfigured(
        tmp_path, heavy=ModelConfig(provider="anthropic", model="m", api_key=None)
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert app.button(key="start_analysis").disabled is True
    assert any("needs an API key" in e.value for e in app.error)


def test_run_is_disabled_without_a_base_url_for_the_light_tier(tmp_path, monkeypatch):
    settings = unconfigured(
        tmp_path,
        light=ModelConfig(provider="openai_compatible", model="m", base_url=None),
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert app.button(key="start_analysis").disabled is True
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

    assert app.button(key="start_analysis").disabled is False
    assert streamlit_app.missing_requirements(settings) == []


def test_a_missing_search_key_never_blocks_the_run(tmp_path, monkeypatch):
    app = run_app(unconfigured(tmp_path, search_api_key=None), monkeypatch=monkeypatch)

    assert app.run().button(key="start_analysis").disabled is False


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


# --- baseline CV: My Profile section -----------------------------------------
#
# tests/test_baseline_cv.py covers the store itself; these cover the UI built
# on top of it - what "My Profile" shows in each state, and that the entry
# page below it truly never re-collects a CV once a baseline exists.


def test_my_profile_shows_the_upload_form_when_no_baseline_exists(tmp_path, monkeypatch):
    settings = unconfigured(tmp_path, with_baseline=False)
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "Baseline CV on file" not in rendered
    assert "Upload your CV once" in rendered
    assert "baseline_text_initial" in [w.key for w in app.text_area]
    assert "A baseline CV is required before a run can start." in rendered
    assert app.button(key="start_analysis").disabled is True


def test_my_profile_shows_the_saved_state_once_a_baseline_exists(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "Baseline CV on file" in rendered
    assert "Replace baseline CV" in [e.label for e in app.expander]
    assert app.button(key="start_analysis").disabled is False


def test_saving_a_baseline_cv_for_the_first_time_stores_it(tmp_path, monkeypatch):
    settings = unconfigured(tmp_path, with_baseline=False)
    app = run_app(settings, monkeypatch=monkeypatch).run()

    app.text_area(key="baseline_text_initial").set_value(CV_TEXT)
    app.button(key="baseline_save_initial").click().run()

    stored = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert stored is not None
    assert stored.cv_text == CV_TEXT
    assert "Baseline CV on file" in rendered_text(app)


def test_replacing_the_baseline_cv_updates_the_stored_text(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    new_cv = "An updated CV with new experience."
    app.text_area(key="baseline_text_replace").set_value(new_cv)
    app.button(key="baseline_save_replace").click().run()

    stored = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert stored.cv_text == new_cv
    assert stored.cv_text != CV_TEXT


def test_replacing_the_baseline_cv_preserves_its_original_created_at(
    settings, monkeypatch
):
    original = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)

    app = run_app(settings, monkeypatch=monkeypatch).run()
    app.text_area(key="baseline_text_replace").set_value("Version two.")
    app.button(key="baseline_save_replace").click().run()

    updated = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at


# --- editing the baseline CV directly ("My Profile") ------------------------
#
# Editing reuses the exact same `BaselineCVStore.save` the initial-save and
# replace flows already exercise above - these tests cover the UI's own new
# wiring: the box is pre-filled, "Save Changes" persists it, another user's
# text never appears, the tailored CV archive is untouched, and blank text is
# refused exactly like every other path into `save` already refuses it.


def _edit_text_area(app):
    """The baseline editor's text_area - its key is tied to `updated_at`
    (see `profile._render_edit_form`), so it is found by prefix, not by an
    exact key."""
    for widget in app.text_area:
        if widget.key and widget.key.startswith("baseline_edit_text_"):
            return widget
    raise AssertionError("baseline edit text area not found")


def test_the_existing_baseline_cv_appears_pre_filled_in_the_editor(
    settings, monkeypatch
):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert "✏️ Edit CV" in [e.label for e in app.expander]
    assert _edit_text_area(app).value == CV_TEXT


def test_the_user_can_save_an_edited_baseline_cv(settings, monkeypatch):
    # A successful save reruns immediately (see `_render_edit_form`), the
    # same as the initial-save and replace flows - so, exactly like those,
    # what survives to be checked is the settled post-rerun state, not the
    # transient toast from the discarded pass before it.
    app = run_app(settings, monkeypatch=monkeypatch).run()

    edited = "Built a Task Management REST API with FastAPI - now with tests."
    _edit_text_area(app).set_value(edited)
    app.button(key="baseline_save_edit").click().run()

    assert not app.exception
    assert _edit_text_area(app).value == edited


def test_the_edited_baseline_is_persisted(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    edited = "Built a Task Management REST API with FastAPI - now with tests."
    _edit_text_area(app).set_value(edited)
    app.button(key="baseline_save_edit").click().run()

    stored = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert stored.cv_text == edited


def test_editing_preserves_created_at_and_moves_updated_at(settings, monkeypatch):
    original = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)

    app = run_app(settings, monkeypatch=monkeypatch).run()
    _edit_text_area(app).set_value("Version two.")
    app.button(key="baseline_save_edit").click().run()

    updated = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at


def test_another_users_baseline_cv_is_never_shown_in_the_editor(
    settings, monkeypatch
):
    other_user = User(
        id=2,
        name="Ada Lovelace",
        email="ada@example.com",
        created_at=datetime(2026, 1, 1),
    )
    BaselineCVStore(settings.tracker_db_path).save(
        other_user.id, "Ada's completely different CV."
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert _edit_text_area(app).value == CV_TEXT
    assert "Ada's completely different CV." not in rendered_text(app)


def test_editing_the_baseline_does_not_change_tailored_cv_history(
    settings, monkeypatch
):
    TailoredCVVersionStore(settings.tracker_db_path).save(
        user_id=TEST_USER.id,
        company="Sephora",
        role="Data Scientist",
        cv_text="An earlier tailored CV for Sephora.",
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()
    _edit_text_area(app).set_value("An edited baseline CV.")
    app.button(key="baseline_save_edit").click().run()

    versions = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )
    assert len(versions) == 1
    assert versions[0].cv_text == "An earlier tailored CV for Sephora."
    assert versions[0].company == "Sephora"


def test_replacing_the_baseline_still_works_alongside_the_editor(
    settings, monkeypatch
):
    """The new "Edit CV" expander must not interfere with the existing
    "Replace baseline CV" flow - both remain available side by side."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    new_cv = "A completely new document, uploaded fresh."
    app.text_area(key="baseline_text_replace").set_value(new_cv)
    app.button(key="baseline_save_replace").click().run()

    stored = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert stored.cv_text == new_cv
    # The editor now reflects the replacement too - no stale text left over.
    assert _edit_text_area(app).value == new_cv


def test_saving_a_blank_edit_is_refused_and_does_not_erase_the_baseline(
    settings, monkeypatch
):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    _edit_text_area(app).set_value("   ")
    app.button(key="baseline_save_edit").click().run()

    assert any("A CV is required." in e.value for e in app.error)
    stored = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert stored.cv_text == CV_TEXT  # unchanged


def test_the_entry_page_has_no_cv_upload_widgets_once_a_baseline_exists(
    settings, monkeypatch
):
    """The entry page's own CV widgets were removed entirely when the
    baseline CV system was added - any CV upload/paste now lives only in My
    Profile. `baseline_*` keys belong to that section, not the entry page."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    file_uploader_keys = [w.key for w in app.file_uploader]
    text_area_keys = [w.key for w in app.text_area]
    assert "cv_file" not in file_uploader_keys
    assert "cv_text" not in text_area_keys
    assert "jd_file" in file_uploader_keys
    assert "jd_text" in text_area_keys


def test_a_completed_run_does_not_change_the_stored_baseline_cv(settings, monkeypatch):
    """The UI-level counterpart to the store-level guarantee already covered
    in test_baseline_cv.py: producing a tailored CV for one job application
    must never change what is stored as the user's baseline."""
    before = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)

    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert "Dear Hiring Team," in rendered_text(app)  # the run actually completed
    after = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert after == before


# --- page flow: Entry -> Loading -> Results ----------------------------------
#
# The four screens (Login/Signup, Entry, Loading, Results) are one continuous
# Streamlit script, not separate routed pages - that is a deliberate, tested
# choice (see the "results survive a rerun" section above: editing the job
# description or downloading a document must not lose a result, which only
# works because the Entry form and the Results are rendered by the same
# script pass). These tests cover the product-level labeling and sequencing
# that make the single script *read* as four guided steps instead of one
# long form: a "Step N of 4" heading per screen, the Entry form stepping back
# into a collapsed section once a result exists, and exactly one pipeline
# call per click.


def test_the_entry_page_is_labeled_step_two_of_four(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert any("Step 2 of 4" in s.value for s in app.subheader)


def test_the_loading_page_is_labeled_step_three_of_four(settings, monkeypatch):
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert any("Step 3 of 4" in s.value for s in app.subheader)


def test_the_results_page_is_labeled_step_four_of_four(completed):
    assert any("Step 4 of 4" in s.value for s in completed.subheader)


def test_clicking_start_analysis_runs_the_pipeline_exactly_once(settings, monkeypatch):
    calls = []

    def pipeline(cv, jd, s, on_progress=None):
        calls.append((cv, jd))
        return finished_context()

    app = run_app(settings, pipeline=pipeline, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert len(calls) == 1


def test_the_entry_page_is_not_collapsed_before_any_result_exists(settings, monkeypatch):
    """Before a first run, the Entry page is the whole point of the screen -
    it must not start out tucked away in a "start over" section."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert "✏️ Start a new analysis" not in [e.label for e in app.expander]


def test_the_entry_page_steps_back_into_a_collapsed_section_once_results_exist(
    completed,
):
    """Once there is a result, the Entry form moves into a collapsed
    "start a new analysis" section so Results reads as its own screen - but
    the same widgets, still fully functional (see the "results survive a
    rerun" tests above), not a second, disconnected copy of the form.

    The collapsing decision is made once per script pass, at the top of
    `main`, before the click that just produced this result was processed -
    so it takes effect from the *next* rerun onward, exactly like every
    other "survives a rerun" case above. A no-op rerun (re-setting the same
    job description) is what any later interaction on this page - a
    download click, a keystroke - would trigger anyway.
    """
    app = completed.text_area("jd_text").set_value(JD_TEXT).run()

    assert "✏️ Start a new analysis" in [e.label for e in app.expander]
    assert "jd_text" in [w.key for w in app.text_area]
    assert app.button(key="start_analysis") is not None
    assert "Dear Hiring Team," in rendered_text(app)  # the result itself survived


def test_the_entry_page_stays_expanded_after_a_failed_run(settings, monkeypatch):
    """A failed run is still "being on the Entry page, fixing what you just
    typed" - not a result, so it must not collapse the form the user needs
    to see."""

    def explode(*_args):
        raise RuntimeError("provider unavailable")

    app = run_app(settings, pipeline=explode, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert "could not be completed" in rendered_text(app)
    assert "✏️ Start a new analysis" not in [e.label for e in app.expander]


def test_the_deploy_button_is_hidden(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    css = "\n".join(m.value for m in app.markdown)
    assert '[data-testid="stAppDeployButton"] { display: none; }' in css


# --- multiple tailored CV versions ("My Tailored CVs") -----------------------
#
# tests/test_tailored_cv_version.py covers the store itself; these cover the
# UI's own wiring - that a completed run actually saves one, that a company
# applied to twice or two different companies never collide, that a run
# which never produced a tailored CV (skipped or ungrounded) never saves one,
# and that the history section only ever shows the logged-in user's own.


def test_a_successful_run_saves_a_tailored_cv_version(settings, monkeypatch):
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    versions = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )
    assert len(versions) == 1
    assert versions[0].company == "Arbisoft"
    assert versions[0].role == "Junior AI Engineer"
    assert "Built a Task Management REST API with FastAPI." in versions[0].cv_text
    # Metadata: which baseline snapshot this version was tailored from.
    assert versions[0].baseline_cv_updated_at is not None


def test_the_complete_tailored_cv_is_what_gets_passed_to_the_store(
    settings, monkeypatch
):
    """Requirement: the store must receive the COMPLETE tailored CV
    (`TailoredCV.full_text`), not just the highlighted bullets - proven by
    checking for content that only the full assembly produces (the role/
    company header and the candidate's skills), not merely the bullet
    text every other test here already checks for. De-prioritised material
    (`omitted`) is deliberately NOT part of that document - see
    `assemble_full_cv` - so it must not appear here either."""
    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    stored_text = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )[0].cv_text

    assert "Tailored for: Junior AI Engineer at Arbisoft" in stored_text
    assert "Python" in stored_text  # from the parsed CV's own skills
    assert "BS Computer Science" not in stored_text  # de-prioritised, left out
    # Retrievable complete, not truncated - the store gives back exactly
    # what the writing agent assembled.
    assert stored_text == finished_context().tailored_cv.full_text


def test_a_second_application_to_the_same_company_adds_a_new_version(
    settings, monkeypatch
):
    """Requirement 8: applying to Sephora twice is two separate records, the
    same way the tracker already allows two applications to one company."""
    first = finished_context()
    first.tailored_cv = first.tailored_cv.model_copy(
        update={"company": "Sephora", "role": "Data Scientist"}
    )
    second = finished_context()
    second.tailored_cv = second.tailored_cv.model_copy(
        update={
            "company": "Sephora",
            "role": "Data Scientist",
            "bullets": ["A different draft."],
            "full_text": "A different draft.",
        }
    )
    contexts = iter([first, second])

    app = run_app(
        settings, pipeline=lambda *a: next(contexts), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()
    app.button(key="start_analysis").click().run()

    versions = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )
    assert len(versions) == 2
    assert all(v.company == "Sephora" for v in versions)
    texts = {v.cv_text for v in versions}
    assert any("Built a Task Management REST API" in t for t in texts)
    assert any("A different draft." in t for t in texts)


def test_sephora_and_nexus_are_kept_as_separate_versions(settings, monkeypatch):
    sephora = finished_context()
    sephora.tailored_cv = sephora.tailored_cv.model_copy(
        update={"company": "Sephora", "role": "Data Scientist"}
    )
    nexus = finished_context()
    nexus.tailored_cv = nexus.tailored_cv.model_copy(
        update={"company": "Nexus", "role": "Data Scientist"}
    )
    contexts = iter([sephora, nexus])

    app = run_app(
        settings, pipeline=lambda *a: next(contexts), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()
    app.button(key="start_analysis").click().run()

    versions = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )
    companies = {v.company for v in versions}
    assert companies == {"Sephora", "Nexus"}

    labels = [e.label for e in app.expander]
    assert "📄 Sephora" in labels
    assert "📄 Nexus" in labels


def test_a_run_with_no_tailored_cv_does_not_save_a_version(settings, monkeypatch):
    """Whether tailoring was skipped (SKIP_TAILORING) or failed grounding
    (`RetryExhaustedError`, caught by the supervisor's `_write` node),
    `context.tailored_cv` is None either way - and that one check is what
    already keeps a skipped or ungrounded draft out of the archive, with no
    separate validity check needed (see `save_tailored_cv_version`)."""
    context = finished_context()
    context.tailored_cv = None
    context.cover_letter = None

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert (
        TailoredCVVersionStore(settings.tracker_db_path).list_for_user(TEST_USER.id)
        == []
    )


def test_a_saved_tailored_cv_version_does_not_change_the_baseline_cv(
    settings, monkeypatch
):
    before = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)

    app = run_app(
        settings, pipeline=lambda *a: finished_context(), monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    # The version was actually saved...
    assert TailoredCVVersionStore(settings.tracker_db_path).list_for_user(
        TEST_USER.id
    )
    # ...and the baseline is untouched by it.
    after = BaselineCVStore(settings.tracker_db_path).get(TEST_USER.id)
    assert after == before


def test_my_tailored_cvs_shows_an_empty_state_before_any_are_generated(
    settings, monkeypatch
):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "My Tailored CVs" in rendered
    assert "No tailored CVs yet" in rendered


def test_the_sidebar_has_a_jump_link_to_my_tailored_cvs(settings, monkeypatch):
    """A plain in-page anchor link next to the account panel - not a second
    copy of the tailored-CV list, just a shortcut to the same section's own
    anchor (see `render_tailored_cv_history`'s `anchor="my-tailored-cvs"`)."""
    app = run_app(settings, monkeypatch=monkeypatch).run()

    rendered = rendered_text(app)
    assert "[📄 My Tailored CVs](#my-tailored-cvs)" in rendered


def test_my_tailored_cvs_lists_a_newly_generated_version_immediately(
    settings, monkeypatch
):
    context = finished_context()
    context.tailored_cv = context.tailored_cv.model_copy(
        update={"company": "Sephora", "role": "Data Scientist"}
    )

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    labels = [e.label for e in app.expander]
    assert "📄 Sephora" in labels


def test_the_tailored_cv_history_is_scoped_to_the_logged_in_user(
    settings, monkeypatch
):
    other_user = User(
        id=2,
        name="Ada Lovelace",
        email="ada@example.com",
        created_at=datetime(2026, 1, 1),
    )
    store = TailoredCVVersionStore(settings.tracker_db_path)
    store.save(
        user_id=TEST_USER.id,
        company="Sephora",
        role="Data Scientist",
        cv_text="Test User's tailored CV.",
    )
    store.save(
        user_id=other_user.id,
        company="Nexus",
        role="Data Scientist",
        cv_text="Ada's tailored CV.",
    )

    app = run_app(settings, monkeypatch=monkeypatch).run()

    labels = [e.label for e in app.expander]
    assert "📄 Sephora" in labels
    assert "📄 Nexus" not in labels


# --- visual summary: chart data helpers --------------------------------------
#
# These are plain Python functions over data the scoring agent already
# produced (`FitReport`/`RoleRequirement`) - no model call, no re-judging,
# and no new fit score (see `agents/scoring.py`'s `compute_fit_score`,
# untouched). Tested directly rather than through the rendered chart:
# Streamlit's AppTest does not expose `st.bar_chart`'s underlying data, so
# the values that feed it are verified here, and the UI-level tests below
# only check that the right sections render without crashing.


def _job_and_report_with_mixed_requirements() -> tuple[JobDescription, FitReport]:
    job = JobDescription(
        role="Data Scientist",
        company="Sephora",
        raw_text="Data Scientist at Sephora.",
        requirements=[
            RoleRequirement(text="Python", must_have=True),
            RoleRequirement(text="SQL", must_have=True),
            RoleRequirement(text="Tableau", must_have=False),
        ],
    )
    report = FitReport(
        role="Data Scientist",
        company="Sephora",
        overall_fit=0.6,
        requirement_matches=[
            RequirementMatch(
                requirement="Python", match_level=MatchLevel.match, met=True
            ),
            RequirementMatch(requirement="SQL", match_level=MatchLevel.missing),
            RequirementMatch(requirement="Tableau", match_level=MatchLevel.partial),
        ],
    )
    return job, report


def test_match_level_counts_tallies_each_level_from_the_scoring_output():
    matches = [
        RequirementMatch(requirement="A", match_level=MatchLevel.match, met=True),
        RequirementMatch(requirement="B", match_level=MatchLevel.match, met=True),
        RequirementMatch(requirement="C", match_level=MatchLevel.partial),
        RequirementMatch(requirement="D", match_level=MatchLevel.related),
        RequirementMatch(requirement="E", match_level=MatchLevel.missing),
    ]

    counts = streamlit_app.match_level_counts(matches)

    assert counts == {
        MatchLevel.match: 2,
        MatchLevel.partial: 1,
        MatchLevel.related: 1,
        MatchLevel.missing: 1,
    }


def test_match_level_counts_of_an_empty_report_is_all_zero():
    assert streamlit_app.match_level_counts([]) == {
        level: 0 for level in MatchLevel
    }


def test_must_have_breakdown_splits_counts_by_the_jds_own_flag():
    job, report = _job_and_report_with_mixed_requirements()

    breakdown = streamlit_app.must_have_breakdown(job, report)

    assert breakdown["Must-have"] == {
        MatchLevel.match: 1,
        MatchLevel.partial: 0,
        MatchLevel.related: 0,
        MatchLevel.missing: 1,
    }
    assert breakdown["Nice-to-have"] == {
        MatchLevel.match: 0,
        MatchLevel.partial: 1,
        MatchLevel.related: 0,
        MatchLevel.missing: 0,
    }


def test_must_have_breakdown_is_all_zero_without_a_job():
    """`job` should always accompany a `fit_report` in a real run (scoring
    only ever runs after parse_jd) - this covers the defensive case where a
    test or a partial context supplies one without the other."""
    _, report = _job_and_report_with_mixed_requirements()

    breakdown = streamlit_app.must_have_breakdown(None, report)

    assert breakdown["Must-have"] == {level: 0 for level in MatchLevel}
    assert breakdown["Nice-to-have"] == {level: 0 for level in MatchLevel}


def test_requirement_table_rows_pairs_each_requirement_with_type_and_status():
    job, report = _job_and_report_with_mixed_requirements()

    rows = streamlit_app.requirement_table_rows(report, job)

    assert rows == [
        {"Requirement": "Python", "Type": "Must-have", "Status": "MATCH"},
        {"Requirement": "SQL", "Type": "Must-have", "Status": "MISSING"},
        {"Requirement": "Tableau", "Type": "Nice-to-have", "Status": "PARTIAL"},
    ]


def test_requirement_table_rows_degrades_gracefully_without_a_job():
    _, report = _job_and_report_with_mixed_requirements()

    rows = streamlit_app.requirement_table_rows(report, None)

    assert rows == [
        {"Requirement": "Python", "Status": "MATCH"},
        {"Requirement": "SQL", "Status": "MISSING"},
        {"Requirement": "Tableau", "Status": "PARTIAL"},
    ]


def test_overall_match_chart_data_has_one_row_per_level_in_a_fixed_order():
    _, report = _job_and_report_with_mixed_requirements()

    rows = streamlit_app.overall_match_chart_data(report)

    assert rows == [
        {"Status": "MATCH", "Count": 1},
        {"Status": "PARTIAL", "Count": 1},
        {"Status": "RELATED", "Count": 0},
        {"Status": "MISSING", "Count": 1},
    ]


def test_must_have_chart_data_has_one_row_per_status_and_category():
    job, report = _job_and_report_with_mixed_requirements()

    rows = streamlit_app.must_have_chart_data(job, report)

    must_have_rows = [r for r in rows if r["Requirement type"] == "Must-have"]
    nice_to_have_rows = [r for r in rows if r["Requirement type"] == "Nice-to-have"]
    assert {r["Status"]: r["Count"] for r in must_have_rows} == {
        "MATCH": 1, "PARTIAL": 0, "RELATED": 0, "MISSING": 1,
    }
    assert {r["Status"]: r["Count"] for r in nice_to_have_rows} == {
        "MATCH": 0, "PARTIAL": 1, "RELATED": 0, "MISSING": 0,
    }


# --- visual summary: rendering ------------------------------------------------


def test_the_visual_summary_shows_the_backend_overall_fit_score(completed):
    """Requirement 1: the score shown is the exact value the fixture's
    `FitReport.overall_fit` already carries (0.75) - never recomputed here."""
    assert "📊 Visual Summary" in [s.value for s in completed.subheader]
    assert any(
        m.label == "Overall fit" and m.value == "75%" for m in completed.metric
    )


def test_graphs_do_not_appear_before_a_completed_result(settings, monkeypatch):
    app = run_app(settings, monkeypatch=monkeypatch).run()

    assert "📊 Visual Summary" not in rendered_text(app)


def test_graphs_do_not_appear_after_a_failed_run(settings, monkeypatch):
    def explode(*_args):
        raise RuntimeError("provider unavailable")

    app = run_app(settings, pipeline=explode, monkeypatch=monkeypatch).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert "📊 Visual Summary" not in rendered_text(app)


def test_a_missing_fit_report_does_not_crash_the_visual_summary(settings, monkeypatch):
    context = finished_context()
    context.fit_report = None

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert not app.exception
    assert "📊 Visual Summary" not in rendered_text(app)


def test_a_fit_report_with_no_requirement_matches_does_not_crash(
    settings, monkeypatch
):
    context = finished_context()
    context.fit_report = context.fit_report.model_copy(
        update={"requirement_matches": []}
    )

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    assert not app.exception
    assert "No requirements were judged for this run." in rendered_text(app)


def test_the_requirement_table_shows_every_requirement_with_its_status(
    settings, monkeypatch
):
    job, report = _job_and_report_with_mixed_requirements()
    context = finished_context()
    context.job = job
    context.fit_report = report

    app = run_app(
        settings, pipeline=lambda *a: context, monkeypatch=monkeypatch
    ).run()
    app.text_area("jd_text").set_value(JD_TEXT)
    app.button(key="start_analysis").click().run()

    table = app.dataframe[0].value.to_dict("records")
    assert table == [
        {"Requirement": "Python", "Type": "Must-have", "Status": "MATCH"},
        {"Requirement": "SQL", "Type": "Must-have", "Status": "MISSING"},
        {"Requirement": "Tableau", "Type": "Nice-to-have", "Status": "PARTIAL"},
    ]


def test_existing_results_sections_still_render_alongside_the_visual_summary(
    completed,
):
    rendered = rendered_text(completed)
    assert "📊 Visual Summary" in rendered
    assert "📊 Fit Report" in rendered
    assert "🔎 Company Research" in rendered
    assert "📝 Tailored CV" in rendered
    assert "🛡️ Validation" in rendered
    assert "📁 Application Tracker" in rendered


def test_the_pipeline_section_stays_hidden_alongside_the_visual_summary(completed):
    rendered = rendered_text(completed)
    for label in ("Parse CV", "Score candidate-role fit", "Model + tool calls"):
        assert label not in rendered


def test_the_deploy_button_stays_hidden_alongside_the_visual_summary(completed):
    css = "\n".join(m.value for m in completed.markdown)
    assert '[data-testid="stAppDeployButton"] { display: none; }' in css

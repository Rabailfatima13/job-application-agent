"""MCP server surface, exercised in-process (FR-8, NFR-9).

These call the server's functions directly, which is the fast way to cover
every branch. `test_mcp_protocol.py` covers the same surface through the real
protocol.

Every test points the server at a tmp_path database and a tmp_path trace log,
so none of them can touch the real tracker or the real data/tool_calls.log -
`server.py` builds its collector from the real `load_settings()`, so leaving
TOOL_CALL_LOG_PATH unset here would otherwise have every run of this file
write stub-driven noise into the project's real trace log.
"""

import json

import pytest
from pydantic import ValidationError

from job_agent.mcp_server import server
from job_agent.memory import SQLiteApplicationTracker
from job_agent.models import ApplicationStatus, CoverLetter, FitReport, TailoredCV
from job_agent.validation import RetryExhaustedError


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Point the server at a throwaway tracker database and trace log."""
    path = tmp_path / "applications.db"
    monkeypatch.setenv("TRACKER_DB_PATH", str(path))
    monkeypatch.setenv("TOOL_CALL_LOG_PATH", str(tmp_path / "tool_calls.log"))
    return path


@pytest.fixture
def tracked(db_path) -> dict:
    return server.track_application(
        role="Junior AI Engineer", company="Arbisoft", fit_score=0.846
    )


# --- Resource ----------------------------------------------------------------


def test_the_resource_is_empty_before_anything_is_tracked(db_path):
    assert json.loads(server.applications_all()) == []


def test_the_resource_returns_every_tracked_application(db_path, tracked):
    server.track_application(role="Data Engineer", company="Arbisoft", fit_score=0.5)

    rows = json.loads(server.applications_all())

    assert len(rows) == 2
    assert {r["role"] for r in rows} == {"Junior AI Engineer", "Data Engineer"}


def test_the_resource_is_json_safe(db_path, tracked):
    (row,) = json.loads(server.applications_all())

    # The enum is a plain string and the timestamp an ISO string, so any MCP
    # client can read it without knowing our Python types.
    assert row["status"] == "draft"
    assert isinstance(row["created_at"], str)
    assert row["created_at"].startswith("20")
    assert set(row) == {
        "application_id",
        "role",
        "company",
        "fit_score",
        "status",
        "created_at",
    }


# --- track_application -------------------------------------------------------


def test_tracking_persists_through_the_real_tracker(db_path, tracked):
    # Read back with the tracker directly: the server wrote to the same store
    # the pipeline uses, not a copy.
    record = SQLiteApplicationTracker(db_path).get(tracked["application_id"])

    assert record is not None
    assert record.role == "Junior AI Engineer"
    assert record.fit_score == 0.846


def test_a_new_application_starts_as_a_draft(db_path, tracked):
    assert tracked["status"] == "draft"


def test_each_tracked_application_gets_its_own_id(db_path):
    first = server.track_application(role="A", company="Arbisoft", fit_score=0.5)
    second = server.track_application(role="B", company="Arbisoft", fit_score=0.5)

    assert first["application_id"] != second["application_id"]


def test_an_out_of_range_score_is_rejected_by_the_same_model(db_path):
    with pytest.raises(ValidationError):
        server.track_application(role="A", company="Arbisoft", fit_score=1.4)


def test_an_unknown_starting_status_is_rejected_with_the_valid_options(db_path):
    with pytest.raises(ValueError, match="draft, ready, submitted, archived"):
        server.track_application(
            role="A", company="Arbisoft", fit_score=0.5, status="interviewing"
        )


# --- list_applications / get_application -------------------------------------


def test_listing_filters_by_status(db_path, tracked):
    server.set_application_status(tracked["application_id"], "ready")
    server.track_application(role="B", company="Arbisoft", fit_score=0.5)

    assert len(server.list_applications()) == 2
    assert len(server.list_applications(status="ready")) == 1
    assert server.list_applications(status="submitted") == []


def test_listing_rejects_an_unknown_status_filter(db_path):
    with pytest.raises(ValueError, match="Unknown status"):
        server.list_applications(status="nonsense")


def test_getting_an_application_returns_it(db_path, tracked):
    fetched = server.get_application(tracked["application_id"])

    assert fetched == tracked


def test_getting_an_unknown_application_fails_loudly(db_path):
    # A missing application must never look like an empty one.
    with pytest.raises(ValueError, match="No application tracked"):
        server.get_application("does-not-exist")


# --- set_application_status --------------------------------------------------


def test_status_can_be_moved_along_the_pipeline(db_path, tracked):
    updated = server.set_application_status(tracked["application_id"], "ready")

    assert updated["status"] == "ready"
    assert (
        SQLiteApplicationTracker(db_path).get(tracked["application_id"]).status
        is ApplicationStatus.ready
    )


def test_submitting_is_an_explicit_human_action_not_a_default(db_path, tracked):
    # Nothing marks an application submitted on its own; it takes this call.
    assert tracked["status"] == "draft"

    updated = server.set_application_status(tracked["application_id"], "submitted")

    assert updated["status"] == "submitted"


def test_status_changes_on_an_unknown_application_fail(db_path):
    with pytest.raises(ValueError, match="No application tracked"):
        server.set_application_status("does-not-exist", "ready")


def test_an_unknown_status_is_rejected(db_path, tracked):
    with pytest.raises(ValueError, match="Unknown status"):
        server.set_application_status(tracked["application_id"], "interviewing")


def test_only_the_status_changes(db_path, tracked):
    updated = server.set_application_status(tracked["application_id"], "archived")

    assert updated["role"] == tracked["role"]
    assert updated["fit_score"] == tracked["fit_score"]
    assert updated["created_at"] == tracked["created_at"]


# --- search_company ----------------------------------------------------------


def test_search_returns_structured_results(db_path, monkeypatch):
    # The HTTP call is injected, so this never leaves the machine. Patching
    # from_settings rather than the module-level default is what actually takes
    # effect - the default is bound when WebSearchClient is defined.
    fake_response = {
        "organic_results": [
            {
                "title": "Arbisoft",
                "link": "https://arbisoft.com",
                "snippet": "A software company.",
            }
        ]
    }
    monkeypatch.setattr(
        server.WebSearchClient,
        "from_settings",
        classmethod(
            lambda cls, settings, **kwargs: cls(
                provider="serpapi",
                api_key="test-key",
                request_fn=lambda url, params, headers: fake_response,
            )
        ),
    )

    results = server.search_company(query='"Arbisoft" company overview', max_results=3)

    assert results == [
        {
            "title": "Arbisoft",
            "url": "https://arbisoft.com",
            "snippet": "A software company.",
        }
    ]


def test_search_without_credentials_names_the_setting_not_a_value(
    db_path, monkeypatch
):
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    monkeypatch.setattr(
        "job_agent.config.load_dotenv", lambda *args, **kwargs: None
    )

    with pytest.raises(ValueError) as exc:
        server.search_company(query="anything")

    message = str(exc.value)
    assert "SEARCH_API_KEY" in message  # the name...
    assert "test-key" not in message  # ...never a value


# --- Model-backed tools ------------------------------------------------------
#
# Every model call is stubbed, so this file consumes no provider quota. What is
# under test is that the MCP layer *delegates* to the existing agents rather
# than reimplementing scoring or grounding.

CV_EXTRACTION = {
    "candidate_name": "Mahnoor Rauf",
    "evidence": [
        {
            "text": (
                "Built a Task Management REST API with FastAPI, SQLAlchemy and "
                "SQLite, with Pydantic validation and pytest coverage."
            ),
            "section": "Experience",
            "skills": ["FastAPI"],
        }
    ],
    "skills": ["Python", "FastAPI", "pytest"],
}

JD_EXTRACTION = {
    "role": "Junior AI Engineer",
    "company": "Arbisoft",
    "location": "Lahore",
    "requirements": [
        {"text": "Strong Python", "must_have": True},
        {"text": "Kubernetes experience", "must_have": False},
    ],
    "responsibilities": ["Build LLM-backed features"],
    "skills": ["Python"],
}

SCORING_JUDGEMENT = {
    "judgements": [
        {"requirement_index": 0, "match_level": "match", "evidence_index": 0},
        {"requirement_index": 1, "match_level": "missing", "evidence_index": None},
    ],
    "recommended_emphasis": [0],
}

WRITING_DRAFT = {
    "bullets": [
        "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite."
    ],
    "omitted": [],
    "cover_letter": (
        "Dear Hiring Team,\n\nI built a Task Management REST API with FastAPI "
        "and SQLite.\n\nSincerely,\nMahnoor Rauf"
    ),
}

FABRICATED_DRAFT = {
    **WRITING_DRAFT,
    "bullets": ["Managed Kubernetes clusters for 5 years at Netflix."],
}

CV_TEXT = (
    "Mahnoor Rauf\n"
    "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite, "
    "with Pydantic validation and pytest coverage.\n"
)
JD_TEXT = "Junior AI Engineer at Arbisoft\nRequirements: Strong Python\n"


@pytest.fixture
def stub_models(monkeypatch):
    """Point the server's router at a stub client that replays canned replies."""

    def _install(replies: list[str]):
        from tests.conftest import StubModelClient

        client = StubModelClient(replies, name="stub")
        monkeypatch.setattr(
            server, "ModelRouter", lambda settings, **kwargs: _StubRouter(client)
        )
        return client

    return _install


class _StubRouter:
    """Minimal stand-in for ModelRouter: every step gets the same stub client."""

    def __init__(self, client):
        self.client_instance = client

    def tier_for(self, step: str) -> str:
        return "light"

    def client(self, tier: str):
        return self.client_instance

    def for_step(self, step: str):
        return self.client_instance


def test_score_fit_returns_a_valid_fit_report(db_path, stub_models):
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
        ]
    )

    report = server.score_fit(cv=CV_TEXT, job_description=JD_TEXT)

    assert FitReport.model_validate(report)
    assert report["role"] == "Junior AI Engineer"
    assert report["company"] == "Arbisoft"
    # One must-have met (1.0) of 1.5 total weight -> the project's own rubric.
    assert report["overall_fit"] == pytest.approx(0.667, abs=1e-3)
    assert report["gaps"] == ["Kubernetes experience"]


def test_score_fit_evidence_comes_from_the_cv(db_path, stub_models):
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
        ]
    )

    report = server.score_fit(cv=CV_TEXT, job_description=JD_TEXT)

    met = [m for m in report["requirement_matches"] if m["met"]]
    assert met and all(m["evidence"] in CV_TEXT for m in met)


def test_score_fit_delegates_to_the_existing_scoring_agent(
    db_path, stub_models, monkeypatch
):
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
        ]
    )
    calls = []
    original = server.ScoringAgent.score

    def spy(self, cv, job):
        calls.append((type(cv).__name__, type(job).__name__))
        return original(self, cv, job)

    monkeypatch.setattr(server.ScoringAgent, "score", spy)

    server.score_fit(cv=CV_TEXT, job_description=JD_TEXT)

    # The MCP layer is an adapter: the agent did the work, on parsed objects.
    assert calls == [("ParsedCV", "JobDescription")]


def test_tailor_application_returns_all_three_documents(db_path, stub_models):
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
            json.dumps(WRITING_DRAFT),
        ]
    )

    result = server.tailor_application(cv=CV_TEXT, job_description=JD_TEXT)

    assert set(result) == {"fit_report", "tailored_cv", "cover_letter"}
    assert FitReport.model_validate(result["fit_report"])
    assert TailoredCV.model_validate(result["tailored_cv"])
    assert CoverLetter.model_validate(result["cover_letter"])
    assert result["tailored_cv"]["company"] == "Arbisoft"
    assert "Mahnoor Rauf" in result["cover_letter"]["body"]


def test_tailor_application_delegates_to_the_existing_writing_agent(
    db_path, stub_models, monkeypatch
):
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
            json.dumps(WRITING_DRAFT),
        ]
    )
    seen = {}
    original = server.WritingAgent.write

    def spy(self, cv, job, report, brief=None):
        seen["brief"] = brief
        seen["types"] = (type(cv).__name__, type(job).__name__, type(report).__name__)
        return original(self, cv, job, report, brief)

    monkeypatch.setattr(server.WritingAgent, "write", spy)

    server.tailor_application(cv=CV_TEXT, job_description=JD_TEXT)

    assert seen["types"] == ("ParsedCV", "JobDescription", "FitReport")
    # No company context over MCP: the tailored CV stays CV-only grounded and
    # an MCP caller cannot inject unverified "company facts".
    assert seen["brief"] is None


def test_tailor_application_rejects_fabricated_output(db_path, stub_models):
    # The writing agent's grounding rules apply unchanged through MCP.
    stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
            *[json.dumps(FABRICATED_DRAFT)] * 3,
        ]
    )

    with pytest.raises(ValueError) as exc:
        server.tailor_application(cv=CV_TEXT, job_description=JD_TEXT)

    message = str(exc.value)
    assert "without unsupported claims" in message
    assert "Kubernetes" in message  # the violation is explained


def test_tailor_application_retries_a_rejected_draft_then_succeeds(
    db_path, stub_models
):
    stub = stub_models(
        [
            json.dumps(CV_EXTRACTION),
            json.dumps(JD_EXTRACTION),
            json.dumps(SCORING_JUDGEMENT),
            json.dumps(FABRICATED_DRAFT),
            json.dumps(WRITING_DRAFT),
        ]
    )

    result = server.tailor_application(cv=CV_TEXT, job_description=JD_TEXT)

    # The fabricated bullet never reaches the documents. ("Kubernetes" still
    # appears in the fit report's gaps, where it belongs - it is a real
    # requirement the candidate does not meet.)
    assert "Kubernetes" not in json.dumps(result["tailored_cv"])
    assert "Kubernetes" not in json.dumps(result["cover_letter"])
    assert result["fit_report"]["gaps"] == ["Kubernetes experience"]
    assert len(stub.calls) == 5  # parse, parse, score, rejected draft, rewrite


def test_a_model_failure_surfaces_as_a_clean_error(db_path, monkeypatch):
    class ExplodingClient:
        name = "stub"

        def complete(self, *args, **kwargs):
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        server, "ModelRouter", lambda settings, **kw: _StubRouter(ExplodingClient())
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        server.score_fit(cv=CV_TEXT, job_description=JD_TEXT)


def test_malformed_model_output_exhausts_the_existing_retry_guard(
    db_path, stub_models
):
    stub_models(["not json", "not json", "not json"])

    with pytest.raises(RetryExhaustedError):
        server.score_fit(cv=CV_TEXT, job_description=JD_TEXT)


def test_an_empty_input_is_rejected_before_any_model_call(db_path, stub_models):
    stub = stub_models([json.dumps(CV_EXTRACTION)])

    with pytest.raises(ValueError, match="empty"):
        server.score_fit(cv="   ", job_description=JD_TEXT)

    assert stub.calls == []


# --- Model tier safety -------------------------------------------------------


def test_model_backed_tools_default_to_the_cheap_tier(db_path, monkeypatch):
    monkeypatch.delenv("MCP_MODEL_TIER", raising=False)
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("HEAVY_MODEL", "expensive-model")
    monkeypatch.setenv("LIGHT_MODEL", "cheap-model")

    router = server._router()

    # Every step - including ones the pipeline sends to the heavy tier - now
    # resolves to the cheap configuration, so MCP cannot bill the big provider.
    assert router._configs["heavy"].model == "cheap-model"
    assert router._configs["light"].model == "cheap-model"


def test_the_heavy_tier_is_opt_in_only(db_path, monkeypatch):
    monkeypatch.setattr("job_agent.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("HEAVY_MODEL", "expensive-model")
    monkeypatch.setenv("LIGHT_MODEL", "cheap-model")
    monkeypatch.setenv("MCP_MODEL_TIER", "heavy")

    router = server._router()

    assert router._configs["heavy"].model == "expensive-model"


# --- Configuration -----------------------------------------------------------


def test_the_server_reads_its_database_path_from_configuration(db_path, tracked):
    # No path is baked in; a client can point the server anywhere.
    assert db_path.exists()
    assert server._tracker().db_path == db_path

"""Custom MCP server exposing the application tracker (FR-8, NFR-9).

The point is portability: another MCP client - Claude Code, an IDE, a
colleague's script - can read the application pipeline and move an application
along it without importing this package or knowing that the rows live in
SQLite. That is exactly what NFR-9 asks for.

Everything here is a thin adapter over logic that already exists and is already
tested:

  resource  applications://all      -> SQLiteApplicationTracker.list()
  tool      list_applications       -> SQLiteApplicationTracker.list()
  tool      get_application         -> SQLiteApplicationTracker.get()
  tool      track_application       -> ApplicationRecord + tracker.add()
  tool      set_application_status  -> SQLiteApplicationTracker.set_status()
  tool      search_company          -> WebSearchClient.search()
  tool      score_fit               -> parse_cv/parse_jd + ScoringAgent
  tool      tailor_application      -> the above + WritingAgent (grounded)

No business rule is reimplemented, so the MCP surface cannot drift from the
pipeline's behaviour: records are validated by the same Pydantic model, the
same status enum applies, the same database file is read, and the scoring
rubric and no-fabrication guard are the agents' own.

**Model-backed tools cost quota.** `score_fit` and `tailor_application` make
real model calls - three and four respectively, plus a retry per malformed or
ungrounded response. They run on the **light** tier by default, whatever that
is configured to be, so an MCP call never quietly bills the expensive provider;
`MCP_MODEL_TIER=heavy` opts in. Nothing else changes that.

Nothing here can mark an application `submitted` on the agent's behalf -
status changes are a human action (FR-14), so they arrive through an explicit
tool call from a human-driven client.

Run it directly over stdio:

    python -m job_agent.mcp_server.server

Note for clients: a tool that returns a list sends one content block per item,
so read them as `[json.loads(b.text) for b in result.content]` rather than
expecting a single JSON array. The `applications://all` resource does return
one JSON array.

The tracker file and search credentials come from the same configuration the
pipeline uses (`.env`); no secret is ever returned by a tool or a resource.
"""

import json
import os
from dataclasses import replace
from datetime import datetime

from mcp.server import MCPServer

from ..agents.scoring import ScoringAgent
from ..agents.supervisor import new_application_id
from ..agents.writing import WritingAgent
from ..config import Settings, load_settings
from ..llm import ModelRouter
from ..memory.tracker import SQLiteApplicationTracker
from ..models import ApplicationRecord, ApplicationStatus, JobDescription, ParsedCV
from ..observability import TraceCollector
from ..tools.parsing import parse_cv, parse_jd
from ..tools.web_search import SearchError, WebSearchClient
from ..validation import RetryExhaustedError

#: Which configured model tier the model-backed tools run on. Defaults to the
#: cheap tier so an MCP call can never quietly bill the expensive provider -
#: see `_router` for why that is the safe default rather than the obvious one.
MCP_MODEL_TIER_VAR = "MCP_MODEL_TIER"

mcp = MCPServer(
    name="job-application-agent",
    instructions=(
        "Exposes the job application tracker: read the pipeline of tracked "
        "applications, record a new one, and move an application's status "
        "along. Also exposes company web search. Application material is never "
        "submitted anywhere automatically."
    ),
)


def _settings() -> Settings:
    """Read configuration per call, so a client that restarts the server with a
    different TRACKER_DB_PATH gets what it asked for (this is also what lets
    the tests point the subprocess at a temporary database)."""
    return load_settings()


def _tracker() -> SQLiteApplicationTracker:
    return SQLiteApplicationTracker(_settings().tracker_db_path)


def _router() -> ModelRouter:
    """The router the model-backed tools use.

    By default every step - including the ones the pipeline routes to the heavy
    tier - runs on the **light** tier. An MCP call is made by a client the
    operator may not be watching, so the expensive provider is opt-in rather
    than opt-out: `MCP_MODEL_TIER=heavy` selects the configured heavy provider,
    and nothing else does.

    This substitution happens on a copy of the settings, so the pipeline's own
    routing is untouched.
    """
    settings = _settings()
    if os.getenv(MCP_MODEL_TIER_VAR, "light").strip().lower() != "heavy":
        settings = replace(settings, heavy=settings.light)
    return ModelRouter(settings)


def _collector() -> TraceCollector:
    """MCP calls are traced into the same log the pipeline writes, so their
    cost and latency show up in the Week 8 numbers rather than going unseen."""
    return TraceCollector(log_path=_settings().tool_call_log_path)


def _parse_inputs(
    cv: str, job_description: str, router: ModelRouter, collector: TraceCollector
) -> tuple[ParsedCV, JobDescription]:
    """Turn the two text inputs into the structured objects the agents expect.

    Uses the pipeline's own parsers - including the verbatim-evidence guard -
    so what the agents see over MCP is exactly what they would see in a run.
    Both arguments accept raw text or a path to a .txt/.md/.pdf file.
    """
    return (
        parse_cv(cv, router, collector),
        parse_jd(job_description, router, collector),
    )


def _as_dict(record: ApplicationRecord) -> dict:
    """JSON-safe form: the enum becomes its value, the timestamp an ISO string."""
    return record.model_dump(mode="json")


def _parse_status(status: str) -> ApplicationStatus:
    try:
        return ApplicationStatus(status)
    except ValueError:
        valid = ", ".join(s.value for s in ApplicationStatus)
        raise ValueError(
            f"Unknown status '{status}' (expected one of: {valid})"
        ) from None


@mcp.resource("applications://all", mime_type="application/json")
def applications_all() -> str:
    """Every tracked application, newest first, as a JSON array.

    Each entry has application_id, role, company, fit_score, status and
    created_at. Read-only - use the tools to change anything.
    """
    return json.dumps([_as_dict(r) for r in _tracker().list()])


@mcp.tool()
def list_applications(status: str | None = None) -> list[dict]:
    """List tracked applications, newest first.

    Args:
        status: optional filter - one of draft, ready, submitted, archived.
    """
    filter_by = _parse_status(status) if status else None
    return [_as_dict(r) for r in _tracker().list(status=filter_by)]


@mcp.tool()
def get_application(application_id: str) -> dict:
    """Fetch one tracked application by its id.

    Raises if the id is unknown, so a caller never mistakes a missing
    application for an empty one.
    """
    record = _tracker().get(application_id)
    if record is None:
        raise ValueError(f"No application tracked with id '{application_id}'")
    return _as_dict(record)


@mcp.tool()
def track_application(
    role: str,
    company: str,
    fit_score: float,
    status: str = "draft",
) -> dict:
    """Record a new application in the pipeline.

    Args:
        role: the advertised role title.
        company: the employer.
        fit_score: candidate-role fit, between 0.0 and 1.0.
        status: starting status; defaults to draft, which is where a new
            application belongs - nothing is ever submitted automatically.

    The record is validated by the same Pydantic model the pipeline uses, so an
    out-of-range score or an unknown status is rejected here too.
    """
    record = ApplicationRecord(
        application_id=new_application_id(),
        role=role,
        company=company,
        fit_score=fit_score,
        status=_parse_status(status),
        created_at=datetime.now(),
    )
    return _as_dict(_tracker().add(record))


@mcp.tool()
def set_application_status(application_id: str, status: str) -> dict:
    """Move a tracked application to a new status.

    Args:
        application_id: the application to update.
        status: one of draft, ready, submitted, archived.

    Marking something submitted records a decision the human has already made;
    this server never submits an application anywhere.
    """
    tracker = _tracker()
    if tracker.get(application_id) is None:
        raise ValueError(f"No application tracked with id '{application_id}'")
    return _as_dict(tracker.set_status(application_id, _parse_status(status)))


@mcp.tool()
def search_company(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for company information.

    Args:
        query: what to search for, e.g. '"Acme" company overview'.
        max_results: how many results to return.

    Returns a list of {title, url, snippet}. Requires a search provider to be
    configured; the error names the missing setting, never its value.
    """
    settings = _settings()
    try:
        client = WebSearchClient.from_settings(settings)
        results = client.search(query, max_results=max_results)
    except SearchError as exc:
        raise ValueError(str(exc)) from None
    return [r.model_dump() for r in results]


@mcp.tool()
def score_fit(cv: str, job_description: str) -> dict:
    """Score how well a CV fits a job description, with evidence and gaps.

    Args:
        cv: the CV text, or a path to a .txt/.md/.pdf file.
        job_description: the posting text, or a path to a file.

    Returns the fit report: role, company, overall_fit (0-1),
    requirement_matches (each with the CV evidence that supports it), gaps and
    recommended_emphasis.

    Runs the project's existing scoring agent, so the score is computed by the
    same rubric the pipeline uses (must-have 1.0, nice-to-have 0.5) and every
    piece of evidence is a real line from the CV.

    Cost: three model calls (parse CV, parse JD, score) against the configured
    light-tier provider, plus a retry per call if the model returns malformed
    JSON. Set MCP_MODEL_TIER=heavy to use the heavy provider instead.
    """
    router, collector = _router(), _collector()
    parsed_cv, job = _parse_inputs(cv, job_description, router, collector)
    report = ScoringAgent(router, collector=collector).score(parsed_cv, job)
    return report.model_dump(mode="json")


@mcp.tool()
def tailor_application(cv: str, job_description: str) -> dict:
    """Tailor the CV for a role and draft a grounded cover letter.

    Args:
        cv: the CV text, or a path to a .txt/.md/.pdf file.
        job_description: the posting text, or a path to a file.

    Returns {"fit_report": ..., "tailored_cv": ..., "cover_letter": ...}.

    Runs the project's existing writing agent, which means the no-fabrication
    guarantee applies unchanged: the tailored CV is checked against the
    candidate's CV alone, generated lines that introduce a technology, employer,
    qualification, metric or duration the CV never stated are rejected, and the
    model is asked to rewrite. If it cannot produce grounded material within its
    attempt budget, this tool fails rather than returning something unverified.

    One tool rather than two, because tailoring the CV and drafting the letter
    are a single model call in this system - splitting them would either double
    the cost or require the server to hold state between calls.

    Cost: four model calls (parse CV, parse JD, score, write) plus a retry per
    rejected draft, against the configured light-tier provider.
    """
    router, collector = _router(), _collector()
    parsed_cv, job = _parse_inputs(cv, job_description, router, collector)
    report = ScoringAgent(router, collector=collector).score(parsed_cv, job)

    try:
        # brief=None: no company research is performed here, so the letter is
        # grounded in the CV alone. Company context comes from the pipeline's
        # research stage, never from an MCP caller - a caller-supplied "fact"
        # would be unverified material this server cannot check.
        tailored, letter = WritingAgent(router, collector=collector).write(
            parsed_cv, job, report, brief=None
        )
    except RetryExhaustedError as exc:
        raise ValueError(
            "Could not produce tailored documents without unsupported claims: "
            f"{exc}"
        ) from None

    return {
        "fit_report": report.model_dump(mode="json"),
        "tailored_cv": tailored.model_dump(mode="json"),
        "cover_letter": letter.model_dump(mode="json"),
    }


if __name__ == "__main__":
    mcp.run()

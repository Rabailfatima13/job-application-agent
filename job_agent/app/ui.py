"""Streamlit view functions (FR-15, NFR-6).

A view, not a participant. It collects a CV and a job description, hands them
to the existing supervisor, and renders the `RunContext` that comes back. No
parsing, scoring, writing or grounding happens here - every number and every
line of text on screen was produced by the pipeline.

The aim is to make the agentic pipeline legible to a human: which stages ran,
how long each took, what each one produced, and - most importantly - what the
grounding guards rejected along the way.

Run it with:

    streamlit run job_agent/app/streamlit_app.py
"""

import tempfile
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import streamlit as st

from job_agent.config import ModelConfig, Settings, load_settings
from job_agent.memory.session import RunContext
from job_agent.memory.tailored_cv_version import TailoredCVVersionStore
from job_agent.models import (
    BaselineCV,
    FitReport,
    JobDescription,
    MatchLevel,
    RequirementMatch,
    RoleRequirement,
    TailoredCV,
    User,
)

from . import auth, profile

# The stages a full run goes through, in order, with labels a human can read.
STAGES: list[tuple[str, str]] = [
    ("parse_cv", "Parse CV"),
    ("parse_jd", "Parse job description"),
    ("research", "Research the company"),
    ("score_fit", "Score candidate-role fit"),
    ("write_application", "Tailor CV + draft cover letter"),
    ("track_application", "Save to the tracker"),
]

# Trace events emitted by the deterministic guards, and how to describe them.
GUARDS: dict[str, str] = {
    "research:relevance_guard": "Search results discarded for not mentioning the company",
    "research:grounding_guard": "Company facts discarded as unsupported by the sources",
    "scoring:evidence_guard": "Fit-report claims rejected for lacking CV evidence",
    "writing:grounding_guard": "Draft rejected: claims not supported by the CV",
}

#: Where the most recent run lives between reruns. Streamlit re-executes this
#: whole script on every interaction, so anything held in a local variable is
#: gone by the next keystroke - see `start_run`.
RESULT_KEY = "run_result"
ERROR_KEY = "run_error"


# --- safety ------------------------------------------------------------------


def redact(text: str, settings: Settings) -> str:
    """Remove any configured credential from text before it reaches the screen.

    Nothing in the pipeline is supposed to put a key in an error message, and
    the search client is explicitly hardened against it - this is the last line
    of defence, because an unfamiliar provider error is exactly the kind of
    thing that would surprise us in front of an audience.
    """
    secrets = [
        settings.heavy.api_key,
        settings.light.api_key,
        settings.search_api_key,
    ]
    for secret in secrets:
        if secret and len(secret) > 8:
            text = text.replace(secret, "***")
    return text


def safe_error(exc: Exception, settings: Settings) -> str:
    """A one-line, credential-free description of a failure.

    Deliberately not a traceback: the user gets what went wrong, the terminal
    keeps the detail.
    """
    return redact(f"{type(exc).__name__}: {exc}", settings)


# --- configuration -----------------------------------------------------------


def tier_problem(config: ModelConfig) -> str | None:
    """Why this model tier could not be built, or None if it can be.

    Deliberately the same rule the adapters in `llm.providers` enforce: an
    Anthropic client needs a key, an OpenAI-compatible one needs a base URL and
    tolerates a missing key because local runtimes such as Ollama ignore it.
    Mirroring the adapters is what stops the Run button being disabled for a
    perfectly runnable keyless local setup.
    """
    if config.provider == "anthropic":
        return None if config.api_key else "an API key"
    if config.provider == "openai_compatible":
        return None if config.base_url else "a base URL"
    return f"a known provider (got '{config.provider}')"


def missing_requirements(settings: Settings) -> list[str]:
    """Everything that would make a run fail before it produced anything.

    Both tiers are checked because both are always used: parsing runs on the
    light tier and scoring on the heavy one, and scoring is the step the
    supervisor refuses to continue without. Web search is absent on purpose -
    a run without it still completes, with company research skipped.
    """
    problems = []
    for label, config in (
        ("The parsing / research model", settings.light),
        ("The scoring / writing model", settings.heavy),
    ):
        problem = tier_problem(config)
        if problem is not None:
            problems.append(f"{label} needs {problem} before a run can start.")
    return problems


# --- the pipeline call -------------------------------------------------------


def run_pipeline(
    cv_source: str,
    jd_source: str,
    settings: Settings,
    on_progress=None,
) -> RunContext:
    """Run one application through the existing supervisor.

    Watched with the graph's own `.stream()` instead of `Supervisor.run()`'s
    `.invoke()` - same supervisor, same LangGraph graph, same nodes, just
    observed one stage at a time instead of waiting for the final result -
    so a caller can show real per-stage progress instead of an opaque wait.
    `on_progress(trace)`, when given, is called once per stage as it
    actually completes; nothing here decides what a stage means or when it
    is "done" - that is still entirely the pipeline's own trace, read the
    same way `render_progress` already reads a finished one.

    Imported lazily so the module can be loaded (and tested) without building
    agents or touching a provider. This is the one seam every UI test
    replaces instead of hitting a real provider - keep its signature stable.
    """
    from job_agent.agents import build_supervisor

    supervisor = build_supervisor(settings)
    state = RunContext(
        cv_text=str(cv_source), jd_text=str(jd_source), trace=supervisor.collector
    )
    final_state = None
    for final_state in supervisor.graph.stream(state, stream_mode="values"):
        if on_progress is not None:
            on_progress(final_state["trace"])
    return RunContext(**final_state)


def read_upload(uploaded, pasted: str, directory: Path) -> str:
    """Whatever the user actually supplied: an uploaded file wins over the box.

    Returns text for .txt/.md. A PDF is written into `directory` and the path
    handed back, so the pipeline's own loader extracts it - the UI does not
    parse documents.
    """
    if uploaded is None:
        return pasted.strip()
    if uploaded.name.lower().endswith(".pdf"):
        # A unique name, because the CV and the job description may both be
        # PDFs and they share one directory.
        path = directory / f"{uuid4().hex}.pdf"
        path.write_bytes(uploaded.getvalue())
        return str(path)
    return uploaded.getvalue().decode("utf-8", errors="replace")


@contextmanager
def prepared_sources(cv_upload, cv_pasted: str, jd_upload, jd_pasted: str):
    """Yield `(cv_source, jd_source)` for one run, then clean up after it.

    An uploaded PDF has to reach the pipeline as a file on disk, and that file
    is the UI's to remove: `load_document_text` only reads what it is handed,
    and deleting a caller's path would be the wrong component making that
    decision. Scoping it to a temporary directory makes the cleanup
    unconditional - it happens even if the pipeline raises.
    """
    with tempfile.TemporaryDirectory(prefix="job-agent-upload-") as directory:
        yield (
            read_upload(cv_upload, cv_pasted, Path(directory)),
            read_upload(jd_upload, jd_pasted, Path(directory)),
        )


# --- cost and latency --------------------------------------------------------


def run_totals(context: RunContext) -> dict[str, float | int]:
    """Cost and latency for one run, without double-counting nested spans.

    `TraceCollector.totals()` sums every event it holds. That is right for
    tokens and wrong for time: the supervisor wraps each pipeline stage in a
    trace event of its own, and the model and tool calls made inside a stage
    are traced again in their own right - so a nested call's duration is
    counted twice, once in the child and once in the parent containing it.

    The stage spans are the run's top-level spans and they do not overlap, so
    summing those alone gives the elapsed time; summing everything else gives
    the time spent inside model and tool calls. Tokens need no such correction
    and are taken from the collector unchanged: only the innermost model call
    ever writes them, and a stage wrapper carries zero.
    """
    stage_names = dict(STAGES)
    stages = [e for e in context.trace.events if e.name in stage_names]
    nested = [e for e in context.trace.events if e.name not in stage_names]
    totals = context.trace.totals()
    return {
        "elapsed_ms": round(sum(e.duration_ms for e in stages), 1),
        "call_ms": round(sum(e.duration_ms for e in nested), 1),
        "calls": len(nested),
        "input_tokens": totals["input_tokens"],
        "output_tokens": totals["output_tokens"],
    }


# --- rendering ---------------------------------------------------------------


def render_sidebar(settings: Settings) -> None:
    """Configuration, without ever showing a credential."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption("Values are never displayed - only whether they are set.")

        st.markdown(
            f"**Parsing / research:** {settings.light.provider} · {settings.light.model}"
        )
        st.markdown(
            f"**Scoring / writing:** {settings.heavy.provider} · {settings.heavy.model}"
        )
        st.markdown(f"**Web search:** {settings.search_provider}")

        st.divider()

        for label, configured in (
            ("Scoring/writing key", bool(settings.heavy.api_key)),
            ("Parsing key", bool(settings.light.api_key)),
            ("Search key", bool(settings.search_api_key)),
        ):
            st.write(("✅ " if configured else "⚠️ ") + label)

        if not settings.search_api_key:
            st.info("No search key: the run works, company research is skipped.")

        st.divider()
        st.caption(f"Tracker: {settings.tracker_db_path.name}")


def render_progress(context: RunContext) -> None:
    """Which stages ran, and how long each took.

    Reconstructed from the run's own trace rather than guessed, so what is
    shown is what actually happened.
    """
    st.subheader("🧭 Pipeline")
    with st.container(border=True):
        durations = {
            event.name: event.duration_ms
            for event in context.trace.events
            if event.name in dict(STAGES)
        }
        for name, label in STAGES:
            if name in durations:
                st.markdown(f"✅ **{label}**  ·  {durations[name] / 1000:.1f}s")
            else:
                st.markdown(f"➖ {label}  ·  _skipped_")

        st.divider()
        totals = run_totals(context)
        columns = st.columns(3)
        columns[0].metric("Model + tool calls", totals["calls"])
        columns[1].metric("Total time", f"{totals['elapsed_ms'] / 1000:.1f}s")
        columns[2].metric(
            "Tokens", f"{totals['input_tokens']} in / {totals['output_tokens']} out"
        )
        st.caption(
            f"{totals['call_ms'] / 1000:.1f}s of that was spent inside model and tool "
            "calls; the remainder is orchestration."
        )


# One marker and one heading per match level. The fit report groups
# requirement_matches by this - the same structured data the scorer already
# produced - into sections, rather than one flat list plus a separate
# bare-text gaps list repeating a subset of it.
MATCH_LEVEL_DISPLAY: dict[MatchLevel, tuple[str, str]] = {
    MatchLevel.match: ("✅", "Strengths / Matches"),
    MatchLevel.partial: ("🟡", "Partial matches"),
    MatchLevel.related: ("🔶", "Related evidence"),
    MatchLevel.missing: ("❌", "Missing from CV"),
}

# Deliberately not the model's own `reason` text for a missing requirement:
# that is free text, and this project promises a specific, careful framing -
# absence of evidence in the CV is not evidence of absence, so the report
# never says "you don't have this skill". Fixed wording here guarantees that
# promise regardless of how any one run's reason happened to be phrased.
MISSING_FROM_CV_CAPTION = "Not mentioned in the CV - no reliable evidence found."


# The status element that makes each match level visually obvious at a
# glance - purely presentational, applied to the same match_level the scorer
# already produced. Never used to decide anything, only to color a badge.
_MATCH_LEVEL_STATUS = {
    MatchLevel.match: st.success,
    MatchLevel.partial: st.warning,
    MatchLevel.related: st.info,
    MatchLevel.missing: st.error,
}

_MATCH_LEVEL_LABEL: dict[MatchLevel, str] = {
    MatchLevel.match: "MATCH",
    MatchLevel.partial: "PARTIAL",
    MatchLevel.related: "RELATED",
    MatchLevel.missing: "MISSING",
}


def _render_requirement_card(match: RequirementMatch, level: MatchLevel) -> None:
    """One requirement as its own visually distinct card: a colored status
    badge, the requirement text, then the CV evidence and the model's
    reasoning kept visibly apart - the same information as before, just
    easier to scan than a run-on paragraph."""
    with st.container(border=True):
        _MATCH_LEVEL_STATUS[level](_MATCH_LEVEL_LABEL[level])
        st.markdown(f"**{match.requirement}**")
        if level == MatchLevel.missing:
            st.caption(MISSING_FROM_CV_CAPTION)
        else:
            if match.evidence:
                st.caption(f"📄 Evidence from your CV: {match.evidence}")
            if match.reason:
                st.caption(f"💭 {match.reason}")


# --- visual summary -----------------------------------------------------
#
# Everything below prepares chart data with plain Python from the fit
# report the scoring agent already produced (`agents/scoring.py`) - no
# model call, no re-judging, and no new score. `compute_fit_score` and
# `MATCH_LEVEL_WEIGHT` there are the only place a fit number is ever
# computed; this only counts and regroups the `RequirementMatch`es and
# `RoleRequirement`s that already exist on the RunContext.


def match_level_counts(matches: list[RequirementMatch]) -> dict[MatchLevel, int]:
    """How many requirements landed at each match level - a plain tally
    over the scorer's own output, in the fixed order every match level
    already renders in elsewhere on this page (see MATCH_LEVEL_DISPLAY)."""
    counts = {level: 0 for level in MatchLevel}
    for match in matches:
        counts[match.match_level] += 1
    return counts


def requirement_pairs(
    job: JobDescription | None, report: FitReport
) -> list[tuple[RoleRequirement, RequirementMatch]]:
    """Each requirement paired with its own judgement, so a must-have /
    nice-to-have split can be computed.

    Safe to zip by position: `ScoringAgent._resolve_matches` always builds
    exactly one `RequirementMatch` per `RoleRequirement`, in the job's own
    order - a requirement the model skipped still appears, as missing. If
    `job` is unavailable (should not happen once scoring has run, but the
    UI must not assume it), this returns [] rather than guessing which
    requirement is which.
    """
    if job is None:
        return []
    return list(zip(job.requirements, report.requirement_matches, strict=False))


def must_have_breakdown(
    job: JobDescription | None, report: FitReport
) -> dict[str, dict[MatchLevel, int]]:
    """Match-level counts split by the JD's own must-have flag - the same
    tally as `match_level_counts`, just grouped instead of pooled."""
    pairs = requirement_pairs(job, report)
    return {
        "Must-have": match_level_counts([m for r, m in pairs if r.must_have]),
        "Nice-to-have": match_level_counts([m for r, m in pairs if not r.must_have]),
    }


_LEVEL_ORDER = (
    MatchLevel.match,
    MatchLevel.partial,
    MatchLevel.related,
    MatchLevel.missing,
)


def overall_match_chart_data(report: FitReport) -> list[dict]:
    """One row per match level - the whole-report breakdown (requirement 2)."""
    counts = match_level_counts(report.requirement_matches)
    return [
        {"Status": _MATCH_LEVEL_LABEL[level], "Count": counts[level]}
        for level in _LEVEL_ORDER
    ]


def must_have_chart_data(job: JobDescription | None, report: FitReport) -> list[dict]:
    """Long-format rows (one per status x category) for a grouped chart -
    must-have performance next to nice-to-have performance (requirement 3)."""
    breakdown = must_have_breakdown(job, report)
    return [
        {
            "Status": _MATCH_LEVEL_LABEL[level],
            "Count": breakdown[category][level],
            "Requirement type": category,
        }
        for category in ("Must-have", "Nice-to-have")
        for level in _LEVEL_ORDER
    ]


def requirement_table_rows(
    report: FitReport, job: JobDescription | None
) -> list[dict]:
    """One row per requirement: what it was, and how it scored - a compact
    companion to the detailed cards `render_fit_report` already shows
    below, not a replacement for their evidence and reasoning."""
    pairs = requirement_pairs(job, report)
    if pairs:
        return [
            {
                "Requirement": match.requirement,
                "Type": "Must-have" if requirement.must_have else "Nice-to-have",
                "Status": _MATCH_LEVEL_LABEL[match.match_level],
            }
            for requirement, match in pairs
        ]
    # No job to pair against - still show a status per requirement rather
    # than nothing at all.
    return [
        {
            "Requirement": match.requirement,
            "Status": _MATCH_LEVEL_LABEL[match.match_level],
        }
        for match in report.requirement_matches
    ]


def render_visual_summary(context: RunContext) -> None:
    """A chart-based summary of the same fit report `render_fit_report`
    shows in detail below - the overall score, a match-level breakdown, a
    must-have vs. nice-to-have comparison, and a compact per-requirement
    table. Every value comes from `context.fit_report`/`context.job`
    through the deterministic helpers above; nothing here is shown when
    there is no result to summarise (see `render_last_run`/`render_results`,
    which only ever call this once a run has actually completed)."""
    report = context.fit_report
    if report is None:
        return

    st.subheader("📊 Visual Summary")

    with st.container(horizontal=True):
        st.metric("Overall fit", f"{report.overall_fit:.0%}", border=True)
        counts = match_level_counts(report.requirement_matches)
        st.metric("Matched", counts[MatchLevel.match], border=True)
        st.metric("Missing", counts[MatchLevel.missing], border=True)

    if not report.requirement_matches:
        st.caption("No requirements were judged for this run.")
        return

    with st.container(horizontal=True):
        with st.container(border=True):
            st.markdown("**Requirement match breakdown**")
            st.bar_chart(overall_match_chart_data(report), x="Status", y="Count")
        with st.container(border=True):
            st.markdown("**Must-have vs. nice-to-have**")
            st.bar_chart(
                must_have_chart_data(context.job, report),
                x="Status",
                y="Count",
                color="Requirement type",
            )

    with st.container(border=True):
        st.markdown("**Every requirement at a glance**")
        st.dataframe(requirement_table_rows(report, context.job), hide_index=True)


def render_fit_report(context: RunContext) -> None:
    report = context.fit_report
    if report is None:
        return

    st.subheader("📊 Fit Report")
    with st.container(border=True):
        left, right = st.columns([1, 2])
        left.metric("Overall fit", f"{report.overall_fit:.0%}")
        with right:
            st.markdown(f"**{report.role}** at **{report.company}**")
            st.caption(
                f"{len(report.requirement_matches)} requirement(s) judged "
                "against your CV evidence."
            )

    by_level: dict[MatchLevel, list[RequirementMatch]] = {
        level: [] for level in MatchLevel
    }
    for match in report.requirement_matches:
        by_level[match.match_level].append(match)

    for level in (
        MatchLevel.match,
        MatchLevel.partial,
        MatchLevel.related,
        MatchLevel.missing,
    ):
        matches = by_level[level]
        if not matches:
            continue
        marker, heading = MATCH_LEVEL_DISPLAY[level]
        st.markdown(f"#### {marker} {heading}")
        for match in matches:
            _render_requirement_card(match, level)


def render_research(context: RunContext) -> None:
    brief = context.brief
    st.subheader("🔎 Company Research")
    if brief is None or not brief.sources:
        st.info(
            brief.summary
            if brief is not None
            else "No company research was performed for this run."
        )
        return

    with st.container(border=True):
        st.markdown(brief.summary)
        if brief.facts:
            st.markdown("**Key findings**")
            for fact in brief.facts:
                st.markdown(f"- {fact}")
        with st.expander(f"Sources ({len(brief.sources)})"):
            for source in brief.sources:
                st.markdown(f"- [{source.title}]({source.url})")


def tailored_cv_text(tailored: TailoredCV) -> str:
    """The one source of "the tailored CV" as text - shown on screen,
    offered as a download, and (see `save_tailored_cv_version`) what a saved
    tailored CV version stores, so all three always agree.

    `full_text` (assembled by the writing agent itself, see
    `agents/writing.py`'s `assemble_full_cv`) is the complete tailored CV,
    not just the highlighted bullets - what gets archived is an actual CV a
    user could submit, not a highlight reel.
    """
    return tailored.full_text


def render_documents(context: RunContext, settings: Settings) -> None:
    st.subheader("📝 Tailored CV")
    if context.tailored_cv is None:
        if settings.skip_tailoring:
            st.info(
                "Tailoring was skipped for this run (SKIP_TAILORING is set) - "
                "no tailored CV was attempted."
            )
        else:
            st.warning(
                "No tailored CV was produced. The grounding check could not verify "
                "a draft, so your original CV stands."
            )
    else:
        with st.container(border=True):
            st.markdown(tailored_cv_text(context.tailored_cv))
            if context.tailored_cv.omitted:
                st.caption(
                    "De-prioritised: " + "; ".join(context.tailored_cv.omitted)
                )
        st.download_button(
            "Download tailored CV",
            data=tailored_cv_text(context.tailored_cv),
            file_name="tailored_cv.txt",
            mime="text/plain",
        )

    st.subheader("✉️ Cover Letter")
    if context.cover_letter is None:
        if settings.skip_tailoring:
            st.info(
                "Tailoring was skipped for this run (SKIP_TAILORING is set) - "
                "no cover letter was attempted."
            )
        else:
            st.warning("No cover letter was produced.")
    else:
        with st.container(border=True):
            st.markdown(context.cover_letter.body)
        st.download_button(
            "Download cover letter",
            data=context.cover_letter.body,
            file_name="cover_letter.txt",
            mime="text/plain",
        )


def render_validation(context: RunContext) -> None:
    """What the guards did - the part that makes the system trustworthy."""
    st.subheader("🛡️ Validation")
    st.caption(
        "Generated claims are checked against your CV. Anything the CV does "
        "not support is rejected and the model is asked to rewrite."
    )

    guard_events = [e for e in context.trace.events if e.name in GUARDS]
    retries = [e for e in context.trace.events if e.name.startswith("validate:")]

    if not guard_events:
        st.success("Every generated claim was supported by your CV.")
    else:
        for event in guard_events:
            with st.container(border=True):
                st.warning(f"**{GUARDS[event.name]}**")
                if event.result:
                    st.caption(event.result)

    if retries:
        st.caption(
            f"{len(retries)} response(s) were rejected for malformed output and "
            "regenerated."
        )


def render_application(context: RunContext, settings: Settings) -> None:
    st.subheader("📁 Application Tracker")
    if context.application is not None:
        record = context.application
        with st.container(border=True):
            left, mid, right = st.columns([2, 1, 1])
            left.markdown(f"**{record.role}** at **{record.company}**")
            mid.metric("Fit", f"{record.fit_score:.0%}")
            right.markdown(f"Status: `{record.status.value}`")
            st.caption(f"Application id: {record.application_id}")
            st.caption("Nothing is submitted anywhere - this is a saved draft.")

    try:
        from job_agent.memory import SQLiteApplicationTracker

        rows = SQLiteApplicationTracker(settings.tracker_db_path).list()
    except Exception as exc:  # noqa: BLE001 - the tracker view is never fatal
        st.caption(f"Could not read the tracker: {safe_error(exc, settings)}")
        return

    if rows:
        st.markdown(f"**All tracked applications ({len(rows)})**")
        st.table(
            [
                {
                    "Role": r.role,
                    "Company": r.company,
                    "Fit": f"{r.fit_score:.0%}",
                    "Status": r.status.value,
                    "Created": r.created_at.strftime("%Y-%m-%d %H:%M"),
                }
                for r in rows
            ]
        )


def render_results(context: RunContext, settings: Settings) -> None:
    for warning in context.warnings:
        st.warning(warning)

    # The technical "Pipeline" section (stage timings, model/tool call counts,
    # token totals) is deliberately not shown to the end user - product
    # decision, not a pipeline change. `render_progress` and `run_totals`
    # are untouched and still fully correct; nothing here calls them. The
    # pipeline itself keeps recording every stage in `context.trace` exactly
    # as before - only this page's display of it is gone.
    render_visual_summary(context)
    st.divider()
    render_fit_report(context)
    st.divider()
    render_research(context)
    st.divider()
    render_documents(context, settings)
    st.divider()
    render_validation(context)
    st.divider()
    render_application(context, settings)


# --- processing / loading page ------------------------------------------------

# Friendlier phrasing for the loading page specifically - keyed to the same
# stage identifiers STAGES already uses, so this is purely a display-label
# choice, never a second source of truth about what the stages are.
_PROCESSING_LABELS: dict[str, str] = {
    "parse_cv": "Preparing CV",
    "parse_jd": "Reading job description",
    "research": "Researching company",
    "score_fit": "Analyzing candidate-job fit",
    "write_application": "Creating tailored CV and cover letter",
    "track_application": "Saving results",
}


def render_processing(cv_source: str, jd_source: str, settings: Settings) -> RunContext:
    """The Loading / Processing page.

    Every row starts pending and only turns into a checkmark once that stage
    has actually finished, reconstructed live from the run's own trace via
    `run_pipeline`'s `on_progress` hook - never marked done ahead of the
    pipeline itself, and never shown at all for a stage this run will never
    execute (the tailoring row is omitted entirely when
    `settings.skip_tailoring` is set, since the graph then has no
    write_application node to complete).
    """
    st.subheader("Step 3 of 4 — ⏳ Analyzing your application")
    st.caption(
        "This takes about a minute. Each stage below updates as it actually "
        "finishes - nothing is marked done ahead of time."
    )

    stages = [
        (name, _PROCESSING_LABELS[name])
        for name, _ in STAGES
        if name != "write_application" or not settings.skip_tailoring
    ]
    with st.container(border=True):
        rows = {name: st.empty() for name, _ in stages}
        for name, label in stages:
            rows[name].markdown(f"⏳ {label}")

        def on_progress(trace) -> None:
            done = {event.name for event in trace.events}
            for name, label in stages:
                rows[name].markdown(f"✅ {label}" if name in done else f"⏳ {label}")

        return run_pipeline(cv_source, jd_source, settings, on_progress)


# --- running -----------------------------------------------------------------


def save_tailored_cv_version(
    context: RunContext, settings: Settings, user: User, baseline: BaselineCV | None
) -> None:
    """Archive this run's tailored CV, if it actually produced one.

    `context.tailored_cv` is only ever non-None when the writing agent's own
    grounding loop returned a clean draft (`WritingAgent.write` raises
    `RetryExhaustedError` instead of returning anything else, and the
    supervisor leaves it as None on that failure or when SKIP_TAILORING
    omits the write_application node entirely - see `supervisor._write`) -
    so this one check is what already keeps an invalid or skipped tailoring
    out of the archive, with no separate validity check needed here.

    `cv_text` is `tailored.full_text` - the complete tailored CV, not just
    its highlighted bullets (see `tailored_cv_text`). `baseline` (the same
    object "My Profile" already fetched for this run) is recorded only as
    its `updated_at` - which baseline *snapshot* this version was tailored
    from - never its text, so the baseline CV itself is still never touched
    or copied here, only referenced.

    A new row every time, never a replace: the same `TailoredCVVersionStore`
    guarantee that lets a second Sephora application and a first Nexus
    application both keep their own version. The baseline CV is untouched -
    nothing here ever calls `BaselineCVStore.save`.
    """
    if context.tailored_cv is None:
        return
    TailoredCVVersionStore(settings.tracker_db_path).save(
        user_id=user.id,
        company=context.tailored_cv.company,
        role=context.tailored_cv.role,
        cv_text=tailored_cv_text(context.tailored_cv),
        baseline_cv_updated_at=baseline.updated_at if baseline is not None else None,
    )


def start_run(
    cv_upload,
    cv_pasted: str,
    jd_upload,
    jd_pasted: str,
    settings: Settings,
    user: User,
    baseline: BaselineCV | None = None,
) -> None:
    """Run the pipeline once and record the outcome in session state.

    Session state rather than a local variable is what makes a result survive.
    Streamlit re-executes the entire script on every interaction - a download
    click, a keystroke in a text area - and on that rerun the Run button reads
    False, so a result held in a local would vanish along with it, taking a
    minute of real model spend with it.
    """
    with prepared_sources(cv_upload, cv_pasted, jd_upload, jd_pasted) as (cv, jd):
        if not cv or not jd:
            # The run never started, so whatever is already on screen still
            # stands: a stray click should not destroy the previous result.
            st.error("Please provide both a CV and a job description.")
            return

        # A new run supersedes the old one, and it clears first so that a
        # failure can never leave the previous run's report sitting on screen
        # underneath the error message.
        st.session_state.pop(RESULT_KEY, None)
        st.session_state.pop(ERROR_KEY, None)

        try:
            context = render_processing(cv, jd, settings)
        except Exception as exc:  # noqa: BLE001 - the UI must not crash
            # The redacted message, not the exception: nothing carrying a
            # credential is worth keeping across reruns.
            st.session_state[ERROR_KEY] = safe_error(exc, settings)
            return

    save_tailored_cv_version(context, settings, user, baseline)
    st.session_state[RESULT_KEY] = context


def render_last_run(settings: Settings) -> None:
    """Whatever the most recent run produced - on every rerun, not just the
    one that started it. Before the first run there is nothing to show.

    This is the Results page. The entry form above stays visible and
    editable throughout (see `start_run`'s own reasoning for why a stray
    interaction must never destroy a result) - "Start a new analysis" is an
    additional, explicit way to clear it without touching the inputs.
    """
    error = st.session_state.get(ERROR_KEY)
    if error is not None:
        st.error("The run could not be completed. Nothing was saved.\n\n" + error)
        return

    context = st.session_state.get(RESULT_KEY)
    if context is None:
        return

    st.success("Run complete.")
    st.subheader("Step 4 of 4 — 📊 Results")
    if st.button("🔄 Start a new analysis", key="start_new_analysis"):
        st.session_state.pop(RESULT_KEY, None)
        st.session_state.pop(ERROR_KEY, None)
        st.rerun()
    render_results(context, settings)


# --- page --------------------------------------------------------------------


def _inject_background_style() -> None:
    """A soft pastel wash behind the app, light-mode only.

    Purely decorative - no element structure, text, or behavior changes, so
    it has no effect on anything a test inspects. Streamlit itself has no
    API for a textured/gradient background, only flat theme colors, so this
    is the one deliberate, narrow use of injected CSS in the app - scoped to
    `prefers-color-scheme: light` so a viewer in dark mode keeps Streamlit's
    normal dark background instead of a jarring bright wash.
    """
    st.markdown(
        """
        <style>
        @media (prefers-color-scheme: light) {
        [data-testid="stAppViewContainer"] {
        background:
        radial-gradient(circle at 12% 18%, rgba(255,179,198,.55), transparent 42%),
        radial-gradient(circle at 82% 12%, rgba(255,200,210,.45), transparent 40%),
        radial-gradient(circle at 78% 58%, rgba(168,214,255,.45), transparent 45%),
        radial-gradient(circle at 8% 68%, rgba(168,214,255,.4), transparent 45%),
        radial-gradient(circle at 55% 80%, rgba(255,240,175,.55), transparent 50%),
        radial-gradient(circle at 42% 42%, rgba(214,190,255,.35), transparent 45%),
        #fdfbf5;
        background-attachment: fixed;
        }
        [data-testid="stHeader"] { background: transparent; }
        }
        /* Hide only the Deploy button - the main menu ("stMainMenu") and the
           rest of the toolbar are untouched, in both light and dark mode. */
        [data-testid="stAppDeployButton"] { display: none; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_tailored_cv_history(settings: Settings, user: User) -> None:
    """"My Tailored CVs": every version this user's baseline has been
    tailored into, one per completed application, newest first.

    Deliberately separate from "My Profile" above: that section is the one
    original baseline CV (see `profile.render_baseline_cv_section`); this is
    the read-only archive of what has been generated *from* it - a Sephora
    application and a Nexus application each keep their own row (see
    `TailoredCVVersionStore`), and neither this list nor a run that produces
    it ever writes back to the baseline.
    """
    st.subheader("📄 My Tailored CVs", anchor="my-tailored-cvs")
    versions = TailoredCVVersionStore(settings.tracker_db_path).list_for_user(user.id)
    if not versions:
        st.caption("No tailored CVs yet - complete an analysis to generate one.")
        return

    for version in versions:
        with st.expander(f"📄 {version.company}"):
            st.markdown(f"**{version.role}**")
            st.caption(f"Created: {version.created_at.strftime('%Y-%m-%d %H:%M')}")
            if version.baseline_cv_updated_at is not None:
                st.caption(
                    "Tailored from your baseline CV as of "
                    f"{version.baseline_cv_updated_at.strftime('%Y-%m-%d %H:%M')}."
                )
            st.markdown(version.cv_text)
            st.download_button(
                "Download this tailored CV",
                data=version.cv_text,
                file_name=f"tailored_cv_{version.company}_{version.id}.txt",
                mime="text/plain",
                key=f"download_tailored_cv_{version.id}",
            )


def render_tailored_cv_sidebar_link() -> None:
    """A quick jump to "My Tailored CVs" from the sidebar, next to the
    account panel - a plain in-page anchor link to the same section
    `render_tailored_cv_history` renders further down (see its
    `anchor="my-tailored-cvs"`), not a second copy of that content and not
    a rerun: clicking it just scrolls the existing page."""
    with st.sidebar:
        st.markdown("[📄 My Tailored CVs](#my-tailored-cvs)")


def main() -> None:
    st.set_page_config(page_title="Job Application Agent", page_icon="📄")
    _inject_background_style()

    settings = load_settings()
    user = auth.render_gate(settings)
    if user is None:
        # Not signed in - the gate above has already rendered the login/
        # signup screen. Nothing else in this app is reachable until then.
        return

    st.title("Job Application Agent")
    st.caption(
        "Save your CV once as your baseline, then add a job description. "
        "The agent researches the company, scores your fit, and tailors "
        "your material using only what your baseline CV already says."
    )

    auth.render_account_sidebar(user)
    render_tailored_cv_sidebar_link()
    # The technical "Configuration" sidebar (provider/model names, which
    # keys are set, the tracker db filename) is deliberately not shown to
    # the end user - product decision, not a settings change. `render_sidebar`
    # is untouched and still fully correct; nothing here calls it. The
    # sidebar now shows only the account panel above (who is signed in, and
    # the way out).

    baseline = profile.render_baseline_cv_section(settings, user)
    st.divider()

    # Once a run has completed, the Entry page steps back into a collapsed
    # "start over" section instead of staying front-and-center next to the
    # Results below it - the four stages should read as a guided flow, not
    # a form and a report permanently glued together. Collapsing changes
    # nothing about what runs or when: the widgets inside execute exactly as
    # before (a collapsed `st.expander` still renders its contents on every
    # rerun, only their visibility is toggled), so editing the job
    # description and clicking Start Analysis again still replaces the
    # result exactly as it always has - see `render_entry_page`/`start_run`.
    # Only a *result* (not a failed run) triggers this: on error the user is
    # still mid-Entry, most likely fixing what they just typed.
    if st.session_state.get(RESULT_KEY) is not None:
        with st.expander("✏️ Start a new analysis", expanded=False):
            render_entry_page(settings, baseline, user)
    else:
        render_entry_page(settings, baseline, user)

    render_last_run(settings)
    st.divider()
    render_tailored_cv_history(settings, user)


def render_entry_page(
    settings: Settings, baseline: BaselineCV | None, user: User
) -> None:
    """The Entry page: a job description, plus whichever CV source applies.

    Once a baseline CV exists (see "My Profile" above), this page never asks
    for a CV again - the whole point of the baseline system is that the same
    original CV is reused for every application, not re-supplied per job.
    Always rendered, even once a result exists further down - editing the
    job description here and clicking Start Analysis again is how a result
    gets replaced (see `start_run`); hiding this after a result appeared
    would break that and give no way back to it.
    """
    st.subheader("Step 2 of 4 — 📥 Job Application")

    if baseline is None:
        st.caption(
            "Save a baseline CV in **My Profile** above before running your "
            "first analysis - every application is tailored from it."
        )
    else:
        st.caption(
            "Your baseline CV (saved "
            f"{baseline.updated_at.strftime('%Y-%m-%d %H:%M')}) is used "
            "automatically - manage it in **My Profile** above. Tailoring "
            "for this job never changes your baseline. Upload a job "
            "description file (.txt, .md or .pdf) or paste the text "
            "directly, then click **Start Analysis** below."
        )

    st.markdown("**Job description** — required")
    jd_file = st.file_uploader(
        "Upload (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="jd_file"
    )
    jd_pasted = st.text_area("...or paste it", height=220, key="jd_text")

    st.divider()

    # Better to refuse the click than to spend a minute failing at it.
    blockers = missing_requirements(settings)
    if baseline is None:
        blockers = [*blockers, "A baseline CV is required before a run can start."]
    for blocker in blockers:
        st.error(blocker)
    if blockers:
        st.caption(
            "Set the missing values in your .env file, or save a baseline "
            "CV above, then try again."
        )

    if st.button(
        "🚀 Start Analysis",
        type="primary",
        disabled=bool(blockers),
        key="start_analysis",
    ):
        cv_text = baseline.cv_text if baseline is not None else ""
        start_run(None, cv_text, jd_file, jd_pasted, settings, user, baseline)

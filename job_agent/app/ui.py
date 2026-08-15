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
from job_agent.models import MatchLevel, RequirementMatch

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


def run_pipeline(cv_source: str, jd_source: str, settings: Settings) -> RunContext:
    """Run one application through the existing supervisor.

    Imported lazily so the module can be loaded (and tested) without building
    agents or touching a provider.
    """
    from job_agent.agents import build_supervisor

    return build_supervisor(settings).run(cv_source, jd_source)


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
        st.header("Configuration")
        st.caption("Values are never displayed - only whether they are set.")
        st.write(
            {
                "Parsing / research model": (
                    f"{settings.light.provider} · {settings.light.model}"
                ),
                "Scoring / writing model": (
                    f"{settings.heavy.provider} · {settings.heavy.model}"
                ),
                "Web search": settings.search_provider,
            }
        )

        for label, configured in (
            ("Scoring/writing key", bool(settings.heavy.api_key)),
            ("Parsing key", bool(settings.light.api_key)),
            ("Search key", bool(settings.search_api_key)),
        ):
            st.write(("✅ " if configured else "⚠️ ") + label)

        if not settings.search_api_key:
            st.info("No search key: the run works, company research is skipped.")

        st.caption(f"Tracker: {settings.tracker_db_path.name}")


def render_progress(context: RunContext) -> None:
    """Which stages ran, and how long each took.

    Reconstructed from the run's own trace rather than guessed, so what is
    shown is what actually happened.
    """
    st.subheader("Pipeline")
    durations = {
        event.name: event.duration_ms
        for event in context.trace.events
        if event.name in dict(STAGES)
    }
    for name, label in STAGES:
        if name in durations:
            st.write(f"✅ {label} · {durations[name] / 1000:.1f}s")
        else:
            st.write(f"➖ {label} · skipped")

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


def _render_match_group(matches: list[RequirementMatch]) -> None:
    for match in matches:
        st.write(f"**{match.requirement}**")
        if match.evidence:
            st.caption(f"Evidence from your CV: {match.evidence}")
        if match.reason:
            st.caption(match.reason)


def render_fit_report(context: RunContext) -> None:
    report = context.fit_report
    if report is None:
        return

    st.subheader("Fit report")
    st.metric("Overall fit", f"{report.overall_fit:.0%}")
    st.caption(f"{report.role} at {report.company}")

    by_level: dict[MatchLevel, list[RequirementMatch]] = {
        level: [] for level in MatchLevel
    }
    for match in report.requirement_matches:
        by_level[match.match_level].append(match)

    for level in (MatchLevel.match, MatchLevel.partial, MatchLevel.related):
        matches = by_level[level]
        if not matches:
            continue
        marker, heading = MATCH_LEVEL_DISPLAY[level]
        st.write(f"{marker} **{heading}**")
        _render_match_group(matches)

    missing = by_level[MatchLevel.missing]
    if missing:
        marker, heading = MATCH_LEVEL_DISPLAY[MatchLevel.missing]
        st.write(f"{marker} **{heading}**")
        for match in missing:
            st.write(f"**{match.requirement}**")
            st.caption(MISSING_FROM_CV_CAPTION)


def render_research(context: RunContext) -> None:
    brief = context.brief
    st.subheader("Company research")
    if brief is None or not brief.sources:
        st.info(
            brief.summary
            if brief is not None
            else "No company research was performed for this run."
        )
        return

    st.write(brief.summary)
    if brief.facts:
        for fact in brief.facts:
            st.write(f"- {fact}")
    st.write("**Sources**")
    for source in brief.sources:
        st.write(f"- [{source.title}]({source.url})")


def render_documents(context: RunContext) -> None:
    st.subheader("Tailored CV")
    if context.tailored_cv is None:
        st.warning(
            "No tailored CV was produced. The grounding check could not verify "
            "a draft, so your original CV stands."
        )
    else:
        for bullet in context.tailored_cv.bullets:
            st.write(f"- {bullet}")
        if context.tailored_cv.omitted:
            st.caption("De-prioritised: " + "; ".join(context.tailored_cv.omitted))
        st.download_button(
            "Download tailored CV",
            data="\n".join(f"- {b}" for b in context.tailored_cv.bullets),
            file_name="tailored_cv.txt",
            mime="text/plain",
        )

    st.subheader("Cover letter")
    if context.cover_letter is None:
        st.warning("No cover letter was produced.")
    else:
        st.write(context.cover_letter.body)
        st.download_button(
            "Download cover letter",
            data=context.cover_letter.body,
            file_name="cover_letter.txt",
            mime="text/plain",
        )


def render_validation(context: RunContext) -> None:
    """What the guards did - the part that makes the system trustworthy."""
    st.subheader("Validation")
    st.caption(
        "Generated claims are checked against your CV. Anything the CV does "
        "not support is rejected and the model is asked to rewrite."
    )

    guard_events = [e for e in context.trace.events if e.name in GUARDS]
    retries = [e for e in context.trace.events if e.name.startswith("validate:")]

    if not guard_events:
        st.success("Every generated claim was supported by your CV.")
    for event in guard_events:
        st.write(f"**{GUARDS[event.name]}**")
        if event.result:
            st.caption(event.result)

    if retries:
        st.caption(
            f"{len(retries)} response(s) were rejected for malformed output and "
            "regenerated."
        )


def render_application(context: RunContext, settings: Settings) -> None:
    st.subheader("Application tracker")
    if context.application is not None:
        record = context.application
        st.write(
            f"**{record.role}** at **{record.company}** · fit "
            f"{record.fit_score:.0%} · status `{record.status.value}`"
        )
        st.caption(f"Application id: {record.application_id}")
        st.caption("Nothing is submitted anywhere - this is a saved draft.")

    try:
        from job_agent.memory import SQLiteApplicationTracker

        rows = SQLiteApplicationTracker(settings.tracker_db_path).list()
    except Exception as exc:  # noqa: BLE001 - the tracker view is never fatal
        st.caption(f"Could not read the tracker: {safe_error(exc, settings)}")
        return

    if rows:
        st.write(f"**All tracked applications ({len(rows)})**")
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

    render_progress(context)
    render_fit_report(context)
    render_research(context)
    render_documents(context)
    render_validation(context)
    render_application(context, settings)


# --- running -----------------------------------------------------------------


def start_run(
    cv_upload, cv_pasted: str, jd_upload, jd_pasted: str, settings: Settings
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

        with st.spinner("Running the pipeline - this takes about a minute..."):
            try:
                context = run_pipeline(cv, jd, settings)
            except Exception as exc:  # noqa: BLE001 - the UI must not crash
                # The redacted message, not the exception: nothing carrying a
                # credential is worth keeping across reruns.
                st.session_state[ERROR_KEY] = safe_error(exc, settings)
                return

    st.session_state[RESULT_KEY] = context


def render_last_run(settings: Settings) -> None:
    """Whatever the most recent run produced - on every rerun, not just the
    one that started it. Before the first run there is nothing to show."""
    error = st.session_state.get(ERROR_KEY)
    if error is not None:
        st.error("The run could not be completed. Nothing was saved.\n\n" + error)
        return

    context = st.session_state.get(RESULT_KEY)
    if context is None:
        return

    st.success("Run complete.")
    render_results(context, settings)


# --- page --------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Job Application Agent", page_icon="📄")
    st.title("Job Application Agent")
    st.caption(
        "Paste a CV and a job description. The agent researches the company, "
        "scores your fit, and tailors your material using only what your CV "
        "already says."
    )

    settings = load_settings()
    render_sidebar(settings)

    left, right = st.columns(2)
    with left:
        st.subheader("Your CV")
        cv_file = st.file_uploader(
            "Upload (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="cv_file"
        )
        cv_pasted = st.text_area("...or paste it", height=220, key="cv_text")
    with right:
        st.subheader("Job description")
        jd_file = st.file_uploader(
            "Upload (.txt, .md, .pdf)", type=["txt", "md", "pdf"], key="jd_file"
        )
        jd_pasted = st.text_area("...or paste it", height=220, key="jd_text")

    # Better to refuse the click than to spend a minute failing at it.
    blockers = missing_requirements(settings)
    for blocker in blockers:
        st.error(blocker)
    if blockers:
        st.caption("Set the missing values in your .env file, then restart the app.")

    if st.button("Run application", type="primary", disabled=bool(blockers)):
        start_run(cv_file, cv_pasted, jd_file, jd_pasted, settings)

    render_last_run(settings)

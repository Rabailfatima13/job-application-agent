"""Supervisor: the Week 6 end-to-end pipeline.

    CV + JD -> parse_cv -> parse_jd -> research -> score -> track

Orchestration only. The supervisor decides *what runs when* and assembles the
result; it never parses, researches or scores itself - each of those already
belongs to a component that is tested on its own. Read top to bottom, the graph
below is the whole Week 6 story, which is the point: it should be explainable
in one screen at the final presentation.

Built on LangGraph (architecture decision D1). The pipeline is linear today;
the branches the proposal describes - fit threshold, no-fabrication retry, the
next-role loop - attach to this graph in Week 7 rather than replacing it.

The state that flows between nodes is the existing `RunContext`, so "what the
research agent produced" and "what the scoring agent may read" remain the same
object, with no hidden channel between workers.
"""

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..config import Settings
from ..llm import ModelRouter
from ..memory.parsed_cv_cache import ParsedCVCache
from ..memory.research_cache import CompanyResearchCache
from ..memory.session import RunContext
from ..memory.tracker import ApplicationTracker, SQLiteApplicationTracker
from ..models import ApplicationRecord, ApplicationStatus
from ..observability import TraceCollector, traced_tool_call
from ..tools import ToolRegistry, WebSearchClient, build_default_registry
from ..validation import RetryExhaustedError
from .research import ResearchAgent
from .scoring import ScoringAgent
from .writing import WritingAgent

PARSE_CV = "parse_cv"
PARSE_JD = "parse_jd"
RESEARCH = "research"
SCORE = "score_fit"
WRITE = "write_application"
TRACK = "track_application"

#: The Week 6 core, always present.
PIPELINE_NODES = (PARSE_CV, PARSE_JD, RESEARCH, SCORE, TRACK)
#: With a writing agent injected, `write_application` runs between them.
PIPELINE_NODES_WITH_WRITING = (PARSE_CV, PARSE_JD, RESEARCH, SCORE, WRITE, TRACK)

RESEARCH_FAILED_WARNING = (
    "Company research could not be completed, so the fit report was produced "
    "without a company brief."
)
WRITING_FAILED_WARNING = (
    "Tailored documents could not be produced without unsupported claims, so "
    "none were written. The original CV and the gap list stand."
)


def new_application_id() -> str:
    """A fresh opaque id per application.

    Not derived from role or company on purpose: the same candidate applying to
    two roles at one company - or twice to the same role - must produce
    distinct records (FR-13). Injectable, so tests get predictable ids.
    """
    return uuid4().hex[:12]


class Supervisor:
    """Runs one application through the pipeline and returns the full context.

    Everything it coordinates is injected: the tool registry (parsers), the two
    worker agents, and the tracker. It therefore has no idea whether the
    tracker is SQLite or in-memory, and never opens a database itself.
    """

    name = "supervisor"

    def __init__(
        self,
        tools: ToolRegistry,
        research: ResearchAgent,
        scoring: ScoringAgent,
        tracker: ApplicationTracker,
        collector: TraceCollector | None = None,
        id_factory: Callable[[], str] = new_application_id,
        clock: Callable[[], datetime] = datetime.now,
        writing: WritingAgent | None = None,
    ) -> None:
        self.tools = tools
        self.research = research
        self.scoring = scoring
        self.tracker = tracker
        # Optional: without a writing agent the graph is the Week 6 pipeline
        # exactly, which is also what a run below the fit threshold will want
        # once that branch lands.
        self.writing = writing
        self.collector = collector
        self._id_factory = id_factory
        self._clock = clock
        self.graph = self._build_graph()

    # --- the graph ----------------------------------------------------------

    def _build_graph(self):
        """START -> parse_cv -> parse_jd -> research -> score_fit ->
        [write_application] -> track_application -> END.

        Writing sits after scoring because it consumes the fit report's
        recommended emphasis and gaps, and before tracking so the record is
        still written last. Grounding is not a node of its own: it validates
        the writer's output, so it belongs inside the writer's own
        validate-and-retry loop, exactly as the proposal's activity diagram
        shows it.
        """
        builder = StateGraph(RunContext)
        builder.add_node(PARSE_CV, self._parse_cv)
        builder.add_node(PARSE_JD, self._parse_jd)
        builder.add_node(RESEARCH, self._research)
        builder.add_node(SCORE, self._score)
        builder.add_node(TRACK, self._track)

        builder.add_edge(START, PARSE_CV)
        builder.add_edge(PARSE_CV, PARSE_JD)
        builder.add_edge(PARSE_JD, RESEARCH)
        builder.add_edge(RESEARCH, SCORE)
        if self.writing is not None:
            builder.add_node(WRITE, self._write)
            builder.add_edge(SCORE, WRITE)
            builder.add_edge(WRITE, TRACK)
        else:
            builder.add_edge(SCORE, TRACK)
        builder.add_edge(TRACK, END)
        return builder.compile()

    def run(self, cv_source: str | Path, jd_source: str | Path) -> RunContext:
        """Run the pipeline for one CV and one job description.

        Both arguments take whatever the parsing layer already accepts: raw
        text, or a path to a .txt/.md/.pdf file.

        Returns the finished `RunContext` - parsed inputs, company brief, fit
        report, the persisted application record, warnings and the trace - so
        the eventual UI has everything without a second result hierarchy.

        Failures that mean the result would be untrustworthy (a CV that cannot
        be parsed, a fit report that never validated) propagate as the
        project's existing `RetryExhaustedError`; nothing partial is written to
        the tracker.
        """
        initial = RunContext(
            cv_text=str(cv_source),
            jd_text=str(jd_source),
            trace=self.collector or TraceCollector(),
        )
        return RunContext(**self.graph.invoke(initial))

    # --- nodes --------------------------------------------------------------

    def _traced(self, name: str):
        return traced_tool_call(self.collector, agent=self.name, name=name)

    def _parse_cv(self, state: RunContext) -> dict:
        with self._traced(PARSE_CV) as outcome:
            parsed = self.tools.get(PARSE_CV).run(source=state.cv_text)
            outcome["result"] = f"{len(parsed.evidence)} evidence item(s)"
        return {"parsed_cv": parsed}

    def _parse_jd(self, state: RunContext) -> dict:
        with self._traced(PARSE_JD) as outcome:
            job = self.tools.get(PARSE_JD).run(source=state.jd_text)
            outcome["result"] = f"{job.role} at {job.company}"
        return {"job": job}

    def _research(self, state: RunContext) -> dict:
        """Research is best-effort: a company brief improves the application
        but is not required to judge candidate-role fit.

        The agent already degrades to an honest brief when the company is
        unnamed, the search fails, or nothing comes back. This catches the one
        case it cannot - summarisation never validating - so an unusable brief
        still does not cost the user their fit report.
        """
        with self._traced(RESEARCH) as outcome:
            try:
                brief = self.research.run(state)
            except RetryExhaustedError as exc:
                outcome["result"] = f"skipped: {exc}"
                return {"warnings": [*state.warnings, RESEARCH_FAILED_WARNING]}
            outcome["result"] = f"{len(brief.facts)} fact(s)"
        return {"brief": brief, "warnings": list(state.warnings)}

    def _score(self, state: RunContext) -> dict:
        """Scoring is essential: no fit report, no application record. A
        failure here raises rather than inventing a score, because a failed
        score and a genuinely poor fit are not the same thing."""
        with self._traced(SCORE) as outcome:
            report = self.scoring.run(state)
            outcome["result"] = f"overall_fit={report.overall_fit}"
        return {"fit_report": report}

    def _write(self, state: RunContext) -> dict:
        """Tailor the CV and draft the cover letter, if it can be done honestly.

        The writer refuses to return material carrying unsupported claims. When
        that happens the run still completes: the application is tracked, and
        the user keeps their original CV plus the gap list - the proposal's
        stated fallback - rather than receiving a rewrite nobody can vouch for.
        """
        with self._traced(WRITE) as outcome:
            try:
                tailored, letter = self.writing.run(state)
            except RetryExhaustedError as exc:
                outcome["result"] = f"skipped: {exc}"
                return {"warnings": [*state.warnings, WRITING_FAILED_WARNING]}
            outcome["result"] = f"{len(tailored.bullets)} bullet(s) + cover letter"
        return {
            "tailored_cv": tailored,
            "cover_letter": letter,
            "warnings": list(state.warnings),
        }

    def _track(self, state: RunContext) -> dict:
        """Persist the application through the injected tracker.

        New applications always start in `draft`: the agent never marks
        anything submitted, and nothing is sent anywhere (FR-14).
        """
        report = state.fit_report
        with self._traced(TRACK) as outcome:
            record = ApplicationRecord(
                application_id=self._id_factory(),
                role=report.role,
                company=report.company,
                fit_score=report.overall_fit,
                status=ApplicationStatus.draft,
                created_at=self._clock(),
            )
            self.tracker.add(record)
            outcome["result"] = record.application_id
        return {"application": record}


def build_supervisor(
    settings: Settings,
    router: ModelRouter | None = None,
    tracker: ApplicationTracker | None = None,
    collector: TraceCollector | None = None,
) -> Supervisor:
    """Wire the production objects together.

    The only place that knows the real tracker is SQLite and that search needs
    credentials. `Supervisor` itself stays ignorant of both, which is what lets
    tests hand it an in-memory tracker and a stub search client.
    """
    router = router or ModelRouter(settings)
    collector = collector or TraceCollector(log_path=settings.tool_call_log_path)
    tracker = tracker or SQLiteApplicationTracker(settings.tracker_db_path)
    search_client = WebSearchClient.from_settings(settings)
    # Same tracker database file, one more table each - not a second store to
    # wire up or a second thing that can go out of sync with either.
    research_cache = CompanyResearchCache(
        settings.tracker_db_path, ttl_hours=settings.research_cache_ttl_hours
    )
    cv_cache = ParsedCVCache(settings.tracker_db_path)
    registry = build_default_registry(
        router, collector, search_client=search_client, cv_cache=cv_cache
    )

    return Supervisor(
        tools=registry,
        research=ResearchAgent(
            router, tools=registry, collector=collector, cache=research_cache
        ),
        scoring=ScoringAgent(router, collector=collector),
        # settings.skip_tailoring reuses the same "no writer injected" shape
        # Supervisor already supports (see _build_graph): the graph simply
        # omits the write_application node, exactly as it always could.
        writing=None
        if settings.skip_tailoring
        else WritingAgent(router, collector=collector),
        tracker=tracker,
        collector=collector,
    )

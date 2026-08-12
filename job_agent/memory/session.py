"""Session memory: everything one application run carries between agents.

This is also the state object the supervisor graph passes from node to node,
so "what the research agent produced" and "what the scoring agent may read"
are the same thing - there is no hidden channel between workers.

Fields are filled in pipeline order and are None until their step has run.
"""

from dataclasses import dataclass, field

from ..models import (
    ApplicationRecord,
    CompanyBrief,
    CoverLetter,
    FitReport,
    JobDescription,
    ParsedCV,
    TailoredCV,
)
from ..observability import TraceCollector


@dataclass
class RunContext:
    """One application run: inputs, per-step outputs, and its trace."""

    # Inputs (raw, as supplied by the user)
    cv_text: str
    jd_text: str

    # Filled by the supervisor's parsing step
    parsed_cv: ParsedCV | None = None
    job: JobDescription | None = None

    # Filled by the workers
    brief: CompanyBrief | None = None
    fit_report: FitReport | None = None
    tailored_cv: TailoredCV | None = None  # Week 7
    cover_letter: CoverLetter | None = None  # Week 7

    # Written by the supervisor's final step, once the fit report exists.
    application: ApplicationRecord | None = None

    # Non-fatal problems worth showing the user (e.g. web search unavailable,
    # tailoring fell back to the original CV) - the run continues.
    warnings: list[str] = field(default_factory=list)

    trace: TraceCollector = field(default_factory=TraceCollector)

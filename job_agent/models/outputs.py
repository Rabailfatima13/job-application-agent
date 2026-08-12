"""What each worker agent produces.

Field shapes follow the proposal's structured-output contract (S13) exactly:
the fit report is {role, company, overall_fit (0-1), requirement_matches,
gaps, recommended_emphasis}.
"""

from pydantic import BaseModel, ConfigDict, Field


class Source(BaseModel):
    """Where a researched fact came from - kept so the brief is checkable."""

    model_config = ConfigDict(extra="forbid")

    title: str
    url: str


class CompanyBrief(BaseModel):
    """Research agent output: short, factual, role-relevant, sourced (FR-2)."""

    model_config = ConfigDict(extra="forbid")

    company: str
    summary: str = Field(description="A few sentences of role-relevant fact.")
    facts: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class RequirementMatch(BaseModel):
    """One job requirement judged against the CV, with the evidence used.

    `evidence` must be text drawn from the CV, not a paraphrase invented by the
    model - that is what makes the score auditable (US-3).
    """

    model_config = ConfigDict(extra="forbid")

    requirement: str
    evidence: str = Field(
        default="", description="Supporting text from the CV; empty when unmet."
    )
    met: bool


class FitReport(BaseModel):
    """Scoring agent output: an evidence-backed score plus explicit gaps (FR-4)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    overall_fit: float = Field(ge=0.0, le=1.0)
    requirement_matches: list[RequirementMatch] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    recommended_emphasis: list[str] = Field(default_factory=list)


class TailoredCV(BaseModel):
    """Writing agent output (Week 7): the same claims, re-ordered and
    re-emphasised for this role. `bullets` must each be grounded in the source
    CV; `omitted` records what was de-prioritised, never anything invented."""

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    bullets: list[str] = Field(default_factory=list)
    omitted: list[str] = Field(default_factory=list)


class CoverLetter(BaseModel):
    """Writing agent output (Week 7): a draft grounded in the CV and the
    company brief. Always a draft - nothing is ever auto-sent (FR-14)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    body: str

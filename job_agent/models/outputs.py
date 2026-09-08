"""What each worker agent produces.

Field shapes follow the proposal's structured-output contract (S13) exactly:
the fit report is {role, company, overall_fit (0-1), requirement_matches,
gaps, recommended_emphasis}.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MatchLevel(StrEnum):
    """How well a single requirement is satisfied by the CV evidence.

    A live run scored 10% and reported "the CV never mentions Bachelor" for a
    candidate whose CV said "BS Data Science student", and "Python" as flatly
    missing when the CV said "Basic knowledge on ... Python" - because the
    scoring model could only ever answer met/unmet. A CV that has *some*
    relevant evidence at a lower proficiency than the posting asks for is a
    real, different situation from having none at all, and collapsing both
    into "unmet" is what produced an inaccurate score and gaps list.
    """

    match = "match"
    partial = "partial"
    related = "related"
    missing = "missing"


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

    `met` is kept as a plain derived convenience (`match_level == MATCH`) for
    any caller that only ever needed a yes/no - it is set alongside
    `match_level`, never independently, so the two can never disagree.
    """

    model_config = ConfigDict(extra="forbid")

    requirement: str
    evidence: str = Field(
        default="", description="Supporting text from the CV; empty when unmet."
    )
    match_level: MatchLevel = MatchLevel.missing
    reason: str = Field(
        default="",
        description="Plain-language explanation for the report - never a "
        "source of new claims, only a display of what evidence_index "
        "already supports.",
    )
    met: bool = False

    @model_validator(mode="after")
    def _met_follows_match_level(self) -> "RequirementMatch":
        """`met` is derived, not independent - whatever was passed in is
        overwritten here so the two fields can never actually disagree,
        rather than that only being true by convention at the one call site
        that constructs these during real scoring."""
        self.met = self.match_level == MatchLevel.match
        return self


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
    CV; `omitted` records what was de-prioritised, never anything invented.

    `full_text` is the complete tailored CV - assembled deterministically
    from `bullets`, `omitted`, and the candidate's own name/skills (see
    `agents/writing.py`'s `assemble_full_cv`), not a second model call - so
    what a user downloads or saves as a tailored CV version is an actual CV
    document, not just the handful of highlighted bullets. Defaults to ""
    only for callers (mostly tests) that build a `TailoredCV` directly
    without going through the writing agent.
    """

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    bullets: list[str] = Field(default_factory=list)
    omitted: list[str] = Field(default_factory=list)
    full_text: str = Field(
        default="",
        description="The complete tailored CV document, ready to save or download.",
    )


class CoverLetter(BaseModel):
    """Writing agent output (Week 7): a draft grounded in the CV and the
    company brief. Always a draft - nothing is ever auto-sent (FR-14)."""

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    body: str

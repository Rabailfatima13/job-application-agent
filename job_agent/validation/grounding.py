"""No-fabrication grounding check - the project's core trust requirement
(FR-10, NFR-4). Implemented in Week 7; the contract is fixed here because the
rest of the system is designed around it from the start.

The rule the checker will enforce: every tailored line and every
requirement_match evidence string must be supported by text present in the
source CV. Re-ordering, re-emphasis, and rewording that preserves the claim
are allowed; a new job, project, skill, achievement, year count, or technology
is not. Unsupported lines are flagged and stripped, and if nothing survives,
the caller falls back to the original CV plus the gap list.

Deliberately a separate module from schema_guard: a perfectly schema-valid
tailored CV can still be a fabrication, so this check runs after that one and
can never be satisfied by it.
"""

from pydantic import BaseModel, ConfigDict, Field

from ..models import ParsedCV


class GroundingIssue(BaseModel):
    """One claim that could not be traced back to the source CV."""

    model_config = ConfigDict(extra="forbid")

    claim: str
    reason: str


class GroundingResult(BaseModel):
    """Outcome of checking a set of generated lines against the source CV."""

    model_config = ConfigDict(extra="forbid")

    grounded: list[str] = Field(default_factory=list)
    issues: list[GroundingIssue] = Field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.issues


def check_grounding(claims: list[str], source_cv: ParsedCV) -> GroundingResult:
    """Split `claims` into those supported by `source_cv` and those that are not.

    Week 7 deliverable. Planned approach: a cheap deterministic pass (entity
    and number extraction - technologies, employers, year counts - that must
    appear in source_cv.raw_text) followed by a model-based entailment check
    per surviving claim, so an obvious fabrication is caught without an LLM
    call and a subtle one still is.
    """
    raise NotImplementedError("Week 7: no-fabrication grounding check")

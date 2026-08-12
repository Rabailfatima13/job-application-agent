"""Parsed representations of the two user-supplied inputs: a CV and a job
description (FR-1, FR-3).

`CVEvidence` is the unit the whole trust story rests on: every claim the system
later makes about the candidate must point back at one of these, and the
no-fabrication check (FR-10) works by asking "which CVEvidence supports this?".
"""

from pydantic import BaseModel, ConfigDict, Field


class CVEvidence(BaseModel):
    """One atomic, verbatim-sourced claim taken from the candidate's CV.

    `text` is the candidate's own wording. It is never rewritten during
    parsing, because it is the reference the grounding check compares against.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(description="The claim, as written in the source CV.")
    section: str | None = Field(
        default=None, description="Where in the CV it came from, e.g. 'Experience'."
    )
    skills: list[str] = Field(
        default_factory=list, description="Skills/technologies named in this claim."
    )


class ParsedCV(BaseModel):
    """A CV reduced to a list of evidence items plus the raw source text."""

    model_config = ConfigDict(extra="forbid")

    candidate_name: str | None = None
    raw_text: str = Field(description="Full source text, kept for grounding checks.")
    evidence: list[CVEvidence] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class RoleRequirement(BaseModel):
    """One requirement extracted from the job description."""

    model_config = ConfigDict(extra="forbid")

    text: str
    must_have: bool = Field(
        default=True, description="False for 'nice to have' / preferred requirements."
    )


class JobDescription(BaseModel):
    """A job posting reduced to role, company, and discrete requirements."""

    model_config = ConfigDict(extra="forbid")

    role: str
    company: str
    raw_text: str
    requirements: list[RoleRequirement] = Field(default_factory=list)
    location: str | None = None

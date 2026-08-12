"""Scoring agent: `ParsedCV` + `JobDescription` -> `FitReport` (FR-4, US-3).

The design principle here is that **the model judges, the code scores**. Asking
an LLM for "a number between 0 and 1" produces something unexplainable and
unstable; instead the model answers one narrow question per requirement - is
this met, and which CV evidence shows it - and application logic turns those
answers into the score using a fixed rubric (see `compute_fit_score`).

The second principle is that **the model never writes evidence text**. It
returns an *index* into the CV evidence the parser already extracted, and the
code looks up the wording. A model that wants to claim "5 years of Python"
has nowhere to put it: there is no free-text evidence field in what it
returns. Anything it points at that does not exist is dropped by
`_judge_requirements`, which then downgrades the match to unmet.
"""

from pydantic import BaseModel, ConfigDict, Field

from ..memory.session import RunContext
from ..models import (
    CVEvidence,
    FitReport,
    JobDescription,
    ParsedCV,
    RequirementMatch,
    RoleRequirement,
)
from ..observability import traced_tool_call
from ..validation import generate_validated
from .base import BaseAgent

SCORING_STEP = "score_fit"
SCORING_MAX_TOKENS = 2048

# --- The rubric -------------------------------------------------------------
# A must-have counts double a nice-to-have. The posting's own "nice to have"
# wording is what sets this, and the JD parser already recorded it as
# RoleRequirement.must_have, so nothing is guessed at scoring time.
MUST_HAVE_WEIGHT = 1.0
NICE_TO_HAVE_WEIGHT = 0.5


def requirement_weight(requirement: RoleRequirement) -> float:
    return MUST_HAVE_WEIGHT if requirement.must_have else NICE_TO_HAVE_WEIGHT


def compute_fit_score(judged: list[tuple[RoleRequirement, bool]]) -> float:
    """The scoring rule, in one place so it can be explained and tested:

        overall_fit = (weight of met requirements) / (weight of all requirements)

    with must-haves weighted 1.0 and nice-to-haves 0.5. Equal-weight postings
    therefore reduce to the obvious "3 of 5 requirements met = 0.6".

    A posting with no requirements scores 0.0: there is nothing the candidate
    has been shown to meet, so claiming a fit would be unfounded.
    """
    total = sum(requirement_weight(requirement) for requirement, _ in judged)
    if total == 0:
        return 0.0
    earned = sum(requirement_weight(r) for r, met in judged if met)
    return round(min(max(earned / total, 0.0), 1.0), 3)


# --- What the model is asked to return ---------------------------------------


class RequirementJudgement(BaseModel):
    """One requirement, judged. `evidence_index` points into the CV evidence
    list the parser produced - the model does not get to write evidence."""

    model_config = ConfigDict(extra="forbid")

    requirement_index: int
    met: bool
    evidence_index: int | None = None


class ScoringJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[RequirementJudgement] = Field(default_factory=list)
    recommended_emphasis: list[int] = Field(
        default_factory=list,
        description="Indices of CV evidence worth emphasising for this role.",
    )


SCORING_SYSTEM_PROMPT = """\
You judge whether a candidate meets each requirement of a job posting. You are \
a judge, not a writer, and you never produce a score.

You are given numbered REQUIREMENTS and numbered CV EVIDENCE.

For every requirement, in order, return one judgement:
- requirement_index: the requirement's number.
- met: true only when a specific piece of CV evidence demonstrates it. If the \
CV does not show it, met is false - do not give the benefit of the doubt, and \
do not infer experience the CV does not state.
- evidence_index: the number of the single best piece of evidence supporting \
it, or null when met is false.

Also return recommended_emphasis: the numbers of the 2-5 pieces of evidence \
most worth foregrounding for this role, strongest first.

Only use the numbers you were given. Never invent a requirement, an experience, \
a skill, a technology or a duration. Judge only what is written.

Reply with a single JSON object and nothing else:
{"judgements": [{"requirement_index": int, "met": bool,
                 "evidence_index": int|null}],
 "recommended_emphasis": [int]}"""


def _format_numbered(items: list[str]) -> str:
    return "\n".join(f"[{index}] {text}" for index, text in enumerate(items))


def build_scoring_prompt(cv: ParsedCV, job: JobDescription) -> str:
    """The user message: the two structured inputs, numbered.

    Only `job.requirements` is sent. Responsibilities are deliberately left out
    - the candidate is not scored on what the role involves doing - and so is
    the raw CV text, so the model can only point at parsed evidence.
    """
    return (
        f"ROLE: {job.role}\n"
        f"COMPANY: {job.company}\n\n"
        "REQUIREMENTS:\n"
        f"{_format_numbered([r.text for r in job.requirements])}\n\n"
        "CV EVIDENCE:\n"
        f"{_format_numbered([e.text for e in cv.evidence])}\n\n"
        f"CANDIDATE SKILLS: {', '.join(cv.skills) or 'none listed'}"
    )


class ScoringAgent(BaseAgent):
    """Produces an evidence-backed `FitReport`. Owns judgement and scoring;
    knows nothing about company research or document writing."""

    name = "scoring"

    def run(self, context: RunContext) -> FitReport:
        if context.parsed_cv is None or context.job is None:
            raise ValueError(
                "ScoringAgent needs a parsed CV and job description on the "
                "RunContext; run the parsers first."
            )
        return self.score(context.parsed_cv, context.job)

    def score(self, cv: ParsedCV, job: JobDescription) -> FitReport:
        judgement = self._judge(cv, job)
        matches, judged = self._resolve_matches(judgement, cv, job)
        return FitReport(
            role=job.role,
            company=job.company,
            overall_fit=compute_fit_score(judged),
            requirement_matches=matches,
            # A gap is simply a requirement the CV did not demonstrate,
            # must-haves first so the biggest problems read first (US-3).
            gaps=[
                requirement.text
                for requirement, met in sorted(
                    judged, key=lambda pair: not pair[0].must_have
                )
                if not met
            ],
            recommended_emphasis=self._resolve_emphasis(judgement, cv),
        )

    def _judge(self, cv: ParsedCV, job: JobDescription) -> ScoringJudgement:
        """One bounded-retry model call returning per-requirement judgements."""
        prompt = build_scoring_prompt(cv, job)

        def produce(feedback: str | None) -> str:
            content = prompt
            if feedback:
                content = (
                    f"{prompt}\n\n---\n"
                    f"Your previous reply was rejected by the schema validator:\n"
                    f"{feedback}\n"
                    "Return corrected JSON only, following the schema exactly."
                )
            result = self.call_model(
                SCORING_STEP,
                [{"role": "user", "content": content}],
                system=SCORING_SYSTEM_PROMPT,
                max_tokens=SCORING_MAX_TOKENS,
            )
            return result.text

        return generate_validated(
            produce,
            ScoringJudgement,
            collector=self.collector,
            agent=self.name,
        )

    def _resolve_matches(
        self, judgement: ScoringJudgement, cv: ParsedCV, job: JobDescription
    ) -> tuple[list[RequirementMatch], list[tuple[RoleRequirement, bool]]]:
        """Turn the model's judgements into `RequirementMatch`es, enforcing the
        evidence rules deterministically.

        The requirement list drives the loop, not the model's reply, which
        means: a requirement the model forgot still appears (as unmet), a
        requirement it invented is discarded, and a "met" claim without usable
        evidence from `cv.evidence` is downgraded to unmet. Every rejection is
        traced rather than silently applied.
        """
        by_index = {j.requirement_index: j for j in judgement.judgements}
        rejected: list[str] = []

        matches: list[RequirementMatch] = []
        judged: list[tuple[RoleRequirement, bool]] = []

        for index, requirement in enumerate(job.requirements):
            claim = by_index.get(index)
            evidence = self._evidence_at(cv, claim.evidence_index) if claim else None
            met = bool(claim and claim.met and evidence is not None)

            if claim and claim.met and evidence is None:
                rejected.append(f"unsupported match: {requirement.text}")

            matches.append(
                RequirementMatch(
                    requirement=requirement.text,
                    evidence=evidence.text if met and evidence else "",
                    met=met,
                )
            )
            judged.append((requirement, met))

        invented = [i for i in by_index if not 0 <= i < len(job.requirements)]
        rejected.extend(f"unknown requirement index: {i}" for i in invented)

        if rejected and self.collector is not None:
            with traced_tool_call(
                self.collector,
                agent=self.name,
                name="scoring:evidence_guard",
                arguments={"rejected": len(rejected)},
            ) as outcome:
                outcome["result"] = "; ".join(rejected)

        return matches, judged

    @staticmethod
    def _evidence_at(cv: ParsedCV, index: int | None) -> CVEvidence | None:
        """The CV evidence at `index`, or None if the model pointed nowhere real."""
        if index is None or not 0 <= index < len(cv.evidence):
            return None
        return cv.evidence[index]

    def _resolve_emphasis(self, judgement: ScoringJudgement, cv: ParsedCV) -> list[str]:
        """Emphasis suggestions, as the candidate's own evidence text.

        These feed the Week 7 writing agent, so they must be real CV lines:
        out-of-range indices are dropped and duplicates collapsed, preserving
        the model's ordering (strongest first).
        """
        emphasis: list[str] = []
        for index in judgement.recommended_emphasis:
            evidence = self._evidence_at(cv, index)
            if evidence is not None and evidence.text not in emphasis:
                emphasis.append(evidence.text)
        return emphasis

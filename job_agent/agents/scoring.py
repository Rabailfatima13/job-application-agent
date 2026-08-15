"""Scoring agent: `ParsedCV` + `JobDescription` -> `FitReport` (FR-4, US-3).

The design principle here is that **the model judges, the code scores**. Asking
an LLM for "a number between 0 and 1" produces something unexplainable and
unstable; instead the model answers one narrow question per requirement - how
well does the evidence satisfy it, and which CV evidence shows it - and
application logic turns those answers into the score using a fixed rubric
(see `compute_fit_score`).

The second principle is that **the model never writes evidence text**. It
returns an *index* into the CV evidence the parser already extracted, and the
code looks up the wording. A model that wants to claim "5 years of Python"
has nowhere to put it: there is no free-text evidence field in what it
returns. Anything it points at that does not exist is dropped by
`_resolve_matches`, which then downgrades the match to missing.

The third is that **met/unmet is not enough**. A live run scored a real CV
10% because its only two states collapsed "the CV explicitly says this" and
"the CV shows related-but-lower evidence" and "the CV never touches this" into
the same "unmet" bucket - "Basic knowledge on ... Python" and pure silence on
pandas were judged identically. `MatchLevel` (match/partial/related/missing,
see `models/outputs.py`) is the fix: still a judgement the model makes and the
code scores, still index-grounded, just a rubric with the granularity the
proposal's own examples need.
"""

from pydantic import BaseModel, ConfigDict, Field

from ..memory.session import RunContext
from ..models import (
    CVEvidence,
    FitReport,
    JobDescription,
    MatchLevel,
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

# How much of a requirement's weight each match strength earns. A CV that
# only shows related, lower-proficiency, or adjacent evidence earns something
# rather than nothing - but strictly less than a direct match, and never as
# little as genuine silence. Deliberately a small fixed table, not a formula
# a stray token could nudge: the same reason MUST_HAVE_WEIGHT is a constant.
MATCH_LEVEL_WEIGHT: dict[MatchLevel, float] = {
    MatchLevel.match: 1.0,
    MatchLevel.partial: 0.5,
    MatchLevel.related: 0.25,
    MatchLevel.missing: 0.0,
}


def requirement_weight(requirement: RoleRequirement) -> float:
    return MUST_HAVE_WEIGHT if requirement.must_have else NICE_TO_HAVE_WEIGHT


def compute_fit_score(judged: list[tuple[RoleRequirement, float]]) -> float:
    """The scoring rule, in one place so it can be explained and tested:

        overall_fit = (weight earned) / (weight of all requirements)

    with must-haves weighted 1.0 and nice-to-haves 0.5, and each requirement
    earning its weight times its match strength (see `MATCH_LEVEL_WEIGHT`) -
    1.0 for a direct match, down to 0.0 for no evidence at all. A plain `bool`
    works here too (`True`/`False` behave as `1.0`/`0.0`), so this is the same
    rubric as before, only widened to accept a partial or related match
    alongside a full one.

    A posting with no requirements scores 0.0: there is nothing the candidate
    has been shown to meet, so claiming a fit would be unfounded.
    """
    total = sum(requirement_weight(requirement) for requirement, _ in judged)
    if total == 0:
        return 0.0
    earned = sum(requirement_weight(r) * strength for r, strength in judged)
    return round(min(max(earned / total, 0.0), 1.0), 3)


# --- What the model is asked to return ---------------------------------------


class RequirementJudgement(BaseModel):
    """One requirement, judged. `evidence_index` points into the CV evidence
    list the parser produced - the model does not get to write evidence."""

    model_config = ConfigDict(extra="forbid")

    requirement_index: int
    match_level: MatchLevel
    evidence_index: int | None = None
    reason: str = Field(
        default="",
        description="One short sentence for the human-facing report.",
    )


class ScoringJudgement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    judgements: list[RequirementJudgement] = Field(default_factory=list)
    recommended_emphasis: list[int] = Field(
        default_factory=list,
        description="Indices of CV evidence worth emphasising for this role.",
    )


SCORING_SYSTEM_PROMPT = """\
You judge how well a candidate's CV evidence satisfies each requirement of a \
job posting. You are a judge, not a writer, and you never produce a score.

You are given numbered REQUIREMENTS and numbered CV EVIDENCE.

For every requirement, in order, return one judgement with a match_level.

Check subject relevance BEFORE proficiency level. A requirement names a \
subject - a technology, library, domain or skill area - and evidence only \
counts toward it if the evidence is actually about that subject. Ask "is \
this evidence about the same thing the requirement asks about?" before you \
ever ask "how strong is it?". Evidence about a different subject is not weak \
evidence for this requirement, it is no evidence for it, no matter how \
strong or how basic that other subject is: basic CSS/HTML knowledge does not \
become partial evidence for "basic understanding of APIs" just because both \
are basic web-adjacent topics - CSS and APIs are different subjects. Basic \
SQL or Python does not become partial evidence for "machine learning \
concepts" or for "APIs" for the same reason. A C++ project does not become \
related evidence for "pandas and NumPy" for the same reason. Once - and only \
once - the subject genuinely matches does proficiency decide match vs. \
partial.

- match: the evidence clearly satisfies the requirement, including the \
proficiency level it asks for. A conventional abbreviation counts - "BS" \
satisfies "Bachelor's degree", "ML" satisfies "machine learning".
- partial: the CV shows evidence of the *same* skill or technology the \
requirement names, just at a lower proficiency than asked for - "basic \
knowledge", "currently learning", "an online course" when the posting wants \
"strong" or "experience with". The technology has to match; only the level \
is lower. Evidence of a *different* skill is never partial evidence for this \
one, no matter how nearby the two seem: "basic knowledge of SQL and Python" \
is not partial evidence for "machine learning concepts", and it is not \
partial evidence for "APIs" either - those are different topics from SQL and \
Python, not a weaker version of the same topic.
- related: the CV shows evidence that is genuinely adjacent to the specific \
technology, library or domain the requirement names - not merely "some real \
CV content exists, somewhere". Coursework in Data Structures and Calculus is \
related to "analytical and problem-solving skills" because the subject \
matter itself overlaps. A general C++ project is NOT related evidence for \
"pandas and NumPy" - those are specific, named Python libraries, and general \
programming ability in a different language says nothing about them, the \
same way it would say nothing about a specific certification or employer. \
When a requirement names a specific library, tool or domain, only evidence \
that actually touches that library, tool or domain - or something \
functionally close to it - qualifies as related; evidence of unrelated \
general skill does not, no matter how strong that unrelated skill is.
- missing: no reliable evidence in the CV, of any kind. This includes the \
common case where the CV has real evidence of a genuinely *different* skill \
that does not bear on this specific requirement. If your own reason would \
have to read "this shows X, but the requirement asks about Y" with X and Y \
not actually the same or adjacent topic, the level is missing - not partial \
or related. A reason that admits the evidence does not establish what the \
requirement asks for is describing a missing requirement, whatever level you \
were about to pick.

Rules that apply to every level except missing:
- evidence_index: the number of the single best piece of CV evidence behind \
your judgement. Required whenever match_level is not "missing" - null only \
for missing. Never invent evidence or point at a number you were not given.
- reason: one short sentence, for the candidate to read, naming what the \
evidence actually shows and why that is the level you chose (e.g. "the CV \
describes only basic Python knowledge and an online course, not the strong \
proficiency the role asks for"). Describe only what evidence_index supports -
this is shown to a human, not used to generate other text, but it must stay \
as honest as the judgement itself.

Never give the benefit of the doubt: a requirement's own proficiency wording \
("strong", "basic", "familiarity with", "experience with") is part of what \
you are judging, not decoration to ignore. Do not infer a skill from a \
field of study, a job title, or what would be plausible for someone like \
this candidate - only from evidence that is actually written down.

Also return recommended_emphasis: the numbers of the 2-5 pieces of evidence \
most worth foregrounding for this role, strongest first.

Only use the numbers you were given. Never invent a requirement, an experience, \
a skill, a technology or a duration. Judge only what is written.

Reply with a single JSON object and nothing else:
{"judgements": [{"requirement_index": int,
                 "match_level": "match"|"partial"|"related"|"missing",
                 "evidence_index": int|null, "reason": str}],
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
            # A gap is a requirement with *no* evidence at all - must-haves
            # first so the biggest problems read first (US-3). A partial or
            # related match is real evidence at a different strength, not a
            # gap: it belongs in the report as what it is, not hidden here as
            # something the writer must avoid mentioning entirely.
            gaps=[
                requirement.text
                for requirement, strength in sorted(
                    judged, key=lambda pair: not pair[0].must_have
                )
                if strength == 0
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
    ) -> tuple[list[RequirementMatch], list[tuple[RoleRequirement, float]]]:
        """Turn the model's judgements into `RequirementMatch`es, enforcing the
        evidence rules deterministically.

        The requirement list drives the loop, not the model's reply, which
        means: a requirement the model forgot still appears (as missing), a
        requirement it invented is discarded, and any non-missing claim
        without usable evidence from `cv.evidence` is downgraded to missing -
        this now applies to a partial or related claim exactly as it always
        applied to a full match, so a model cannot earn score by citing
        evidence that is not real regardless of which level it claims. Every
        rejection is traced rather than silently applied.
        """
        by_index = {j.requirement_index: j for j in judgement.judgements}
        rejected: list[str] = []

        matches: list[RequirementMatch] = []
        judged: list[tuple[RoleRequirement, float]] = []

        for index, requirement in enumerate(job.requirements):
            claim = by_index.get(index)
            evidence = self._evidence_at(cv, claim.evidence_index) if claim else None
            claimed_level = claim.match_level if claim else MatchLevel.missing
            unsupported = claimed_level != MatchLevel.missing and evidence is None
            level = MatchLevel.missing if unsupported else claimed_level

            if unsupported:
                rejected.append(f"unsupported match: {requirement.text}")

            matches.append(
                RequirementMatch(
                    requirement=requirement.text,
                    evidence=evidence.text if evidence and not unsupported else "",
                    match_level=level,
                    met=level == MatchLevel.match,
                    reason=(
                        "no verifiable evidence was found"
                        if unsupported
                        else (claim.reason if claim else "")
                    ),
                )
            )
            judged.append((requirement, MATCH_LEVEL_WEIGHT[level]))

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

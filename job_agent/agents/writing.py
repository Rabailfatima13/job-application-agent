"""Writing agent: tailored CV + cover letter (FR-5, FR-6, US-4, US-5).

Consumes what Week 6 already produced - the parsed CV, the parsed posting, the
company brief and the fit report - and re-presents the candidate's *existing*
material for this role. It may reorder, re-emphasise, combine and reword. It
may not invent.

    build prompt -> heavy model -> WritingDraft -> schema validation
                                                       |
                                        (bounded retry on malformed JSON)
                                                       v
                                              grounding validation
                                                       |
                            clean -> TailoredCV + CoverLetter        (accept)
                          not clean -> retry, telling the model exactly which
                                       claims were unsupported and why
                          out of attempts -> RetryExhaustedError     (never
                                       return material we cannot vouch for)

The two validation layers are deliberately separate and both must pass: schema
validation cannot detect a fabrication, and grounding cannot detect a malformed
response. See validation/grounding.py for why.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

from ..memory.session import RunContext
from ..models import (
    CompanyBrief,
    CoverLetter,
    FitReport,
    JobDescription,
    ParsedCV,
    TailoredCV,
)
from ..observability import traced_tool_call
from ..validation import (
    EntailmentVerdicts,
    GroundingResult,
    RetryExhaustedError,
    apply_verdicts,
    check_grounding,
    generate_validated,
    split_sentences,
)
from ..validation.grounding import (
    ENTAILMENT_SYSTEM_PROMPT,
    build_company_context,
    build_entailment_prompt,
)
from .base import BaseAgent

WRITING_STEP = "tailor_cv"
VERIFY_STEP = "verify_claims"
WRITING_MAX_TOKENS = 2048
VERIFY_MAX_TOKENS = 1024

# How many times the writer may be sent back to fix unsupported claims before
# the run is abandoned. Bounded for the same reason the schema guard is.
DEFAULT_GROUNDING_ATTEMPTS = 3


# A letter's opening and closing lines are format, not claims about the
# candidate: "Dear Hiring Team," asserts nothing that could be true or false of
# them. A live run wasted an attempt when "Hiring" was read as an unsupported
# proper noun. These patterns are deliberately tight - a single short line that
# is nothing but a salutation or a valediction - so nothing factual can hide
# behind them. The signature line itself is NOT exempt: it still has to name
# someone the CV names, which is what keeps "[Candidate]" rejected.
_SALUTATION = re.compile(r"^dear\b[^.!?]{0,60}[,:]$", re.IGNORECASE)
_SALUTATION_PREFIX = re.compile(r"^\s*dear\b[^.!?]{0,60}?[,:]\s*", re.IGNORECASE)
_VALEDICTION = re.compile(
    r"^(sincerely|regards|best regards|kind regards|best wishes|"
    r"yours (sincerely|faithfully|truly))\s*[,.]?$",
    re.IGNORECASE,
)


def is_letter_boilerplate(line: str) -> bool:
    """True for a standalone salutation or valediction line."""
    stripped = line.strip()
    return bool(_SALUTATION.match(stripped) or _VALEDICTION.match(stripped))


def strip_salutation(claim: str) -> str:
    """Remove a leading "Dear ...," from a claim, keeping the rest intact.

    Asking the model to put the salutation on its own line is not enough: a
    live run showed it writing "Dear Hiring Team, I am excited to apply..." as
    one sentence two attempts out of three, and the greeting then read as an
    unsupported proper noun. Only the greeting itself is removed - whatever
    follows it is still checked exactly as strictly as before.
    """
    return _SALUTATION_PREFIX.sub("", claim, count=1).strip()


def letter_claims(body: str) -> list[str]:
    """The checkable claims in a cover letter.

    Split by line first, then by sentence, so a salutation on its own line is
    its own unit and can be recognised as boilerplate instead of being glued to
    the first real sentence. Everything that is not boilerplate is checked
    exactly as strictly as before.
    """
    claims: list[str] = []
    for line in body.splitlines():
        if not line.strip() or is_letter_boilerplate(line):
            continue
        for sentence in split_sentences(line):
            # Handles the salutation written inline with the first sentence,
            # which is what models actually do.
            remainder = strip_salutation(sentence)
            if remainder:
                claims.append(remainder)
    return claims


class WritingDraft(BaseModel):
    """What the model returns: re-presented CV bullets plus a letter body."""

    model_config = ConfigDict(extra="forbid")

    bullets: list[str] = Field(default_factory=list)
    omitted: list[str] = Field(
        default_factory=list,
        description="Real CV content de-prioritised for this role.",
    )
    cover_letter: str = ""


WRITING_SYSTEM_PROMPT = """\
You tailor an existing CV to one job and draft a cover letter. Everything you \
write must already be true of this candidate according to their CV EVIDENCE. \
The result must read as the SAME candidate's CV, adapted for this one job - \
not a new CV. You are choosing what to emphasise and how to phrase it, never \
what is true: make small, targeted changes to an existing document, not a \
rewrite from scratch.

Read WHAT THE ROLE ASKS FOR below and check it against every line in CV \
EVIDENCE, not only what WORTH FOREGROUNDING lists (that list is capped and \
may leave out a real match). When a requirement is already supported \
somewhere in the evidence but is not prominent, bring that evidence forward \
and, if it helps make the match obvious, reword it slightly to use the \
posting's own term for the same real thing - the underlying fact does not \
change, only its position and phrasing do. A requirement with no supporting \
evidence anywhere stays unmentioned; do not paper over it with related-\
sounding wording.

You MAY:
- reorder and re-prioritise the evidence so the most relevant comes first
- reword for clarity and to echo the posting's language
- combine two pieces of evidence into one line when both are real
- leave weaker evidence out (record it under omitted)

You MUST NOT introduce anything the evidence does not state: a job, employer, \
project, technology, tool, achievement, metric, number, duration, seniority, \
qualification, certification or degree. Do not add a technology because the \
posting asks for it. Do not estimate years of experience. Do not invent \
figures. If the candidate lacks something the role wants, say nothing about \
it - the gaps are handled elsewhere.

The evidence's own proficiency level is part of what is true, not something \
to round up: if it says "basic knowledge", "an online course" or "currently \
learning", say that - not "strong", "proficient" or "experienced". The \
posting's wording is not a license to describe the candidate at the level it \
asks for; only the evidence sets the level.

bullets: 4-8 tailored CV lines, strongest first.
cover_letter: 3-4 short paragraphs addressed to the company, drawing only on \
the evidence, plus the company facts supplied. Do not claim knowledge of the \
company beyond those facts.

Lay the letter out like this, each on its own line:
    Dear Hiring Team,
    <paragraphs>
    Sincerely,
    <the candidate's name>

You may refer to the company using the COMPANY FACTS supplied - "your work on \
X", "the company's Y" - and that is encouraged, it is what makes the letter \
specific. But a fact about the company is never a fact about the candidate: if \
the company works with a technology the CV does not mention, you may say the \
company works with it and you may say it interests you, and you may NOT say \
the candidate has used it, knows it, or has skills in it.

Put a company fact and a statement about the candidate in separate sentences, \
never combined into one. Write "The company partners with major retailers. I \
would welcome the chance to contribute to that work." - not "Your work with \
major retailers aligns with my ambition to contribute." This applies to facts \
from COMPANY FACTS specifically (a client, a technology, a location, a \
metric) - simply naming the company itself, as the employer you are writing \
to, is always fine in either kind of sentence.

Use exactly "Dear Hiring Team," as the salutation - do not address a named \
person, a department or the company. Never write a template placeholder such \
as [Candidate], [Your Name], [Date] or [Hiring Manager]. If no candidate name \
is given, end the letter without a signature line.

Reply with a single JSON object and nothing else:
{"bullets": [str], "omitted": [str], "cover_letter": str}"""


class WritingAgent(BaseAgent):
    """Produces a `TailoredCV` and a `CoverLetter` that are both schema-valid
    and grounded in the candidate's real CV."""

    name = "writing"

    def __init__(
        self,
        *args,
        grounding_attempts: int = DEFAULT_GROUNDING_ATTEMPTS,
        verify_with_model: bool = False,
        **kwargs,
    ) -> None:
        """`verify_with_model` adds the optional entailment pass (an extra
        model call per attempt). Off by default: the deterministic check
        catches every fabrication category the proposal lists, and this is the
        expensive belt-and-braces layer."""
        super().__init__(*args, **kwargs)
        self.grounding_attempts = grounding_attempts
        self.verify_with_model = verify_with_model

    def run(self, context: RunContext) -> tuple[TailoredCV, CoverLetter]:
        if context.parsed_cv is None or context.job is None:
            raise ValueError(
                "WritingAgent needs a parsed CV and job description on the "
                "RunContext; run the parsers first."
            )
        if context.fit_report is None:
            raise ValueError(
                "WritingAgent needs a fit report on the RunContext; run the "
                "scoring agent first."
            )
        return self.write(
            context.parsed_cv, context.job, context.fit_report, context.brief
        )

    def write(
        self,
        cv: ParsedCV,
        job: JobDescription,
        report: FitReport,
        brief: CompanyBrief | None = None,
    ) -> tuple[TailoredCV, CoverLetter]:
        """Write, validate, and re-write until the material is grounded."""
        grounding_feedback = ""
        last_result: GroundingResult | None = None

        for _ in range(self.grounding_attempts):
            draft = self._draft(cv, job, report, brief, grounding_feedback)
            last_result = self._ground(draft, cv, job, brief)
            if last_result.is_clean:
                return self._build_outputs(draft, job, cv)
            grounding_feedback = last_result.feedback()

        # Nothing unverified is ever returned: the proposal's fallback (the
        # original CV plus the gap list) is the caller's decision, not ours.
        raise RetryExhaustedError(
            "WritingDraft",
            self.grounding_attempts,
            f"unsupported claims remained: {last_result.feedback()}",
        )

    # --- generation ---------------------------------------------------------

    def _draft(
        self,
        cv: ParsedCV,
        job: JobDescription,
        report: FitReport,
        brief: CompanyBrief | None,
        grounding_feedback: str,
    ) -> WritingDraft:
        prompt = build_writing_prompt(cv, job, report, brief)
        if grounding_feedback:
            prompt = (
                f"{prompt}\n\n---\n"
                "Your previous draft was rejected because these claims are not "
                "supported by the CV evidence:\n"
                f"{grounding_feedback}\n"
                "Rewrite without them. Do not restate them in different words - "
                "remove the unsupported facts entirely."
            )

        def produce(schema_feedback: str | None) -> str:
            content = prompt
            if schema_feedback:
                content = (
                    f"{prompt}\n\n---\n"
                    f"Your previous reply was rejected by the schema validator:\n"
                    f"{schema_feedback}\n"
                    "Return corrected JSON only, following the schema exactly."
                )
            result = self.call_model(
                WRITING_STEP,
                [{"role": "user", "content": content}],
                system=WRITING_SYSTEM_PROMPT,
                max_tokens=WRITING_MAX_TOKENS,
            )
            return result.text

        return generate_validated(
            produce, WritingDraft, collector=self.collector, agent=self.name
        )

    # --- validation ---------------------------------------------------------

    def _ground(
        self,
        draft: WritingDraft,
        cv: ParsedCV,
        job: JobDescription,
        brief: CompanyBrief | None = None,
    ) -> GroundingResult:
        """Check every generated line, in two separate evidence domains.

        The **tailored CV** is checked against the candidate's CV and nothing
        else - `company_context` is deliberately not passed, so a company fact
        can never become a CV bullet.

        The **cover letter** is checked sentence by sentence (so one bad
        sentence is reported precisely) and may additionally cite the verified
        company brief - but only for statements that are about the company.
        `is_candidate_claim` decides which, and it defaults to the strict
        candidate domain whenever the subject is not clearly the employer.
        """
        cv_result = check_grounding(
            draft.bullets, cv, context_terms=(job.company, job.role)
        )
        letter_result = check_grounding(
            letter_claims(draft.cover_letter),
            cv,
            context_terms=(job.company, job.role),
            company_context=build_company_context(brief),
            company_name=job.company,
        )
        result = GroundingResult(
            grounded=[*cv_result.grounded, *letter_result.grounded],
            issues=[*cv_result.issues, *letter_result.issues],
        )

        if self.verify_with_model and result.grounded:
            result = apply_verdicts(result, self._verify(result.grounded, cv))

        if not result.is_clean and self.collector is not None:
            with traced_tool_call(
                self.collector,
                agent=self.name,
                name="writing:grounding_guard",
                arguments={
                    "checked": len(result.checked_claims),
                    "unsupported": len(result.issues),
                },
                # A rejected draft is a failure of the attempt, and the trace
                # should say so - it is why a retry happened.
            ) as outcome:
                outcome["result"] = result.feedback()

        return result

    def _verify(self, claims: list[str], cv: ParsedCV) -> EntailmentVerdicts:
        """Optional model-assisted entailment over the claims stage 1 allowed.

        The call goes through the router like every other, so the provider is
        never named here and the call is traced with its own token cost.
        """
        prompt = build_entailment_prompt(claims, cv)

        def produce(feedback: str | None) -> str:
            content = prompt if not feedback else f"{prompt}\n\n---\n{feedback}"
            result = self.call_model(
                VERIFY_STEP,
                [{"role": "user", "content": content}],
                system=ENTAILMENT_SYSTEM_PROMPT,
                max_tokens=VERIFY_MAX_TOKENS,
            )
            return result.text

        return generate_validated(
            produce, EntailmentVerdicts, collector=self.collector, agent=self.name
        )

    # --- assembly -----------------------------------------------------------

    @staticmethod
    def _build_outputs(
        draft: WritingDraft, job: JobDescription, cv: ParsedCV
    ) -> tuple[TailoredCV, CoverLetter]:
        """Role and company come from the parsed posting, never from the model."""
        return (
            TailoredCV(
                role=job.role,
                company=job.company,
                bullets=list(draft.bullets),
                omitted=list(draft.omitted),
                full_text=assemble_full_cv(cv, job, draft.bullets),
            ),
            CoverLetter(role=job.role, company=job.company, body=draft.cover_letter),
        )


def assemble_full_cv(cv: ParsedCV, job: JobDescription, bullets: list[str]) -> str:
    """The complete tailored CV - the candidate's own baseline CV, tailored
    for this role, not a short highlight reel and not a second document.

    `bullets` are already grounded (they come from a `WritingDraft` that
    passed `_ground`, the same guarantee `TailoredCV` itself relies on) -
    this makes no model call and adds no new claim, it only re-presents
    content already verified true of the candidate: their own name, the
    real CV lines re-ordered and re-emphasised for this role, and their
    real skills.

    Deliberately excludes what the writer de-prioritised (`WritingDraft.
    omitted`, still shown separately on the Results page as "De-
    prioritised") - a *tailored* CV is exactly the candidate's baseline CV
    with the less relevant material left out for this one application, not
    everything stitched back in under a second heading. That is also what
    keeps this to roughly one page: bullets are already bounded to 4-8
    lines by the writer's own prompt, so a name, a role/company line, those
    bullets and a skills line comfortably fit on one.
    """
    sections: list[str] = []
    if cv.candidate_name:
        sections.append(cv.candidate_name)
    sections.append(f"Tailored for: {job.role} at {job.company}")
    sections.append("\n".join(f"- {bullet}" for bullet in bullets))
    if cv.skills:
        sections.append("Skills: " + ", ".join(cv.skills))
    return "\n\n".join(sections)


def build_writing_prompt(
    cv: ParsedCV,
    job: JobDescription,
    report: FitReport,
    brief: CompanyBrief | None = None,
) -> str:
    """Everything the writer is allowed to draw on, and nothing else.

    The scoring agent's `recommended_emphasis` is passed through as guidance -
    that is what it was produced for - and the gaps are named explicitly so the
    model knows what *not* to write about. The candidate's raw CV text is not
    included: the parsed evidence is the authoritative list, and sending the
    raw text invites the model to mine it for half-read details.

    The full evidence list is included even though `recommended_emphasis` is
    already there: scoring caps that list at 2-5 items (see
    `scoring.py`'s prompt), so a requirement can be genuinely satisfied by
    evidence that list left out. The writer needs the complete list to be
    able to bring such a match forward at all - see the system prompt's own
    instruction to check every requirement against every evidence line, not
    only what is flagged here.
    """
    evidence = "\n".join(f"- {e.text}" for e in cv.evidence)
    requirements = "\n".join(
        f"- {'must have' if r.must_have else 'nice to have'}: {r.text}"
        for r in job.requirements
    )
    emphasis = "\n".join(f"- {item}" for item in report.recommended_emphasis)
    company_facts = (
        "\n".join(f"- {fact}" for fact in brief.facts)
        if brief and brief.facts
        else "- (no company research available)"
    )
    return (
        f"CANDIDATE NAME: {cv.candidate_name or '(not given - do not invent one)'}\n"
        f"ROLE: {job.role}\n"
        f"COMPANY: {job.company}\n\n"
        f"WHAT THE ROLE ASKS FOR:\n{requirements}\n\n"
        f"CV EVIDENCE (the only facts you may use about the candidate):\n{evidence}\n\n"
        f"WORTH FOREGROUNDING (from the fit analysis - not exhaustive; a "
        "requirement may also be satisfied by evidence not listed here):\n"
        f"{emphasis}\n\n"
        f"KNOWN GAPS - do not claim these:\n"
        f"{chr(10).join(f'- {gap}' for gap in report.gaps) or '- (none)'}\n\n"
        f"COMPANY FACTS (for the letter only):\n{company_facts}"
    )

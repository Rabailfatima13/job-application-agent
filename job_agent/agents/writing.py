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
from ..validation.grounding import ENTAILMENT_SYSTEM_PROMPT, build_entailment_prompt
from .base import BaseAgent

WRITING_STEP = "tailor_cv"
VERIFY_STEP = "verify_claims"
WRITING_MAX_TOKENS = 2048
VERIFY_MAX_TOKENS = 1024

# How many times the writer may be sent back to fix unsupported claims before
# the run is abandoned. Bounded for the same reason the schema guard is.
DEFAULT_GROUNDING_ATTEMPTS = 3


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
write must already be true of this candidate according to their CV EVIDENCE.

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

bullets: 4-8 tailored CV lines, strongest first.
cover_letter: 3-4 short paragraphs addressed to the company, drawing only on \
the evidence, plus the company facts supplied. Do not claim knowledge of the \
company beyond those facts.

Never write a template placeholder such as [Candidate], [Your Name], [Date] or \
[Hiring Manager]. Sign off with the candidate's name exactly as it is given, \
and if no name is given, end the letter without a signature line.

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
            last_result = self._ground(draft, cv, job)
            if last_result.is_clean:
                return self._build_outputs(draft, job)
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
        self, draft: WritingDraft, cv: ParsedCV, job: JobDescription
    ) -> GroundingResult:
        """Check every generated line against the candidate's real CV.

        The cover letter is checked sentence by sentence, so one bad sentence
        is reported precisely rather than condemning the whole letter. The
        employer's own proper nouns are allowed as context - a cover letter has
        to be able to name the company it is addressed to.
        """
        claims = [*draft.bullets, *split_sentences(draft.cover_letter)]
        result = check_grounding(claims, cv, context_terms=(job.company, job.role))

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
        draft: WritingDraft, job: JobDescription
    ) -> tuple[TailoredCV, CoverLetter]:
        """Role and company come from the parsed posting, never from the model."""
        return (
            TailoredCV(
                role=job.role,
                company=job.company,
                bullets=list(draft.bullets),
                omitted=list(draft.omitted),
            ),
            CoverLetter(role=job.role, company=job.company, body=draft.cover_letter),
        )


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
        f"WORTH FOREGROUNDING (from the fit analysis):\n{emphasis}\n\n"
        f"KNOWN GAPS - do not claim these:\n"
        f"{chr(10).join(f'- {gap}' for gap in report.gaps) or '- (none)'}\n\n"
        f"COMPANY FACTS (for the letter only):\n{company_facts}"
    )

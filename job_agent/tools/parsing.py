"""`parse_jd` and `parse_cv` - the input layer (FR-1, FR-3).

Both tools follow the same shape:

    source -> document text -> light-tier model -> JSON -> Pydantic validation
                                                     ^          |
                                                     +-- bounded retry --+

Three rules the implementation is built around:

1. **The raw text never round-trips through the model.** The model is asked
   only for the extracted fields; `raw_text` is spliced in from the source
   afterwards. That makes verbatim preservation a property of the code rather
   than something we hope the model respected - and Week 7's no-fabrication
   check compares against exactly that text.
2. **Extraction, not summarisation.** The prompts require claims to be copied
   from the source. For the CV, a deterministic guard then drops any evidence
   line that is not actually present in the source text, so a model that
   paraphrases cannot quietly rewrite the candidate's experience.
3. **No provider is named here.** The tools receive a `ModelRouter` and ask it
   for the step; the router decides that `parse_cv`/`parse_jd` run on the
   cheap tier (see llm/router.py).
"""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ..llm import ModelRouter
from ..models import CVEvidence, JobDescription, ParsedCV, RoleRequirement
from ..observability import TraceCollector, traced_tool_call
from ..validation import generate_validated
from . import Tool
from .document_io import load_document_text

# Used when a posting genuinely does not name the employer. An explicit
# placeholder, never a guessed company name.
UNKNOWN_COMPANY = "Unknown"

PARSING_MAX_TOKENS = 2048


# --- What the model is asked to return --------------------------------------
# Deliberately *not* JobDescription/ParsedCV: those carry `raw_text`, which the
# model must never be asked to reproduce. These drafts are validated first, then
# combined with the real source text.


class JDExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str
    company: str | None = None
    location: str | None = None
    requirements: list[RoleRequirement] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


class CVEvidenceExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    section: str | None = None
    skills: list[str] = Field(default_factory=list)


class CVExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_name: str | None = None
    evidence: list[CVEvidenceExtraction] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)


# --- Prompts ----------------------------------------------------------------

JD_SYSTEM_PROMPT = """\
You extract structured data from job postings. You are an extractor, not a \
writer.

Rules:
- Use only what the posting says. Never infer or invent a requirement, skill, \
company or location that is not written there.
- Copy requirement wording from the posting; do not rephrase it into your own \
words.
- A requirement is something the candidate must have or should have. A \
responsibility is something the job involves doing. Keep them separate.
- Mark a requirement must_have=false when the posting lists it under \
"nice to have", "preferred", "bonus" or similar; otherwise must_have=true.
- skills lists concrete technologies, tools and languages named in the posting.
- Use null for company or location if the posting does not state them.

Reply with a single JSON object and nothing else:
{"role": str, "company": str|null, "location": str|null,
 "requirements": [{"text": str, "must_have": bool}],
 "responsibilities": [str], "skills": [str]}"""

CV_SYSTEM_PROMPT = """\
You extract structured data from a CV. You are an extractor, not a writer, and \
the candidate's own wording is the point.

Rules:
- Copy each evidence line VERBATIM from the CV. Do not reword, summarise, \
merge, embellish or "improve" it. Strip only leading bullet characters.
- Never add a job, project, skill, achievement, technology, employer, date or \
number of years that is not written in the CV.
- One evidence item per distinct claim (a bullet, a project, a degree, an \
achievement).
- section is the CV heading the claim came from, e.g. "Experience", \
"Projects", "Education", "Skills", "Achievements". Use null if unclear.
- skills on an evidence item are the technologies named in that line. The \
top-level skills list is the candidate's overall skills as stated in the CV.
- candidate_name is the name as written, or null if the CV does not give one.

Reply with a single JSON object and nothing else:
{"candidate_name": str|null,
 "evidence": [{"text": str, "section": str|null, "skills": [str]}],
 "skills": [str]}"""


# --- Verbatim guard ---------------------------------------------------------

_BULLET_PREFIX = re.compile(r"^[\s\-•\*·–—]+")


def _normalise(text: str) -> str:
    """Collapse the differences that do not change what a line claims:
    bullet markers, casing, and whitespace."""
    return re.sub(r"\s+", " ", _BULLET_PREFIX.sub("", text)).strip().lower()


def is_supported_by(claim: str, source_text: str) -> bool:
    """True when `claim` actually appears in `source_text`.

    Cheap and deterministic on purpose - this is the parsing-time guarantee
    that evidence is the candidate's own wording. The richer semantic check on
    *generated* text is a separate Week 7 concern (validation/grounding.py).
    """
    normalised_claim = _normalise(claim)
    return bool(normalised_claim) and normalised_claim in _normalise(source_text)


# --- Shared extraction runner -----------------------------------------------


def _extract(
    step: str,
    schema: type[BaseModel],
    system_prompt: str,
    document_text: str,
    router: ModelRouter,
    collector: TraceCollector | None,
    max_attempts: int,
):
    """Run one light-tier extraction and return the validated draft.

    Retries are handled by `generate_validated`; the feedback it supplies is
    folded into the same user message rather than appended as a second one,
    because providers require alternating roles.
    """
    client = router.for_step(step)
    tier = router.tier_for(step)

    def produce(feedback: str | None) -> str:
        content = document_text
        if feedback:
            content = (
                f"{document_text}\n\n---\n"
                f"Your previous reply was rejected by the schema validator:\n"
                f"{feedback}\n"
                "Return corrected JSON only, following the schema exactly."
            )
        with traced_tool_call(
            collector,
            agent=step,
            name=f"model:{tier}:{step}",
            arguments={"model": client.name, "retry": bool(feedback)},
            kind="model",
        ) as outcome:
            result = client.complete(
                [{"role": "user", "content": content}],
                system=system_prompt,
                max_tokens=PARSING_MAX_TOKENS,
            )
            outcome["result"] = result.text
            outcome["input_tokens"] = result.input_tokens
            outcome["output_tokens"] = result.output_tokens
        return result.text

    return generate_validated(
        produce,
        schema,
        max_attempts=max_attempts,
        collector=collector,
        agent=step,
    )


# --- The two tools -----------------------------------------------------------


def parse_jd(
    source: str | Path,
    router: ModelRouter,
    collector: TraceCollector | None = None,
    max_attempts: int = 3,
) -> JobDescription:
    """Parse a job posting into a validated `JobDescription`.

    `source` is the posting text or a path to a .txt/.md/.pdf file. A posting
    that does not name the employer gets `UNKNOWN_COMPANY` rather than a
    guess.
    """
    text = load_document_text(source)
    draft = _extract(
        "parse_jd", JDExtraction, JD_SYSTEM_PROMPT, text, router, collector, max_attempts
    )
    return JobDescription(
        role=draft.role,
        company=draft.company or UNKNOWN_COMPANY,
        raw_text=text,
        requirements=[
            RoleRequirement(text=r.text, must_have=r.must_have)
            for r in draft.requirements
        ],
        responsibilities=list(draft.responsibilities),
        skills=list(draft.skills),
        location=draft.location,
    )


def parse_cv(
    source: str | Path,
    router: ModelRouter,
    collector: TraceCollector | None = None,
    max_attempts: int = 3,
) -> ParsedCV:
    """Parse a CV into a validated `ParsedCV`, keeping the candidate's wording.

    Evidence the model did not copy from the CV is dropped here rather than
    carried forward: unverbatim evidence would silently become the reference
    that Week 7's no-fabrication check trusts. Dropped lines are recorded in
    the trace so the loss is visible, not silent.
    """
    text = load_document_text(source)
    draft = _extract(
        "parse_cv", CVExtraction, CV_SYSTEM_PROMPT, text, router, collector, max_attempts
    )

    kept, dropped = [], []
    for item in draft.evidence:
        target = kept if is_supported_by(item.text, text) else dropped
        target.append(item)

    if dropped and collector is not None:
        with traced_tool_call(
            collector,
            agent="parse_cv",
            name="parse_cv:verbatim_guard",
            arguments={"dropped": len(dropped), "kept": len(kept)},
        ) as outcome:
            outcome["result"] = "; ".join(item.text for item in dropped)

    return ParsedCV(
        candidate_name=draft.candidate_name,
        raw_text=text,
        evidence=[
            CVEvidence(text=item.text, section=item.section, skills=list(item.skills))
            for item in kept
        ],
        skills=list(draft.skills),
    )


# --- Tool definitions --------------------------------------------------------

_SOURCE_SCHEMA = {
    "type": "object",
    "properties": {
        "source": {
            "type": "string",
            "description": "The document text, or a path to a .txt/.md/.pdf file.",
        }
    },
    "required": ["source"],
}


def build_parsing_tools(
    router: ModelRouter, collector: TraceCollector | None = None
) -> list[Tool]:
    """The parsing tools, bound to a router and trace collector.

    A factory rather than module-level constants because a tool needs a router
    to exist, and each run may want its own collector - the same reason
    ToolRegistry is an instance.
    """
    return [
        Tool(
            name="parse_jd",
            description=(
                "Extract role, company, requirements, responsibilities and skills "
                "from a job description."
            ),
            input_schema=_SOURCE_SCHEMA,
            run=lambda source: parse_jd(source, router, collector),
        ),
        Tool(
            name="parse_cv",
            description=(
                "Extract the candidate's skills and verbatim evidence lines from a CV."
            ),
            input_schema=_SOURCE_SCHEMA,
            run=lambda source: parse_cv(source, router, collector),
        ),
    ]

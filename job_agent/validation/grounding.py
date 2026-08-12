"""No-fabrication grounding check - the project's core trust requirement
(FR-10, NFR-4).

Deliberately a separate module from schema_guard: a perfectly schema-valid
tailored CV can still be a fabrication, so this check runs *after* that one and
can never be satisfied by it. Schema validation answers "is this the right
shape?"; grounding answers "is this true of *this* candidate?".

The rule: every generated line must be supported by the candidate's own CV.
Re-ordering, re-emphasis, combining two real bullets, and rewording that
preserves the claim are all allowed - a generated line is *not* required to
appear in the CV verbatim, which is why substring matching is not what happens
here. What is forbidden is introducing a fact the CV never stated.

Two stages, cheap first:

1. `check_grounding` - deterministic, no model call. Catches the fabrications
   that carry a fingerprint: a technology, employer, institution, product or
   certification the CV never names (a distinctive term), or a number the CV
   never stated (years of experience, team sizes, percentage improvements).
   This alone covers every category the proposal lists.

2. `build_entailment_prompt` / `apply_verdicts` - an optional model-assisted
   pass over the claims stage 1 accepted, for fabrications with no distinctive
   fingerprint ("led the migration", "mentored juniors"). This module builds
   the prompt and merges the verdicts; the actual call is made by the caller
   through the existing `ModelClient`/router plumbing, so no provider SDK is
   named here and the call is traced like every other.
"""

import re

from pydantic import BaseModel, ConfigDict, Field

from ..models import ParsedCV

# Words that are capitalised often enough in ordinary prose that treating them
# as a candidate-specific claim would produce noise rather than protection.
_COMMON_WORDS = {
    "a", "about", "across", "after", "all", "also", "an", "and", "any", "are",
    "as", "at", "be", "been", "both", "build", "building", "built", "but", "by",
    "can", "collaborated", "contributed", "created", "delivered", "designed",
    "developed", "during", "each", "engineered", "every", "experience", "for",
    "from", "had", "has", "have", "her", "his", "how", "i", "if", "implemented",
    "in", "including", "into", "introduced", "is", "it", "its", "led", "maintained",
    "me", "migrated", "my", "of", "on", "or", "our", "over", "own", "position",
    "role", "shipped", "since", "so", "team", "that", "the", "their", "them",
    "these", "they", "this", "those", "through", "to", "used", "using", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "with", "within", "without", "work", "worked", "working", "would", "years",
    "you", "your",
}

# Spelled-out numbers, so "five years of Python" is checked the same way "5
# years of Python" would be.
_NUMBER_WORDS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
    "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "fifteen": "15", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "hundred": "100", "thousand": "1000",
}

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-_]*")
_DIGITS = re.compile(r"\d+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


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
        """True when every checked claim was supported."""
        return not self.issues

    @property
    def passed(self) -> bool:
        return self.is_clean

    @property
    def checked_claims(self) -> list[str]:
        """Everything that was examined, supported or not."""
        return [*self.grounded, *(issue.claim for issue in self.issues)]

    def feedback(self) -> str:
        """The violations, phrased so a model can act on them in a retry."""
        return "\n".join(f"- {i.claim}\n  -> {i.reason}" for i in self.issues)


def split_sentences(text: str) -> list[str]:
    """Break generated prose into individually checkable claims."""
    return [part.strip() for part in _SENTENCE_SPLIT.split(text) if part.strip()]


def _normalise(text: str) -> str:
    replaced = text
    for word, digit in _NUMBER_WORDS.items():
        replaced = re.sub(rf"\b{word}\b", digit, replaced, flags=re.IGNORECASE)
    return replaced


def numbers_in(text: str) -> set[str]:
    """Every number a text asserts, with spelled-out numbers normalised."""
    return set(_DIGITS.findall(_normalise(text)))


def distinctive_terms(claim: str) -> list[str]:
    """Terms that would identify a specific technology, employer, product,
    institution or credential.

    A term counts as distinctive when it is capitalised mid-sentence, or
    contains a digit or a symbol that ordinary prose does not (`C#`, `Python3`,
    `Node.js`). The first word of the claim is skipped because sentences and
    bullets start with a capital regardless, and common English words are
    excluded so the check flags substance rather than grammar.
    """
    tokens = _WORD.findall(claim)
    terms: list[str] = []
    for index, token in enumerate(tokens):
        stripped = token.strip(".,;:")
        if not stripped or stripped.lower() in _COMMON_WORDS:
            continue
        looks_technical = any(c.isdigit() or c in "+#" for c in stripped)
        is_acronym = stripped.isupper() and len(stripped) > 1
        if index == 0 and not looks_technical and not is_acronym:
            # A sentence-initial capital carries no information ("Built ...") -
            # unless it is an acronym, where the capitalisation is the meaning
            # ("AWS Certified ...").
            continue
        if stripped[0].isupper() or looks_technical:
            terms.append(stripped)
    return terms


def check_grounding(
    claims: list[str],
    source_cv: ParsedCV,
    context_terms: tuple[str, ...] | list[str] = (),
) -> GroundingResult:
    """Split `claims` into those supported by `source_cv` and those that are not.

    The authoritative evidence is the candidate's own CV - `raw_text` plus the
    parsed evidence lines - and nothing else. A generated document cannot widen
    what counts as evidence, which is the whole point: a job posting mentioning
    Kubernetes must never make "experience with Kubernetes" a supported claim.

    `context_terms` is a deliberately narrow allowance for the *employer's*
    proper nouns - the company name and the role title - so a cover letter may
    address the company it is written to. It is never used for technologies or
    achievements.
    """
    corpus = "\n".join(
        [
            source_cv.raw_text,
            *(e.text for e in source_cv.evidence),
            # A skill the candidate listed is something they claimed, so it is
            # evidence too - even when no bullet happens to spell it out.
            *source_cv.skills,
            *(skill for e in source_cv.evidence for skill in e.skills),
        ]
    )
    corpus_lower = corpus.lower()
    corpus_numbers = numbers_in(corpus)
    allowed = {
        term.lower()
        for phrase in context_terms
        for term in _WORD.findall(phrase)
    }

    result = GroundingResult()
    for claim in claims:
        unsupported_terms = [
            term
            for term in distinctive_terms(claim)
            if term.lower() not in corpus_lower and term.lower() not in allowed
        ]
        unsupported_numbers = sorted(numbers_in(claim) - corpus_numbers)

        if unsupported_terms:
            result.issues.append(
                GroundingIssue(
                    claim=claim,
                    reason=(
                        "the CV never mentions "
                        + ", ".join(dict.fromkeys(unsupported_terms))
                    ),
                )
            )
        elif unsupported_numbers:
            result.issues.append(
                GroundingIssue(
                    claim=claim,
                    reason=(
                        "the CV never states the figure(s) "
                        + ", ".join(unsupported_numbers)
                    ),
                )
            )
        else:
            result.grounded.append(claim)
    return result


# --- Optional second stage: model-assisted entailment ------------------------


class ClaimVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_index: int
    supported: bool
    reason: str = ""


class EntailmentVerdicts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


ENTAILMENT_SYSTEM_PROMPT = """\
You check whether claims about a candidate are supported by their CV. You are \
a fact-checker, not an editor, and you never rewrite anything.

For each numbered CLAIM, decide whether the CV EVIDENCE supports it.

- supported = true when the evidence states it, or states something the claim \
merely rewords, reorders, summarises or combines.
- supported = false when the claim adds anything the evidence does not state: \
a job, project, employer, technology, achievement, metric, seniority, duration, \
qualification or responsibility. Being plausible for the role is not support.
- When unsure, answer false and say what is missing.

Reply with a single JSON object and nothing else:
{"verdicts": [{"claim_index": int, "supported": bool, "reason": str}]}"""


def build_entailment_prompt(claims: list[str], source_cv: ParsedCV) -> str:
    """The fact-checking prompt: the CV's own evidence, then the claims."""
    evidence = "\n".join(f"- {e.text}" for e in source_cv.evidence) or source_cv.raw_text
    numbered = "\n".join(f"[{i}] {claim}" for i, claim in enumerate(claims))
    return f"CV EVIDENCE:\n{evidence}\n\nCLAIMS:\n{numbered}"


def apply_verdicts(
    result: GroundingResult, verdicts: EntailmentVerdicts
) -> GroundingResult:
    """Fold model verdicts into a deterministic result.

    Only demotes: the model can reject a claim the deterministic pass allowed,
    never rehabilitate one it rejected. A missing or out-of-range verdict leaves
    its claim as the deterministic pass judged it.
    """
    rejected = {
        v.claim_index: v.reason or "not supported by the CV"
        for v in verdicts.verdicts
        if not v.supported and 0 <= v.claim_index < len(result.grounded)
    }
    if not rejected:
        return result

    return GroundingResult(
        grounded=[c for i, c in enumerate(result.grounded) if i not in rejected],
        issues=[
            *result.issues,
            *(
                GroundingIssue(claim=result.grounded[i], reason=reason)
                for i, reason in sorted(rejected.items())
            ),
        ],
    )

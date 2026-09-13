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
_BRACKETED = re.compile(r"\[([^\]]{1,60})\]")

# Signals used to decide whose claim a sentence is making. Kept narrow and
# readable rather than clever - this is the rule an evaluator will ask about.
_FIRST_PERSON = re.compile(r"\b(i|i'm|i've|my|mine|me)\b", re.IGNORECASE)
_EXPERIENCE = re.compile(
    r"\b(experience[ds]?|expertise|skill(s|ed)?|proficien\w+|familiar(ity)?|"
    r"background|knowledge|worked|working|built|building|develop(ed|ing)|"
    r"deliver(ed|ing)|implement(ed|ing)|creat(ed|ing)|design(ed|ing)|"
    r"achiev(ed|ement|ements)|improv(ed|ement|ements)|led|manag(ed|ing)|"
    r"mentor(ed|ing)|manag\w*|year|years|certified|certification[s]?|degree|"
    r"studied|graduated|qualification[s]?)\b",
    re.IGNORECASE,
)
_COMPANY_REFERENCE = re.compile(
    r"\b(your|their|the)\s+(company|team|organisation|organization|work|"
    r"mission|product[s]?|client[s]?|engineer[s]?|employee[s]?)\b|"
    r"\byour\b|\bthe company\b",
    re.IGNORECASE,
)


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


# A live tailoring run had a writer reword the CV's own "1000+" as "1,000+" -
# a real, grounded number, just re-formatted with a thousands separator. `\d+`
# on "1,000" finds two runs ("1", "000"), neither of which matches the
# source's single "1000", so a truthful claim was rejected as an invented
# figure. Stripping a comma only when it sits directly between two digits
# fixes exactly that formatting difference without touching any other use of
# a comma (list separators, "Python, SQL", are untouched - a comma there is
# never between two digits).
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d)")


def _normalise(text: str) -> str:
    replaced = text
    for word, digit in _NUMBER_WORDS.items():
        replaced = re.sub(rf"\b{word}\b", digit, replaced, flags=re.IGNORECASE)
    return _THOUSANDS_SEPARATOR.sub("", replaced)


def numbers_in(text: str) -> set[str]:
    """Every number a text asserts, with spelled-out numbers normalised."""
    return set(_DIGITS.findall(_normalise(text)))


def singular(term: str) -> str:
    """A conservative singular form of `term`, for plural-only differences.

    Exists because "REST APIs" is the same claim as "REST API" - a live run
    had a perfectly grounded bullet rejected over exactly that `s`. The rules
    are deliberately shallow: they undo regular English pluralisation and
    nothing else. No stemming, no lemmatisation, no synonyms - a term that
    differs from the CV by more than a plural ending stays unsupported.
    """
    lowered = term.lower()
    if len(lowered) <= 3 or not lowered.endswith("s"):
        return lowered
    if lowered.endswith("ss") or lowered.endswith("us"):
        return lowered  # "access", "status" are not plurals
    if lowered.endswith("ies"):
        return lowered[:-3] + "y"  # libraries -> library
    if lowered.endswith("es") and lowered[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return lowered[:-2]  # boxes -> box, batches -> batch
    return lowered[:-1]  # APIs -> api, clusters -> cluster


# Standard academic degree abbreviations: a CV that says "BS" and generated
# text that says "Bachelor's" (or the reverse) are the same claim - a live run
# rejected exactly this ("the CV never mentions Bachelor" when the CV said "BS
# Data Science student"). Not a general synonym system: a small, closed list
# for one common CV-specific case, the same kind of narrow exception
# `_NUMBER_WORDS` already makes for spelled-out digits. Periods are stripped
# before comparing ("B.S." and "BS" are the same claim), and matching stays
# whole-word: "ms"/"bs" are short enough to appear inside unrelated words
# ("months", "jobs") or names ("MS" as in Microsoft), so - unlike the rest of
# this module - a plain substring check would be unsafe here.
_DEGREE_EQUIVALENTS: list[set[str]] = [
    {"bachelor", "bachelors", "bs", "bsc", "ba"},
    {"master", "masters", "ms", "msc", "ma"},
]


def _has_degree_abbreviation(term: str, corpus_lower: str) -> bool:
    """Whether a conventional abbreviation or spelled-out form of the degree
    `term` names appears in the corpus. None of this fires for anything that
    is not literally one of the words in `_DEGREE_EQUIVALENTS` - a
    candidate's actual degree level still has to appear somewhere; this only
    stops the exact wording of it from being treated as a fabrication.
    """
    lowered = term.lower().replace(".", "")
    group = next((g for g in _DEGREE_EQUIVALENTS if lowered in g), None)
    if group is None:
        return False
    corpus_no_dots = corpus_lower.replace(".", "")
    return any(
        re.search(rf"\b{re.escape(variant)}\b", corpus_no_dots) for variant in group
    )



# A handful of standard adjective/noun equivalents for the same technology -
# a live run had "REST APIs" (the CV's own words) rejected because the writer
# said "RESTful APIs" instead. Same shape and same discipline as
# `_DEGREE_EQUIVALENTS`: a small, closed list for one common CV-specific
# case, not a general synonym system - a term that isn't literally one of
# these words keeps its normal strictness.
_TERM_EQUIVALENTS: list[set[str]] = [
    {"rest", "restful"},
]


def _has_term_equivalent(term: str, corpus_lower: str) -> bool:
    """Whether a known equivalent word for `term` appears in the corpus.

    Structurally identical to `_has_degree_abbreviation` - whole-word only,
    since "rest" is short enough to appear inside unrelated words ("interest",
    "restore").
    """
    lowered = term.lower()
    group = next((g for g in _TERM_EQUIVALENTS if lowered in g), None)
    if group is None:
        return False
    return any(
        re.search(rf"\b{re.escape(variant)}\b", corpus_lower) for variant in group
    )


# Domain acronyms this project's own writing prompts are prone to spelling
# out in full - a live run had "Retrieval-Augmented Generation" rejected
# term-by-term ("Retrieval", "Augmented", "Generation" each individually
# absent) when the CV's own word was "RAG". Unlike `_TERM_EQUIVALENTS`, this
# is a word-count change, not a word-form change, so it is handled as a
# phrase-level rewrite before term extraction (see `_collapse_known_acronyms`)
# rather than inside `is_term_supported` - matching a bare "Generation" or
# "Retrieval" on its own would be far too easy to satisfy by accident.
#
# The separator between the phrase's words has to tolerate more than a plain
# ASCII hyphen: the same live run's model wrote "Retrieval‑Augmented"
# with U+2011 NON-BREAKING HYPHEN, a "smart" character plain `-` never
# matches. `_HYPHENS` covers every hyphen/dash form seen from a model in
# practice; `\s` still allows the fully spelled-out "Retrieval Augmented
# Generation" with ordinary spaces.
_HYPHENS = "-‐‑‒–—"
_ACRONYM_EXPANSIONS: dict[str, str] = {
    "retrieval augmented generation": "RAG",
}


def _collapse_known_acronyms(claim: str, corpus_lower: str) -> str:
    """Rewrite a known full-form phrase back to its acronym, but only when
    the acronym is real - i.e. actually appears in the CV - so a claim that
    invents the phrase out of nothing (no "RAG" anywhere in the CV either)
    is still rejected exactly as before this fix."""
    rewritten = claim
    for phrase, acronym in _ACRONYM_EXPANSIONS.items():
        if not re.search(rf"\b{re.escape(acronym.lower())}\b", corpus_lower):
            continue
        pattern = rf"[{_HYPHENS}\s]+".join(re.escape(w) for w in phrase.split())
        rewritten = re.sub(pattern, acronym, rewritten, flags=re.IGNORECASE)
    return rewritten


def is_term_supported(term: str, corpus_lower: str) -> bool:
    """Whether `term` appears in the CV, allowing a plural/singular difference.

    Matching stays substring-based, as it always was, so a term is still found
    inside a longer word ("SQL" within "SQLAlchemy"). The singular form, the
    degree-abbreviation check, and the term-equivalent check are only
    *additional* ways to match, never a replacement - together they widen
    what counts as supported by exactly three things: the plural `s`, a
    handful of standard degree abbreviations, and a handful of standard term
    equivalents (see `_TERM_EQUIVALENTS`).
    """
    lowered = term.lower()
    return (
        lowered in corpus_lower
        or singular(term) in corpus_lower
        or _has_degree_abbreviation(term, corpus_lower)
        or _has_term_equivalent(term, corpus_lower)
    )


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

    # Anything in square brackets is a template placeholder - "[Candidate]",
    # "[Your Name]" - and is flagged wherever it appears, including as the only
    # word on a signature line, where the sentence-initial rule would otherwise
    # let it through.
    for placeholder in _BRACKETED.findall(claim):
        terms.extend(
            token
            for token in _WORD.findall(placeholder)
            if token.lower() not in _COMMON_WORDS
        )
    return terms


# How many characters may separate a first-person pronoun from an experience
# word and still count as the same clause ("I have built Django services" -
# a handful of words apart). Any further apart and the words in between are
# almost always a different clause - found live: "I am excited to apply for
# the Data Science Intern role at Nexora Analytics, a finance and operations
# intelligence platform built for multi-unit restaurant operators" was forced
# into the strict candidate-only domain because the sentence contains "I" and,
# far later, "built" - which describes the COMPANY's platform, not the
# candidate. Not a parse of clause structure (nothing else in this module
# does real parsing either) - a deliberately simple proximity heuristic wide
# enough for a genuine candidate sentence and narrow enough to exclude a full
# company-descriptor clause sitting between the pronoun and the verb.
_EXPERIENCE_PROXIMITY_CHARS = 40


def _first_person_experience_claim(claim: str) -> bool:
    """True only when a first-person pronoun and an experience word are close
    enough together to plausibly be the candidate's own clause - not merely
    both present somewhere in the same sentence. Reuses `_FIRST_PERSON` and
    `_EXPERIENCE` exactly as `is_candidate_claim` always has; this only adds a
    distance check between where each one matches.
    """
    pronoun_positions = [m.start() for m in _FIRST_PERSON.finditer(claim)]
    if not pronoun_positions:
        return False
    experience_positions = [m.start() for m in _EXPERIENCE.finditer(claim)]
    return any(
        abs(pronoun - experience) <= _EXPERIENCE_PROXIMITY_CHARS
        for pronoun in pronoun_positions
        for experience in experience_positions
    )


def is_candidate_claim(
    claim: str,
    company_name: str = "",
    context_terms: tuple[str, ...] | list[str] = (),
) -> bool:
    """Whether `claim` asserts something about the *candidate*.

    This is the hinge of the two-domain rule. A claim about the candidate may
    only draw on the CV; a statement about the company may also draw on the
    verified company brief. Getting this wrong in the permissive direction is
    how "Arbisoft uses Django" would turn into "I have Django experience", so
    the default is deliberately strict:

    - first person plus a *nearby* experience word ("I have experience...",
      "my skills...", "I built...") -> candidate claim, CV only. This wins
      even when the company is also named, so "Arbisoft uses Django, and I
      have built Django services" is still checked strictly - the pronoun and
      the verb are a handful of words apart. It does not fire on a first
      person pronoun and an experience word that merely occur somewhere in
      the same sentence with a company-descriptor clause between them; see
      `_first_person_experience_claim`.
    - otherwise, if the claim is *about* the company - naming it, or saying
      "your company", "the team" - it is a company statement.
    - anything else falls back to candidate, so a bare "Built data platforms."
      cannot quietly borrow a company fact.
    """
    if _first_person_experience_claim(claim):
        return True
    return not refers_to_company(claim, company_name, context_terms)


def refers_to_company(
    claim: str,
    company_name: str = "",
    context_terms: tuple[str, ...] | list[str] = (),
) -> bool:
    """Whether the claim's subject is the employer rather than the candidate.

    Naming the employer or the advertised role counts, which is what lets a
    letter say "I am applying for the X role at Y" without that being read as a
    claim of experience.
    """
    if _COMPANY_REFERENCE.search(claim):
        return True
    lowered = claim.lower()
    for name in (company_name, *context_terms):
        name = name.strip().lower()
        if not name:
            continue
        if name in lowered:
            return True
        head = name.split()[0]
        if len(head) > 2 and head in lowered:
            return True
    return False


def build_company_context(brief) -> str:
    """The verified company material a letter may cite.

    Only what already survived the research agent's own grounding: the brief's
    summary and facts, each of which had to cite a retrieved result, plus the
    source titles. Nothing here can ever support a claim about the candidate.
    """
    if brief is None:
        return ""
    return "\n".join(
        [brief.company, brief.summary, *brief.facts, *(s.title for s in brief.sources)]
    )


def check_grounding(
    claims: list[str],
    source_cv: ParsedCV,
    context_terms: tuple[str, ...] | list[str] = (),
    company_context: str = "",
    company_name: str = "",
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

    `company_context` is the second evidence domain: verified company material
    a *company statement* may cite. It is kept entirely separate from the CV
    corpus and is never consulted for a claim about the candidate, so a company
    fact can never become the candidate's experience. Leave it empty - as the
    tailored CV does - and the check is exactly as strict as it has always been.
    """
    corpus = "\n".join(
        [
            source_cv.raw_text,
            # The candidate's own name, so a cover letter may be signed.
            source_cv.candidate_name or "",
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
        variant
        for phrase in context_terms
        for term in _WORD.findall(phrase)
        for variant in (term.lower(), singular(term))
    }
    # Just the employer's own name, not the full `allowed` set - see the
    # `nameable` comment below for why a candidate claim gets only this much
    # naming allowance and not the rest of `allowed` (the advertised role).
    company_only = {
        variant
        for term in _WORD.findall(company_name)
        for variant in (term.lower(), singular(term))
    }

    company_lower = company_context.lower()
    company_numbers = numbers_in(company_context)

    result = GroundingResult()
    for claim in claims:
        # Which evidence domain this claim is allowed to draw on.
        about_candidate = is_candidate_claim(claim, company_name, context_terms)
        searchable = (
            corpus_lower if about_candidate else f"{corpus_lower}\n{company_lower}"
        )
        known_numbers = (
            corpus_numbers
            if about_candidate
            else corpus_numbers | company_numbers
        )

        # The employer's proper nouns are a naming allowance for statements
        # about the employer - never evidence for a claim about the candidate.
        # A role advertised as "Junior AI Engineer" must not make "my AI skills"
        # supportable, any more than "Kubernetes Engineer" would license
        # Kubernetes - which is why a candidate claim gets only `company_only`
        # (just the employer's name), never the full `allowed` set (which also
        # contains words drawn from the *advertised role*, exactly the words
        # that trap would need). Naming the employer itself carries no such
        # risk - a live run had "...to apply my Python skills to Lumen's AI
        # initiatives" rejected because "Lumen" is not on the CV, even though
        # naming who the letter is for is not a claim of experience with them.
        nameable = company_only if about_candidate else allowed

        unsupported_terms = [
            term
            for term in distinctive_terms(_collapse_known_acronyms(claim, searchable))
            if not is_term_supported(term, searchable)
            and term.lower() not in nameable
            and singular(term) not in nameable
        ]
        unsupported_numbers = sorted(numbers_in(claim) - known_numbers)

        if unsupported_terms:
            # When the term exists in the company material but the claim is
            # about the candidate, say so - it is the difference between "we
            # never found this" and "this is true of them, not of you", and the
            # retry needs to understand which.
            borrowed = [
                term
                for term in unsupported_terms
                if company_lower and is_term_supported(term, company_lower)
            ]
            named = ", ".join(dict.fromkeys(unsupported_terms))
            reason = f"the CV never mentions {named}"
            if borrowed:
                reason += (
                    " - "
                    + ", ".join(dict.fromkeys(borrowed))
                    + " is something the company does, not something the CV "
                    "claims about the candidate"
                )
            result.issues.append(GroundingIssue(claim=claim, reason=reason))
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

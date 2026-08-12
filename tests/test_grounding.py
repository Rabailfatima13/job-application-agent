"""No-fabrication grounding (FR-10, NFR-4).

The deterministic checker needs no model at all, so these tests are pure
functions in and out. What they pin down is the line the project cares about:
rewording and reordering are fine, inventing is not.
"""

import pytest

from job_agent.models import CVEvidence, ParsedCV
from job_agent.validation import (
    EntailmentVerdicts,
    GroundingResult,
    apply_verdicts,
    check_grounding,
    split_sentences,
)
from job_agent.validation.grounding import (
    build_entailment_prompt,
    distinctive_terms,
    numbers_in,
)

EVIDENCE = [
    "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite.",
    "Built a research agent with tool calling, session memory and hooks.",
    "Task Manager API: JWT authentication, per-user task authorization.",
    "BS Computer Science",
]


@pytest.fixture
def cv() -> ParsedCV:
    return ParsedCV(
        candidate_name="Mahnoor Rauf",
        raw_text="\n".join(EVIDENCE),
        evidence=[CVEvidence(text=text, section="Experience") for text in EVIDENCE],
        skills=["Python", "FastAPI", "pytest", "SQLite"],
    )


def check(claim: str, cv: ParsedCV, **kwargs) -> GroundingResult:
    return check_grounding([claim], cv, **kwargs)


# --- Supported material passes ----------------------------------------------


def test_a_fully_grounded_set_of_cv_bullets_passes(cv):
    result = check_grounding(
        [
            "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite.",
            "Built a research agent with tool calling, session memory and hooks.",
        ],
        cv,
    )

    assert result.passed and result.is_clean
    assert len(result.grounded) == 2
    assert result.issues == []


def test_a_grounded_cover_letter_passes(cv):
    letter = (
        "I am applying for the Junior AI Engineer role at Arbisoft. "
        "I built a Task Management REST API using FastAPI and SQLite. "
        "I also built a research agent with tool calling and session memory."
    )

    result = check_grounding(
        split_sentences(letter), cv, context_terms=("Arbisoft", "Junior AI Engineer")
    )

    assert result.passed
    assert len(result.checked_claims) == 3


def test_rewording_supported_evidence_is_allowed(cv):
    # Not a substring of the CV anywhere - but every fact in it is.
    claim = "Designed and delivered a task management REST API using FastAPI and SQLite."

    assert check(claim, cv).passed


def test_combining_two_real_bullets_is_allowed(cv):
    claim = (
        "Built a Task Manager API with JWT authentication and a research agent "
        "with session memory."
    )

    assert check(claim, cv).passed


def test_reordering_and_re_emphasis_change_nothing(cv):
    forwards = check_grounding(EVIDENCE, cv)
    backwards = check_grounding(list(reversed(EVIDENCE)), cv)

    assert forwards.passed and backwards.passed
    assert sorted(forwards.grounded) == sorted(backwards.grounded)


# --- Each fabrication category is caught ------------------------------------


def test_an_invented_technology_is_rejected(cv):
    result = check(
        "Built production systems using Python, FastAPI, and Kubernetes.", cv
    )

    assert not result.passed
    assert "Kubernetes" in result.issues[0].reason
    # The supported technologies are not blamed.
    assert "FastAPI" not in result.issues[0].reason


def test_invented_years_of_experience_are_rejected(cv):
    digits = check("Has 5 years of professional Python experience.", cv)
    spelled = check("Has five years of professional Python experience.", cv)

    assert not digits.passed and not spelled.passed
    assert "5" in digits.issues[0].reason
    # Spelled-out numbers are normalised, so the wording cannot dodge the check.
    assert "5" in spelled.issues[0].reason


def test_an_invented_metric_is_rejected(cv):
    result = check("Improved API response times by 40% across the platform.", cv)

    assert not result.passed
    assert "40" in result.issues[0].reason


def test_an_invented_project_is_rejected(cv):
    result = check("Built the Atlas recommendation engine for a retail client.", cv)

    assert not result.passed
    assert "Atlas" in result.issues[0].reason


def test_an_invented_certification_is_rejected(cv):
    result = check("AWS Certified Solutions Architect, Associate level.", cv)

    assert not result.passed
    assert "AWS" in result.issues[0].reason


def test_invented_education_is_rejected(cv):
    result = check("MSc in Machine Learning from Stanford University.", cv)

    assert not result.passed
    assert "Stanford" in result.issues[0].reason


def test_an_invented_employer_is_rejected(cv):
    result = check("Worked as a backend engineer at Netflix.", cv)

    assert not result.passed
    assert "Netflix" in result.issues[0].reason


def test_a_mixed_set_separates_the_good_from_the_bad(cv):
    result = check_grounding(
        [
            "Built a Task Management REST API with FastAPI.",
            "Deployed it to Kubernetes in production.",
            "Built a research agent with tool calling.",
        ],
        cv,
    )

    assert len(result.grounded) == 2
    assert len(result.issues) == 1
    assert result.issues[0].claim.startswith("Deployed")
    assert len(result.checked_claims) == 3


# --- The evidence boundary ---------------------------------------------------


def test_the_job_posting_cannot_widen_what_counts_as_evidence(cv):
    # The posting asking for Kubernetes must never make claiming it acceptable.
    result = check("Experienced with Kubernetes, as the role requires.", cv)

    assert not result.passed


def test_context_terms_allow_naming_the_employer_only(cv):
    addressed = check(
        "I would be glad to join Arbisoft as a Junior AI Engineer.",
        cv,
        context_terms=("Arbisoft", "Junior AI Engineer"),
    )
    smuggled = check(
        "I have deep Kubernetes experience.",
        cv,
        context_terms=("Arbisoft", "Junior AI Engineer"),
    )

    assert addressed.passed
    assert not smuggled.passed  # context is for the employer, not for skills


def test_parsed_evidence_counts_as_evidence_even_if_raw_text_is_thin():
    cv = ParsedCV(
        raw_text="(unreadable scan)",
        evidence=[CVEvidence(text="Built a FastAPI service.")],
    )

    assert check("Built a service with FastAPI.", cv).passed


# --- The primitives ----------------------------------------------------------


def test_distinctive_terms_ignore_sentence_openers_and_plain_words():
    terms = distinctive_terms("Built a REST API with FastAPI and Kubernetes.")

    assert "Built" not in terms  # sentence-initial capital carries no meaning
    assert {"REST", "API", "FastAPI", "Kubernetes"} <= set(terms)


def test_distinctive_terms_catch_technical_spellings():
    assert "C#" in distinctive_terms("wrote C# services")
    assert "Node.js" in distinctive_terms("maintained Node.js tooling")


def test_numbers_normalise_spelled_out_words():
    assert numbers_in("three years and 40%") == {"3", "40"}


def test_an_empty_claim_list_is_trivially_clean(cv):
    assert check_grounding([], cv).passed


# --- Model-assisted second stage --------------------------------------------


def test_the_entailment_prompt_shows_the_cv_evidence_and_numbered_claims(cv):
    prompt = build_entailment_prompt(["Led the migration."], cv)

    assert "CV EVIDENCE:" in prompt
    assert "- BS Computer Science" in prompt
    assert "[0] Led the migration." in prompt


def test_model_verdicts_can_reject_a_claim_the_deterministic_pass_allowed(cv):
    # "Led a team" has no distinctive term and no number, so stage 1 allows it;
    # this is exactly what the optional entailment pass exists for.
    deterministic = check_grounding(["Led a team of engineers."], cv)
    assert deterministic.passed

    verdicts = EntailmentVerdicts.model_validate(
        {
            "verdicts": [
                {
                    "claim_index": 0,
                    "supported": False,
                    "reason": "the CV never mentions leading a team",
                }
            ]
        }
    )
    final = apply_verdicts(deterministic, verdicts)

    assert not final.passed
    assert final.grounded == []
    assert "leading a team" in final.issues[0].reason


def test_model_verdicts_never_rehabilitate_a_rejected_claim(cv):
    deterministic = check_grounding(["Deployed to Kubernetes."], cv)
    verdicts = EntailmentVerdicts.model_validate(
        {"verdicts": [{"claim_index": 0, "supported": True}]}
    )

    final = apply_verdicts(deterministic, verdicts)

    assert not final.passed  # the deterministic rejection stands


def test_missing_or_out_of_range_verdicts_leave_the_result_alone(cv):
    deterministic = check_grounding(["Built a FastAPI service."], cv)

    final = apply_verdicts(
        deterministic,
        EntailmentVerdicts.model_validate(
            {"verdicts": [{"claim_index": 99, "supported": False}]}
        ),
    )

    assert final.passed
    assert final.grounded == deterministic.grounded


def test_feedback_lists_every_violation_with_its_reason(cv):
    result = check_grounding(
        ["Deployed to Kubernetes.", "Has 9 years of experience."], cv
    )

    feedback = result.feedback()
    assert "Kubernetes" in feedback
    assert "9" in feedback
    assert feedback.count("->") == 2

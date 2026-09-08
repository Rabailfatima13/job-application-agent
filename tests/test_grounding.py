"""No-fabrication grounding (FR-10, NFR-4).

The deterministic checker needs no model at all, so these tests are pure
functions in and out. What they pin down is the line the project cares about:
rewording and reordering are fine, inventing is not.
"""

import pytest

from job_agent.models import CompanyBrief, CVEvidence, ParsedCV, Source
from job_agent.validation import (
    EntailmentVerdicts,
    GroundingResult,
    apply_verdicts,
    check_grounding,
    split_sentences,
)
from job_agent.validation.grounding import (
    build_company_context,
    build_entailment_prompt,
    distinctive_terms,
    is_candidate_claim,
    numbers_in,
    refers_to_company,
    singular,
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


# --- Regression: comma thousands-separators (found in a live tailoring run) --


def test_numbers_normalise_comma_thousands_separators():
    # A live tailoring run had the writer reword the CV's own "1000+" as
    # "1,000+" - `\d+` alone splits "1,000" into two runs ("1", "000"),
    # neither of which matches the source's single "1000", so a genuinely
    # grounded claim was rejected as an invented figure.
    assert numbers_in("1,000+ attendees") == numbers_in("1000+ attendees")


def test_a_comma_formatted_number_matching_the_cv_is_supported():
    cv_with_a_number = ParsedCV(
        raw_text="Organized a conference for 1000+ attendees.",
        evidence=[
            CVEvidence(text="Organized a conference for 1000+ attendees.")
        ],
    )

    result = check("Organized a conference for 1,000+ attendees.", cv_with_a_number)

    assert result.passed, result.issues[0].reason if result.issues else ""


def test_a_comma_formatted_number_absent_from_the_cv_is_still_rejected():
    # The fix must not stop checking numbers altogether - only stop
    # mis-splitting a real one across a thousands separator.
    cv_without_that_number = ParsedCV(
        raw_text="Built internal tools.",
        evidence=[CVEvidence(text="Built internal tools.")],
    )

    result = check(
        "Organized a conference for 1,000+ attendees.", cv_without_that_number
    )

    assert not result.passed


# --- Regression: plural/singular variation (found in the live rehearsal) -----


def test_a_plural_of_a_real_cv_term_is_supported(cv):
    # The live run rejected exactly this: the CV says "REST API", the writer
    # wrote "REST APIs", and a perfectly grounded bullet was thrown away.
    result = check(
        "I have experience building REST APIs with FastAPI and testing with pytest.",
        cv,
    )

    assert result.passed, result.issues[0].reason if result.issues else ""


@pytest.mark.parametrize(
    "claim",
    [
        "Delivered REST APIs backed by SQLite.",
        "Built Task Management APIs with FastAPI.",
        "Secured Task Manager APIs with JWTs.",
    ],
)
def test_regular_plurals_of_supported_terms_pass(cv, claim):
    assert check(claim, cv).passed


def test_singular_of_a_plural_cv_term_is_supported():
    cv = ParsedCV(
        raw_text="Maintained data pipelines and Kubernetes clusters.",
        evidence=[CVEvidence(text="Maintained data pipelines and Kubernetes clusters.")],
    )

    assert check("Maintained a Kubernetes cluster.", cv).passed


def test_plural_normalisation_does_not_rescue_an_invented_term(cv):
    # The whole risk of loosening the match: it must not let fabrication in.
    for claim in (
        "Managed Kubernetes clusters in production.",
        "Ran workloads on AWS.",
        "Holds Azure certifications.",
        "Studied at Stanford.",
    ):
        assert not check(claim, cv).passed, claim


def test_words_that_merely_end_in_s_are_not_stripped(cv):
    # "Kafkas" is not the plural of anything in this CV; "Access" and "Status"
    # must not be mangled into "acces"/"statu" and accidentally matched.
    assert not check("Streamed events through Kafkas.", cv).passed
    assert not check("Administered MS Access databases.", cv).passed


def test_the_singular_helper_only_undoes_regular_plurals():
    assert singular("APIs") == "api"
    assert singular("clusters") == "cluster"
    assert singular("models") == "model"
    assert singular("projects") == "project"
    assert singular("libraries") == "library"
    assert singular("batches") == "batch"
    # Left alone: too short, not a plural, or a false plural.
    assert singular("AWS") == "aws"
    assert singular("access") == "access"
    assert singular("status") == "status"
    assert singular("Python") == "python"


def test_matching_is_still_substring_based_for_compound_terms(cv):
    # "SQL" inside "SQLAlchemy" was supported before this change and still is.
    assert check("Wrote SQL against the task database.", cv).passed


def test_the_candidates_own_name_is_grounded(cv):
    # So a cover letter can be signed without tripping the guard.
    assert check("Sincerely, Mahnoor Rauf", cv).passed


def test_a_template_placeholder_is_still_flagged(cv):
    # The prompt is what stops these being written; grounding stays strict.
    assert not check("Sincerely, [Candidate]", cv).passed


@pytest.mark.parametrize(
    "claim", ["[Candidate]", "[Your Name]", "[Hiring Manager]", "Dear [Company] team"]
)
def test_a_bracketed_placeholder_is_flagged_wherever_it_appears(cv, claim):
    # Including as the only word on a line, where the sentence-initial rule
    # would otherwise skip it.
    assert not check(claim, cv).passed


# --- Two evidence domains: candidate vs company ------------------------------
#
# The rule under test: verified company facts may support statements ABOUT THE
# COMPANY, and may never support a claim about the candidate.

COMPANY_CONTEXT = (
    "Arbisoft\n"
    "Arbisoft is a global software and product development company.\n"
    "Arbisoft uses Python, Django and Flask\n"
    "Arbisoft specializes in AI, Data Science and custom development\n"
    "86% of Arbisoft employees would recommend working there to a friend\n"
)


def company_check(claim: str, cv: ParsedCV) -> GroundingResult:
    return check_grounding(
        [claim],
        cv,
        context_terms=("Arbisoft", "Junior AI Engineer"),
        company_context=COMPANY_CONTEXT,
        company_name="Arbisoft",
    )


@pytest.mark.parametrize(
    "claim",
    [
        "I have experience with Django.",
        "My Django skills align with Arbisoft.",
        "I have worked with Django on production systems.",
        "I have experience in Data Science.",
        "My background in Data Science fits the role.",
        "I improved performance by 86%.",
        "I achieved an 86% reduction in latency.",
    ],
)
def test_a_company_fact_can_never_support_a_candidate_claim(cv, claim):
    # The core security rule. Django, Data Science and 86% are all in the
    # company context and none are in the CV.
    assert not company_check(claim, cv).passed, claim


@pytest.mark.parametrize(
    "claim",
    [
        "Your company uses Django.",
        "Arbisoft uses Django and Flask.",
        "86% of employees recommend the company.",
        "I am interested in Arbisoft's AI work.",
        "I am drawn to the company's Data Science work.",
        "Arbisoft specializes in AI and Data Science.",
    ],
)
def test_a_statement_about_the_company_may_cite_the_verified_brief(cv, claim):
    assert company_check(claim, cv).passed, claim


def test_the_advertised_role_title_cannot_license_a_candidate_skill(cv):
    # Found live: the role is "Junior AI Engineer", so "AI" was allowed as an
    # employer proper noun - and leaked into a claim about the candidate.
    assert not company_check("My AI skills match Arbisoft's work.", cv).passed
    assert not company_check("I have strong AI experience.", cv).passed


def test_a_skill_named_in_the_role_title_is_still_not_evidence(cv):
    # The general form of the same hole: a role called "Kubernetes Engineer"
    # must not make Kubernetes claimable.
    result = check_grounding(
        ["I have deep Kubernetes experience."],
        cv,
        context_terms=("Arbisoft", "Kubernetes Engineer"),
        company_name="Arbisoft",
    )

    assert not result.passed


def test_a_company_named_by_its_leading_word_still_counts_as_the_subject():
    # "Arbisoft Pvt Ltd" referred to as just "Arbisoft".
    assert refers_to_company("Arbisoft builds platforms.", "Arbisoft Pvt Ltd")
    assert not refers_to_company("Netflix builds platforms.", "Arbisoft Pvt Ltd")


def test_naming_the_role_is_still_allowed_in_a_company_statement(cv):
    assert company_check(
        "I am applying for the Junior AI Engineer role at Arbisoft.", cv
    ).passed


def test_the_rejection_explains_that_the_fact_belongs_to_the_company(cv):
    result = company_check("I have experience with Django.", cv)

    reason = result.issues[0].reason
    assert "Django" in reason
    assert "something the company does" in reason


def test_company_context_is_ignored_entirely_for_candidate_claims(cv):
    # Same claim, same CV: the company context must make no difference.
    with_context = company_check("I have experience with Django.", cv)
    without_context = check_grounding(["I have experience with Django."], cv)

    assert not with_context.passed and not without_context.passed


def test_an_unattributed_claim_defaults_to_the_strict_candidate_domain(cv):
    # No "I", no company reference: it must NOT quietly borrow a company fact.
    assert not company_check("Built Django services for clients.", cv).passed


def test_a_mixed_sentence_is_judged_as_a_candidate_claim(cv):
    # Naming the company must not launder a claim about the candidate.
    assert not company_check(
        "Arbisoft uses Django, and I have built Django services.", cv
    ).passed


def test_company_facts_never_widen_the_cv_for_the_tailored_cv_path(cv):
    # The tailored CV is checked with no company context at all.
    assert not check_grounding(["Built Django services."], cv).passed


def test_a_fabrication_absent_from_both_domains_still_fails(cv):
    for claim in (
        "Your company uses Kubernetes.",  # not in the brief either
        "I hold an AWS certification.",
        "I studied at Stanford.",
    ):
        assert not company_check(claim, cv).passed, claim


def test_claim_subject_detection():
    assert is_candidate_claim("I have experience with Django.", "Arbisoft")
    assert is_candidate_claim("My skills include Django.", "Arbisoft")
    # First person, but the predicate is an attitude, not experience.
    assert not is_candidate_claim("I am interested in Arbisoft's AI work.", "Arbisoft")
    assert not is_candidate_claim("Your company uses Django.", "Arbisoft")
    assert not is_candidate_claim("Arbisoft builds data platforms.", "Arbisoft")
    # Unattributed defaults to candidate, i.e. strict.
    assert is_candidate_claim("Built Django services.", "Arbisoft")


def test_the_existing_regressions_survive_the_two_domain_change(cv):
    # Everything the earlier live runs taught us, re-checked with company
    # context switched on.
    assert company_check(
        "I have experience building REST APIs with FastAPI and testing with pytest.",
        cv,
    ).passed
    for fabrication in (
        "I managed Kubernetes clusters.",
        "AWS Certified Solutions Architect.",
        "I improved latency by 40%.",
        "MSc from Stanford University.",
        "Sincerely, [Candidate]",
    ):
        assert not company_check(fabrication, cv).passed, fabrication


def test_build_company_context_uses_only_verified_brief_material():
    brief = CompanyBrief(
        company="Arbisoft",
        summary="Arbisoft builds data platforms.",
        facts=["Uses Django."],
        sources=[Source(title="Arbisoft home", url="https://arbisoft.com")],
    )

    context = build_company_context(brief)

    assert "Arbisoft builds data platforms." in context
    assert "Uses Django." in context
    assert "Arbisoft home" in context
    assert build_company_context(None) == ""


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


# --- Regression: two false rejections from a real live run -------------------
#
# A live run against a real CV (a BS Data Science student applying to a
# "Data Science Intern" role at "Nexora Analytics") scored 10% and produced no
# tailored documents: every grounding attempt was rejected, and the trace
# showed two distinct false positives rather than genuine fabrications. Both
# are reproduced here with the exact sentences the writer actually generated.


@pytest.fixture
def data_science_cv() -> ParsedCV:
    """The relevant slice of the real CV from the live run - "BS", not the
    spelled-out "Bachelor's" the writer used."""
    evidence = (
        "Motivated 20-year-old BS Data Science student at the National "
        "University of Computer and Emerging Sciences."
    )
    return ParsedCV(
        candidate_name="Rabail Fatima",
        raw_text=evidence,
        evidence=[CVEvidence(text=evidence, section="Education")],
    )


# Fix 1: degree abbreviations ---------------------------------------------


def test_a_bs_degree_supports_a_bachelors_phrasing(data_science_cv):
    # A: the exact live failure - "the CV never mentions Bachelor".
    result = check(
        "Currently pursuing a Bachelor's degree in Data Science at the "
        "National University of Computer and Emerging Sciences.",
        data_science_cv,
    )

    assert result.passed, result.issues[0].reason if result.issues else ""


def test_a_dotted_bs_abbreviation_supports_a_bachelors_phrasing():
    cv = ParsedCV(
        raw_text="B.S. Computer Science, National University.",
        evidence=[CVEvidence(text="B.S. Computer Science, National University.")],
    )

    result = check("Bachelor's degree in Computer Science.", cv)

    assert result.passed, result.issues[0].reason if result.issues else ""


def test_degree_abbreviations_do_not_bleed_across_degree_levels(data_science_cv):
    # The CV says BS (bachelor's) - a claim of a MASTER's must still fail.
    # Proves the equivalence table is closed, not a general synonym system.
    result = check(
        "Currently pursuing a Master's degree in Data Science.", data_science_cv
    )

    assert not result.passed
    assert "Master" in result.issues[0].reason


def test_an_absent_degree_is_still_rejected():
    # No degree of any kind on this CV - the abbreviation table must not
    # manufacture support out of nothing. "Bachelor's" is not sentence-initial
    # here, unlike a bare "Bachelor's degree..." claim, so it is actually
    # checked (a sentence-initial capital carries no distinctive-term weight
    # regardless of this fix - see `distinctive_terms`).
    cv = ParsedCV(
        raw_text="Built a FastAPI service.",
        evidence=[CVEvidence(text="Built a FastAPI service.")],
    )

    result = check("Currently pursuing a Bachelor's degree in Computer Science.", cv)

    assert not result.passed
    assert "Bachelor" in result.issues[0].reason


def test_unrelated_capitalised_terms_are_unaffected_by_the_degree_table(cv):
    # The degree-abbreviation check must only ever fire for the exact words
    # in _DEGREE_EQUIVALENTS - everything else keeps its normal strictness.
    assert not check("Built production systems using Kubernetes.", cv).passed


# Fix 2: a company-descriptor clause is not a candidate experience claim -----

NEXORA_COMPANY_CONTEXT = (
    "Nexora Analytics\n"
    "Nexora Analytics is a finance and operations intelligence platform for "
    "multi-unit restaurant operators and franchise brands, providing "
    "real-time visibility into sales, labor, and inventory.\n"
    "The company operates in the QSR analytics space, delivering "
    "cutting-edge technology solutions.\n"
)


def nexora_check(claim: str, cv: ParsedCV) -> GroundingResult:
    return check_grounding(
        [claim],
        cv,
        context_terms=("Nexora Analytics", "Data Science Intern"),
        company_context=NEXORA_COMPANY_CONTEXT,
        company_name="Nexora Analytics",
    )


def test_a_company_description_clause_is_not_a_candidate_experience_claim(
    data_science_cv,
):
    # B: the exact live failure - rejected as "the CV never mentions Intern,
    # Nexora, Analytics", even though the rejection's own explanation said
    # "Nexora, Analytics is something the company does, not something the CV
    # claims about the candidate". "built" here describes Nexora Analytics'
    # platform, not the candidate.
    claim = (
        "I am excited to apply for the Data Science Intern role at Nexora "
        "Analytics, a finance and operations intelligence platform built for "
        "multi-unit restaurant operators and franchise brands."
    )

    result = nexora_check(claim, data_science_cv)

    assert result.passed, result.issues[0].reason if result.issues else ""


def test_a_second_company_description_clause_from_the_same_run(data_science_cv):
    # The second rejected sentence from the same live run - same bug
    # ("delivering" instead of "built"), different wording.
    claim = (
        "I am particularly drawn to Nexora Analytics' focus on delivering "
        "real-time visibility into sales, labor, and inventory."
    )

    result = nexora_check(claim, data_science_cv)

    assert result.passed, result.issues[0].reason if result.issues else ""


def test_genuine_unsupported_candidate_experience_is_still_rejected(data_science_cv):
    # C: a real fabrication must still be caught - the fix narrows a false
    # positive, it must not open a hole for a true one.
    result = nexora_check("I built production ML systems.", data_science_cv)

    assert not result.passed
    assert "ML" in result.issues[0].reason


def test_a_genuinely_close_candidate_claim_still_passes(data_science_cv):
    result = nexora_check("I built a Python application.", data_science_cv)

    # Not claimed anywhere on this CV, so still correctly rejected - this
    # proves the proximity check narrows the *false positive*, not the check
    # itself: "I built" stays close together and is still evaluated strictly.
    assert not result.passed
    assert "Python" in result.issues[0].reason


def test_the_django_leak_protection_still_holds(cv):
    # D: preserve the original protection this whole design exists for.
    company_context = "Arbisoft\nArbisoft uses Django.\n"
    result = check_grounding(
        ["I have experience with Django."],
        cv,
        context_terms=("Arbisoft",),
        company_context=company_context,
        company_name="Arbisoft",
    )

    assert not result.passed
    assert "Django" in result.issues[0].reason


def test_a_mixed_sentence_naming_the_company_is_still_checked_strictly(cv):
    # The proximity check must not reopen the hole the module's own docstring
    # warns about: pronoun and experience word close together, company also
    # named - still a candidate claim, still checked against the CV alone.
    assert not company_check(
        "Arbisoft uses Django, and I have built Django services.", cv
    ).passed


# --- Offline replay: the exact captured trace from the live run --------------


def test_offline_replay_of_the_real_rejected_run(data_science_cv):
    """Every claim `writing:grounding_guard` actually rejected in the live
    run, replayed verbatim through the fixed checker - no live API call."""
    claims = [
        "Currently pursuing a Bachelor's degree in Data Science at the "
        "National University of Computer and Emerging Sciences.",
        "I am excited to apply for the Data Science Intern role at Nexora "
        "Analytics, a finance and operations intelligence platform built for "
        "multi-unit restaurant operators and franchise brands.",
        "I am particularly drawn to Nexora Analytics' focus on delivering "
        "real-time visibility into sales, labor, and inventory.",
    ]

    result = check_grounding(
        claims,
        data_science_cv,
        context_terms=("Nexora Analytics", "Data Science Intern"),
        company_context=NEXORA_COMPANY_CONTEXT,
        company_name="Nexora Analytics",
    )

    assert result.passed, result.feedback()
    assert len(result.grounded) == 3

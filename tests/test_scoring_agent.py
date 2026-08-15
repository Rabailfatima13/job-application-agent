"""Scoring agent (FR-4, US-3).

The model is a stub returning judgements, so these tests check what the agent
is actually responsible for: the rubric, the evidence rules, and the shape of
the FitReport - not a model's judgement quality. No test makes an API call.
"""

import json

import pytest

from job_agent.agents import ScoringAgent, compute_fit_score
from job_agent.agents.scoring import SCORING_SYSTEM_PROMPT, build_scoring_prompt
from job_agent.memory.session import RunContext
from job_agent.models import (
    CVEvidence,
    FitReport,
    JobDescription,
    ParsedCV,
    RoleRequirement,
)
from job_agent.observability import TraceCollector
from job_agent.validation import RetryExhaustedError

EVIDENCE = [
    "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite.",
    "Built a research agent with tool calling, session memory and hooks.",
    "Task Manager API: JWT authentication, per-user task authorization.",
    "BS Computer Science",
]

REQUIREMENTS = [
    "Strong Python",
    "Experience building REST APIs (FastAPI or similar)",
    "Familiarity with LLM APIs and prompt engineering",
    "Testing discipline (pytest)",
    "Comfortable with SQL databases",
]


@pytest.fixture
def cv() -> ParsedCV:
    return ParsedCV(
        candidate_name="Mahnoor Rauf",
        raw_text="\n".join(EVIDENCE),
        evidence=[CVEvidence(text=text, section="Experience") for text in EVIDENCE],
        skills=["Python", "FastAPI", "pytest", "SQL"],
    )


@pytest.fixture
def job() -> JobDescription:
    return JobDescription(
        role="Junior AI Engineer",
        company="Arbisoft",
        raw_text="...",
        requirements=[RoleRequirement(text=text) for text in REQUIREMENTS],
        responsibilities=["Help build LLM-backed product features"],
        skills=["Python", "FastAPI"],
    )


def judgement(
    met_pairs: dict[int, int | None],
    emphasis: list[int] | None = None,
    level: str = "match",
) -> str:
    """A model reply: {requirement_index: evidence_index or None}.

    Only ever produces the two ends of the match-level scale - "match" when
    evidence is given, "missing" when it is not - which is exactly the
    met/unmet distinction every pre-existing test in this file was written
    against. `level` overrides the non-missing side for the handful of tests
    that need a partial or related claim instead.
    """
    return json.dumps(
        {
            "judgements": [
                {
                    "requirement_index": requirement,
                    "match_level": level if evidence is not None else "missing",
                    "evidence_index": evidence,
                    "reason": "",
                }
                for requirement, evidence in met_pairs.items()
            ],
            "recommended_emphasis": emphasis or [],
        }
    )


def score_with(make_router, cv, job, reply, collector=None) -> FitReport:
    router = make_router(heavy_replies=[reply])
    return ScoringAgent(router, collector=collector).score(cv, job)


# --- The scoring rule --------------------------------------------------------


def test_three_of_five_equally_weighted_requirements_scores_zero_point_six(
    make_router, cv, job
):
    # Requirements 0, 1, 3 met; 2 and 4 not. All must-have, so 3/5 exactly.
    report = score_with(
        make_router, cv, job, judgement({0: 0, 1: 0, 2: None, 3: 0, 4: None})
    )

    assert report.overall_fit == 0.6
    assert sum(1 for m in report.requirement_matches if m.met) == 3


def test_nice_to_have_requirements_count_half(make_router, cv):
    job = JobDescription(
        role="AI Engineer",
        company="Arbisoft",
        raw_text="...",
        requirements=[
            RoleRequirement(text="Python", must_have=True),
            RoleRequirement(text="REST APIs", must_have=True),
            RoleRequirement(text="Kubernetes", must_have=False),
            RoleRequirement(text="React", must_have=False),
        ],
    )

    # One must-have (1.0) and one nice-to-have (0.5) met, out of 3.0 total.
    report = score_with(make_router, cv, job, judgement({0: 0, 1: None, 2: 1, 3: None}))

    assert report.overall_fit == 0.5


def test_the_rubric_is_testable_on_its_own():
    must, nice = RoleRequirement(text="a"), RoleRequirement(text="b", must_have=False)

    assert compute_fit_score([(must, True), (must, True)]) == 1.0
    assert compute_fit_score([(must, False), (nice, False)]) == 0.0
    assert compute_fit_score([(must, True), (nice, False)]) == pytest.approx(
        0.667, abs=1e-3
    )
    # A posting with no requirements cannot demonstrate fit.
    assert compute_fit_score([]) == 0.0


def test_a_strong_candidate_scores_higher_than_a_weak_one(make_router, cv, job):
    strong = score_with(
        make_router, cv, job, judgement({0: 0, 1: 0, 2: 1, 3: 0, 4: 0})
    )
    weak = score_with(
        make_router, cv, job, judgement({0: 0, 1: None, 2: None, 3: None, 4: None})
    )

    assert strong.overall_fit == 1.0
    assert weak.overall_fit == 0.2
    assert strong.overall_fit > weak.overall_fit


@pytest.mark.parametrize(
    "reply",
    [
        judgement({0: 0, 1: 0, 2: 0, 3: 0, 4: 0}),  # everything met
        judgement({}),  # nothing judged at all
        judgement({0: 99, 1: 99, 2: 99, 3: 99, 4: 99}),  # everything unsupported
    ],
)
def test_the_score_always_stays_within_zero_and_one(make_router, cv, job, reply):
    report = score_with(make_router, cv, job, reply)
    assert 0.0 <= report.overall_fit <= 1.0


# --- Evidence rules ----------------------------------------------------------


def test_every_met_requirement_carries_evidence_from_the_cv(make_router, cv, job):
    reply = judgement({0: 0, 1: 2, 2: None, 3: 1, 4: 0})
    report = score_with(make_router, cv, job, reply)

    for match in report.requirement_matches:
        if match.met:
            assert match.evidence in EVIDENCE
        else:
            assert match.evidence == ""


def test_a_match_claimed_without_real_evidence_is_downgraded(make_router, cv, job):
    collector = TraceCollector()
    # The model claims requirement 2 is met but points at evidence that does
    # not exist - the classic way an unsupported claim would sneak in.
    reply = judgement({0: 0, 1: 0, 2: 42, 3: None, 4: None})
    report = score_with(make_router, cv, job, reply, collector)

    unsupported = report.requirement_matches[2]
    assert unsupported.met is False
    assert unsupported.evidence == ""
    assert report.overall_fit == 0.4  # the rejected claim earns nothing
    guard = [e for e in collector.events if e.name == "scoring:evidence_guard"]
    assert guard and "unsupported match" in guard[0].result


def test_a_requirement_the_model_ignored_becomes_a_gap(make_router, cv, job):
    # Only requirements 0 and 1 judged; the rest were never mentioned.
    report = score_with(make_router, cv, job, judgement({0: 0, 1: 0}))

    assert len(report.requirement_matches) == len(REQUIREMENTS)
    met_flags = [m.met for m in report.requirement_matches]
    assert met_flags == [True, True, False, False, False]
    assert report.gaps == REQUIREMENTS[2:]


def test_a_requirement_the_model_invented_is_discarded(make_router, cv, job):
    collector = TraceCollector()
    reply = json.loads(judgement({0: 0}))
    reply["judgements"].append(
        {"requirement_index": 77, "match_level": "match", "evidence_index": 0}
    )

    report = score_with(make_router, cv, job, json.dumps(reply), collector)

    assert len(report.requirement_matches) == len(REQUIREMENTS)
    assert [m.requirement for m in report.requirement_matches] == REQUIREMENTS
    guard = [e for e in collector.events if e.name == "scoring:evidence_guard"]
    assert guard and "unknown requirement index" in guard[0].result


def test_gaps_list_the_unmet_requirements_must_haves_first(make_router, cv):
    job = JobDescription(
        role="AI Engineer",
        company="Arbisoft",
        raw_text="...",
        requirements=[
            RoleRequirement(text="Kubernetes", must_have=False),
            RoleRequirement(text="Python", must_have=True),
        ],
    )

    report = score_with(make_router, cv, job, judgement({0: None, 1: None}))

    assert report.gaps == ["Python", "Kubernetes"]


# --- Requirements vs. responsibilities ---------------------------------------


def test_responsibilities_are_never_scored(make_router, cv, job):
    report = score_with(make_router, cv, job, judgement({0: 0, 1: 0}))

    scored = [m.requirement for m in report.requirement_matches]
    assert scored == REQUIREMENTS
    assert "Help build LLM-backed product features" not in scored
    assert "Help build LLM-backed product features" not in report.gaps


def test_responsibilities_are_not_even_sent_to_the_model(cv, job):
    prompt = build_scoring_prompt(cv, job)

    assert "Help build LLM-backed product features" not in prompt
    assert "Strong Python" in prompt
    # The raw CV text is withheld too, so evidence can only be pointed at.
    assert "CV EVIDENCE:" in prompt


# --- Recommended emphasis ----------------------------------------------------


def test_recommended_emphasis_is_real_cv_evidence(make_router, cv, job):
    report = score_with(
        make_router, cv, job, judgement({0: 0, 1: 0}, emphasis=[1, 0])
    )

    assert report.recommended_emphasis == [EVIDENCE[1], EVIDENCE[0]]
    for suggestion in report.recommended_emphasis:
        assert suggestion in cv.raw_text


def test_emphasis_pointing_at_nothing_is_dropped_and_duplicates_collapse(
    make_router, cv, job
):
    report = score_with(
        make_router, cv, job, judgement({0: 0}, emphasis=[0, 99, 0, -1, 2])
    )

    assert report.recommended_emphasis == [EVIDENCE[0], EVIDENCE[2]]


# --- Report shape, routing, tracing, retry -----------------------------------


def test_role_and_company_come_from_the_parsed_job(make_router, cv, job):
    report = score_with(make_router, cv, job, judgement({0: 0}))

    assert (report.role, report.company) == ("Junior AI Engineer", "Arbisoft")
    assert FitReport.model_validate(report.model_dump()) == report


def test_scoring_runs_on_the_heavy_tier_through_the_router(make_router, cv, job):
    router = make_router(heavy_replies=[judgement({0: 0})])

    ScoringAgent(router).score(cv, job)

    assert len(router.client("heavy").calls) == 1
    assert router.client("light").calls == []  # judgement is not a cheap step


def test_the_model_call_is_traced(make_router, cv, job):
    collector = TraceCollector()
    score_with(make_router, cv, job, judgement({0: 0}), collector)

    (event,) = collector.events
    assert event.kind == "model"
    assert event.name == "model:heavy:score_fit"
    assert event.agent == "scoring"
    assert (event.input_tokens, event.output_tokens) == (10, 5)


def test_malformed_output_is_retried_through_the_existing_guard(make_router, cv, job):
    router = make_router(
        heavy_replies=["I think they're a great fit!", judgement({0: 0})]
    )

    report = ScoringAgent(router).score(cv, job)

    assert report.overall_fit == 0.2
    stub = router.client("heavy")
    assert len(stub.calls) == 2
    assert "rejected by the schema validator" in stub.calls[1]["messages"][0]["content"]


def test_persistently_malformed_output_never_becomes_a_report(make_router, cv, job):
    router = make_router(heavy_replies=["nope", "nope", "nope"])

    with pytest.raises(RetryExhaustedError):
        ScoringAgent(router).score(cv, job)


def test_run_scores_from_the_run_context(make_router, cv, job):
    context = RunContext(cv_text=cv.raw_text, jd_text=job.raw_text)
    context.parsed_cv, context.job = cv, job
    router = make_router(heavy_replies=[judgement({0: 0, 1: 0})])

    report = ScoringAgent(router).run(context)

    assert report.overall_fit == 0.4


def test_run_refuses_to_score_unparsed_inputs(make_router, cv):
    router = make_router(heavy_replies=[judgement({0: 0})])

    with pytest.raises(ValueError, match="parsed CV"):
        ScoringAgent(router).run(RunContext(cv_text="cv", jd_text="jd"))


# --- Regression: the four-level match rubric ---------------------------------
#
# A live run against a real CV scored 10% and reported "the CV never mentions
# Bachelor" (the CV said "BS Data Science student") and treated Python as
# entirely missing (the CV said "Basic knowledge on ... Python"). The root
# cause was a met/unmet-only rubric with no way to represent "the evidence is
# real, just weaker than the posting asks for" or "this is adjacent, not
# direct" - every one of those collapsed into "unmet". These tests use the
# real CV bullets and the real JD requirements from that run, with a stubbed
# judgement standing in for what a correctly-reasoning model would return -
# what is under test is that the *system* scores, downgrades, and reports each
# level correctly, not whether any particular live model reasons correctly.

REAL_CV_BULLETS = [
    "Motivated BS Data Science student at the National University of Computer "
    "and Emerging Sciences, actively seeking programming and data-centric "
    "project opportunities to address real-world problems.",
    "Taking online python course to enhance skills.",
    "Efficient C++ programmer with experience in Image Processing/Editing and "
    "Console based games like mine sweeper.",
    "Basic knowledge on SQL and Python.",
    "Demonstrates strong academic performance, volunteer involvement, and "
    "exceptional communication abilities in both English and Urdu.",
    "Key Modules: Programming Fundamentals C++, Object Oriented Programming, "
    "COAL, Data Structures, Calculus, Multivariable Calculus.",
    "Subjects: Math(A+), Physics(A+), Chemistry(A+)",
]

REAL_JD_REQUIREMENTS = [
    "Currently pursuing a Bachelor's degree in Data Science, Computer Science, "
    "Statistics, Mathematics, or a related field.",
    "Strong understanding of Python",
    "Familiarity with pandas and NumPy",
    "Understanding of basic machine learning concepts",
    "Experience with data visualization using tools such as Matplotlib or "
    "Seaborn",
    "Good analytical and problem-solving skills",
    "Ability to communicate technical concepts clearly",
    "Experience with scikit-learn",
    "Familiarity with Git and GitHub",
    "Experience working on academic or personal data science projects",
    "Basic understanding of APIs",
    "Previous experience working with large datasets",
    "Basic knowledge of SQL",
]


@pytest.fixture
def real_cv() -> ParsedCV:
    return ParsedCV(
        candidate_name=None,
        raw_text="\n".join(REAL_CV_BULLETS),
        evidence=[CVEvidence(text=text, section=None) for text in REAL_CV_BULLETS],
    )


@pytest.fixture
def real_job() -> JobDescription:
    return JobDescription(
        role="Data Science Intern",
        company="Nexora Analytics",
        raw_text="...",
        requirements=[RoleRequirement(text=text) for text in REAL_JD_REQUIREMENTS],
    )


def judged_reply(levels: dict[int, tuple[str, int | None, str]]) -> str:
    """A model reply built directly from {requirement_index: (level,
    evidence_index, reason)} - the shape a correctly-reasoning judgement
    actually has, including the level-specific reason the report shows."""
    return json.dumps(
        {
            "judgements": [
                {
                    "requirement_index": index,
                    "match_level": level,
                    "evidence_index": evidence_index,
                    "reason": reason,
                }
                for index, (level, evidence_index, reason) in levels.items()
            ],
            "recommended_emphasis": [],
        }
    )


def match_at(report: FitReport, index: int):
    return report.requirement_matches[index]


# Test 1 - the exact live failure: "BS" must satisfy "Bachelor's degree".


def test_a_bs_degree_matches_a_bachelors_degree_requirement(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {0: ("match", 0, "The candidate is a BS Data Science student.")}
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 0)
    assert match.match_level == "match"
    assert match.met is True
    assert match.evidence == REAL_CV_BULLETS[0]


# Test 2 - "basic knowledge" / "an online course" is not "strong understanding".


def test_basic_python_knowledge_is_partial_not_a_full_match(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {
            1: (
                "partial",
                3,
                "The CV shows only basic Python knowledge and an online "
                "course, not the strong proficiency the role asks for.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 1)
    assert match.match_level == "partial"
    assert match.met is False  # partial is real evidence, but not a full match
    assert match.evidence == REAL_CV_BULLETS[3]
    assert "basic" in match.reason.lower()


# Test 3 - the same "basic knowledge" line fully satisfies a "basic" requirement.


def test_basic_sql_knowledge_matches_a_basic_sql_requirement(
    make_router, real_cv, real_job
):
    reply = judged_reply({12: ("match", 3, "Basic SQL knowledge is stated.")})
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 12)
    assert match.match_level == "match"
    assert match.met is True


# Test 4 & 5 - genuinely absent tools/concepts stay missing, never invented.


@pytest.mark.parametrize("index", [2, 3, 7])  # pandas/NumPy, ML, scikit-learn
def test_absent_tools_are_reported_missing_not_invented(
    make_router, real_cv, real_job, index
):
    # No judgement at all for these - exactly what happens when nothing in the
    # CV evidence gives the model anything to point at.
    report = score_with(make_router, real_cv, real_job, judged_reply({}))

    match = match_at(report, index)
    assert match.match_level == "missing"
    assert match.evidence == ""
    assert match.requirement in report.gaps


# Test 6 - coursework is related to analytical skills, not proof of them.


def test_coursework_is_related_evidence_for_analytical_skills_not_a_match(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {
            5: (
                "related",
                5,
                "Coursework in Data Structures and Calculus is relevant but "
                "does not itself demonstrate professional analytical "
                "experience.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 5)
    assert match.match_level == "related"
    assert match.met is False
    assert match.requirement not in report.gaps  # real evidence, not a gap


# Test 7 - the CV's own communication line matches a general requirement.


def test_stated_communication_ability_matches_a_general_requirement(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {6: ("match", 4, "The CV states exceptional communication ability.")}
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 6)
    assert match.match_level == "match"
    assert match.evidence == REAL_CV_BULLETS[4]


# Test 8 - a programming project is not automatically a data-science project.


def test_a_non_data_science_project_is_related_not_a_direct_match(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {
            9: (
                "related",
                2,
                "Image processing and a console game are programming "
                "projects, but the CV does not establish they are data "
                "science projects.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 9)
    assert match.match_level == "related"
    assert match.met is False
    assert match.requirement not in report.gaps


# --- The weighted score itself ------------------------------------------------


def test_compute_fit_score_weighs_each_match_level_as_specified():
    must = RoleRequirement(text="a")

    assert compute_fit_score([(must, 1.0)]) == 1.0  # match
    assert compute_fit_score([(must, 0.5)]) == 0.5  # partial
    assert compute_fit_score([(must, 0.25)]) == 0.25  # related
    assert compute_fit_score([(must, 0.0)]) == 0.0  # missing


def test_partial_and_related_earn_less_than_a_full_match_but_more_than_missing(
    make_router, real_cv, real_job
):
    # Same requirement, three different strengths - the score must order them.
    match_report = score_with(
        make_router, real_cv, real_job, judged_reply({0: ("match", 0, "")})
    )
    partial_report = score_with(
        make_router, real_cv, real_job, judged_reply({0: ("partial", 0, "")})
    )
    related_report = score_with(
        make_router, real_cv, real_job, judged_reply({0: ("related", 0, "")})
    )
    missing_report = score_with(make_router, real_cv, real_job, judged_reply({}))

    assert (
        match_report.overall_fit
        > partial_report.overall_fit
        > related_report.overall_fit
        > missing_report.overall_fit
    )


# --- Grounding still holds: a claimed level without real evidence is downgraded


@pytest.mark.parametrize("level", ["match", "partial", "related"])
def test_any_non_missing_level_without_real_evidence_is_downgraded_to_missing(
    make_router, real_cv, real_job, level
):
    collector = TraceCollector()
    # Evidence index 99 does not exist - the model cannot earn any credit,
    # regardless of which level it claims, by pointing at nothing real. This
    # is the same protection the old met/unmet code had, now covering all
    # three non-missing levels instead of just one.
    reply = judged_reply({0: (level, 99, "claims support that does not exist")})
    report = score_with(make_router, real_cv, real_job, reply, collector)

    match = match_at(report, 0)
    assert match.match_level == "missing"
    assert match.met is False
    assert match.evidence == ""
    assert match.requirement in report.gaps
    guard = [e for e in collector.events if e.name == "scoring:evidence_guard"]
    assert guard and "unsupported match" in guard[0].result


def test_only_missing_requirements_are_gaps_partial_and_related_are_not(
    make_router, real_cv, real_job
):
    reply = judged_reply(
        {
            0: ("match", 0, ""),
            1: ("partial", 3, ""),
            5: ("related", 5, ""),
            2: ("missing", None, ""),
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    assert REAL_JD_REQUIREMENTS[2] in report.gaps  # only the genuinely missing one
    assert REAL_JD_REQUIREMENTS[1] not in report.gaps  # partial - real evidence
    assert REAL_JD_REQUIREMENTS[5] not in report.gaps  # related - real evidence
    assert REAL_JD_REQUIREMENTS[0] not in report.gaps  # a full match either


# --- Regression: related/partial require genuinely topic-adjacent evidence ---
#
# A live run (this exact CV and JD) still scored these three requirements too
# generously even after the four-level rubric landed:
#   - pandas/NumPy -> "related", citing "C++ programmer... Image Processing...
#     Console games" - general programming ability, not pandas or NumPy.
#   - basic ML concepts -> "partial", citing "Basic knowledge on SQL and
#     Python" - a different topic, not a weaker version of ML.
#   - basic understanding of APIs -> "partial", citing the same SQL/Python
#     line, with a reason that admitted "not the basic understanding of APIs
#     the role asks for" while still scoring it partial - self-contradictory.
# The fix is the rubric's own text (SCORING_SYSTEM_PROMPT), not new code -
# the model judges, the code only scores and downgrades. These tests pin the
# corrective language in place, and prove the *system* reports the
# now-correct judgement properly for these exact three requirement/evidence
# pairs - they cannot force a live model's behaviour, which is why the prompt
# assertions come first.


def test_the_rubric_explicitly_warns_against_the_three_live_over_generous_calls():
    prompt = SCORING_SYSTEM_PROMPT

    # pandas/NumPy: general programming must not count as related.
    assert "pandas and NumPy" in prompt
    assert "general programming ability in a different language" in prompt
    # ML concepts and APIs: SQL/Python must not count as partial for either.
    assert '"machine learning concepts"' in prompt
    assert '"APIs"' in prompt
    assert "different topics from SQL and Python" in prompt
    # The self-contradiction rule: a reason admitting non-support means missing.
    assert "reason that admits the evidence does not establish" in prompt


def test_general_programming_evidence_is_not_related_to_named_libraries(
    make_router, real_cv, real_job
):
    # The exact live case: pandas/NumPy (requirement 2) judged correctly as
    # missing rather than "related" to the C++/image-processing evidence.
    reply = judged_reply(
        {
            2: (
                "missing",
                None,
                "The CV shows general programming ability but never mentions "
                "pandas or NumPy specifically.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 2)
    assert match.match_level == "missing"
    assert match.evidence == ""
    assert match.requirement in report.gaps


def test_sql_and_python_evidence_is_not_partial_for_machine_learning(
    make_router, real_cv, real_job
):
    # The exact live case: ML concepts (requirement 3) judged correctly as
    # missing rather than "partial" toward the SQL/Python line.
    reply = judged_reply(
        {
            3: (
                "missing",
                None,
                "The CV mentions basic SQL and Python but nothing about "
                "machine learning concepts.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 3)
    assert match.match_level == "missing"
    assert match.requirement in report.gaps


def test_sql_evidence_is_not_partial_for_api_understanding(
    make_router, real_cv, real_job
):
    # The exact live case: APIs (requirement 10) judged correctly as missing
    # rather than "partial" toward the same SQL/Python line, whose own reason
    # in the live run had already admitted it did not establish API knowledge.
    reply = judged_reply(
        {
            10: (
                "missing",
                None,
                "The CV mentions basic SQL knowledge, which does not "
                "establish an understanding of APIs.",
            )
        }
    )
    report = score_with(make_router, real_cv, real_job, reply)

    match = match_at(report, 10)
    assert match.match_level == "missing"
    assert match.requirement in report.gaps


# --- Regression: subject relevance before proficiency ------------------------
#
# A second live run hit a *new* pair the previous fix never named: "Basic
# knowledge on CSS and HTML" was judged "partial" evidence for "Basic
# understanding of APIs" - the model's own reason already said "which is not
# basic understanding of APIs", the same self-contradiction pattern as the
# SQL/APIs case, just with a pair the prompt hadn't enumerated. Listing bad
# pairs teaches examples, not the rule; the fix is the general principle the
# prompt now leads with, so it is not dependent on which specific pair shows
# up next.


def test_the_rubric_leads_with_subject_relevance_before_proficiency():
    prompt = SCORING_SYSTEM_PROMPT

    assert "Check subject relevance BEFORE proficiency level" in prompt
    assert "CSS/HTML" in prompt
    assert "different subjects" in prompt


def test_css_html_evidence_is_not_partial_for_api_understanding(
    make_router, real_job
):
    # The exact second live failure: a CV whose only web-adjacent evidence is
    # CSS/HTML must not let "Basic understanding of APIs" become partial.
    cv = ParsedCV(
        raw_text="Basic knowledge on CSS and HTML.",
        evidence=[CVEvidence(text="Basic knowledge on CSS and HTML.")],
    )
    reply = judged_reply(
        {
            10: (
                "missing",
                None,
                "The CV shows basic CSS/HTML, a different subject from APIs.",
            )
        }
    )
    report = score_with(make_router, cv, real_job, reply)

    match = match_at(report, 10)
    assert match.match_level == "missing"
    assert match.requirement in report.gaps

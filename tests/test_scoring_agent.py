"""Scoring agent (FR-4, US-3).

The model is a stub returning judgements, so these tests check what the agent
is actually responsible for: the rubric, the evidence rules, and the shape of
the FitReport - not a model's judgement quality. No test makes an API call.
"""

import json

import pytest

from job_agent.agents import ScoringAgent, compute_fit_score
from job_agent.agents.scoring import build_scoring_prompt
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


def judgement(met_pairs: dict[int, int | None], emphasis: list[int] | None = None) -> str:
    """A model reply: {requirement_index: evidence_index or None}."""
    return json.dumps(
        {
            "judgements": [
                {
                    "requirement_index": requirement,
                    "met": evidence is not None,
                    "evidence_index": evidence,
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
        {"requirement_index": 77, "met": True, "evidence_index": 0}
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

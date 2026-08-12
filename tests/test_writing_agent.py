"""Writing agent (FR-5, FR-6, US-4, US-5).

The model is the existing stub, so these tests are about the agent's contract:
schema validation and grounding both have to pass, a rejected draft is retried
with the violations explained, and material that cannot be vouched for is never
returned.
"""

import json

import pytest

from job_agent.agents import WritingAgent
from job_agent.agents.writing import WRITING_SYSTEM_PROMPT, build_writing_prompt
from job_agent.memory.session import RunContext
from job_agent.models import (
    CompanyBrief,
    CoverLetter,
    CVEvidence,
    FitReport,
    JobDescription,
    ParsedCV,
    RequirementMatch,
    RoleRequirement,
    Source,
    TailoredCV,
)
from job_agent.observability import TraceCollector
from job_agent.validation import RetryExhaustedError

EVIDENCE = [
    "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite.",
    "Built a research agent with tool calling, session memory and hooks.",
    "Task Manager API: JWT authentication, per-user task authorization.",
]

GOOD_DRAFT = {
    "bullets": [
        "Built a Task Management REST API with FastAPI, SQLAlchemy and SQLite.",
        "Delivered JWT authentication and per-user task authorization.",
        "Built a research agent with tool calling and session memory.",
    ],
    "omitted": ["BS Computer Science"],
    "cover_letter": (
        "I am writing about the Junior AI Engineer role at Arbisoft. "
        "I built a Task Management REST API with FastAPI and SQLite. "
        "I also built a research agent with tool calling and session memory."
    ),
}

FABRICATED_DRAFT = {
    **GOOD_DRAFT,
    "bullets": [
        *GOOD_DRAFT["bullets"],
        "Deployed the platform to Kubernetes with 5 years of production experience.",
    ],
}


@pytest.fixture
def cv() -> ParsedCV:
    return ParsedCV(
        candidate_name="Mahnoor Rauf",
        raw_text="\n".join([*EVIDENCE, "BS Computer Science"]),
        evidence=[CVEvidence(text=text, section="Experience") for text in EVIDENCE],
        skills=["Python", "FastAPI", "pytest", "SQLite"],
    )


@pytest.fixture
def job() -> JobDescription:
    return JobDescription(
        role="Junior AI Engineer",
        company="Arbisoft",
        raw_text="...",
        requirements=[
            RoleRequirement(text="Strong Python"),
            RoleRequirement(text="Kubernetes experience", must_have=False),
        ],
        responsibilities=["Build LLM-backed product features"],
        skills=["Python", "Kubernetes"],
    )


@pytest.fixture
def report() -> FitReport:
    return FitReport(
        role="Junior AI Engineer",
        company="Arbisoft",
        overall_fit=0.8,
        requirement_matches=[
            RequirementMatch(requirement="Strong Python", evidence=EVIDENCE[0], met=True)
        ],
        gaps=["Kubernetes experience"],
        recommended_emphasis=[EVIDENCE[0], EVIDENCE[1]],
    )


@pytest.fixture
def brief() -> CompanyBrief:
    return CompanyBrief(
        company="Arbisoft",
        summary="Arbisoft builds data platforms.",
        facts=["Builds data platforms."],
        sources=[Source(title="Arbisoft", url="https://arbisoft.com")],
    )


def agent_for(make_router, replies, collector=None, **kwargs) -> WritingAgent:
    router = make_router(heavy_replies=replies)
    return WritingAgent(router, collector=collector, **kwargs)


# --- Happy path --------------------------------------------------------------


def test_a_grounded_draft_becomes_a_tailored_cv_and_cover_letter(
    make_router, cv, job, report, brief
):
    agent = agent_for(make_router, [json.dumps(GOOD_DRAFT)])

    tailored, letter = agent.write(cv, job, report, brief)

    assert isinstance(tailored, TailoredCV) and isinstance(letter, CoverLetter)
    assert TailoredCV.model_validate(tailored.model_dump()) == tailored
    assert CoverLetter.model_validate(letter.model_dump()) == letter
    assert len(tailored.bullets) == 3
    assert tailored.omitted == ["BS Computer Science"]
    assert "Junior AI Engineer" in letter.body


def test_role_and_company_come_from_the_posting_not_the_model(
    make_router, cv, job, report
):
    agent = agent_for(make_router, [json.dumps(GOOD_DRAFT)])

    tailored, letter = agent.write(cv, job, report)

    assert (tailored.role, tailored.company) == ("Junior AI Engineer", "Arbisoft")
    assert (letter.role, letter.company) == ("Junior AI Engineer", "Arbisoft")


def test_reworded_and_reordered_evidence_is_accepted(make_router, cv, job, report):
    reworded = {
        "bullets": [
            "Delivered a research agent featuring tool calling and session memory.",
            "Engineered a task management REST API on FastAPI, SQLAlchemy and SQLite.",
        ],
        "omitted": [],
        "cover_letter": "I built a task management API with FastAPI.",
    }
    agent = agent_for(make_router, [json.dumps(reworded)])

    tailored, _ = agent.write(cv, job, report)

    # Neither bullet appears verbatim in the CV, and both are still accepted.
    assert len(tailored.bullets) == 2
    assert not any(b in cv.raw_text for b in tailored.bullets)


def test_the_writing_runs_on_the_heavy_tier(make_router, cv, job, report):
    router = make_router(heavy_replies=[json.dumps(GOOD_DRAFT)])

    WritingAgent(router).write(cv, job, report)

    assert len(router.client("heavy").calls) == 1
    assert router.client("light").calls == []


# --- The prompt --------------------------------------------------------------


def test_the_prompt_supplies_evidence_emphasis_gaps_and_company_facts(
    cv, job, report, brief
):
    prompt = build_writing_prompt(cv, job, report, brief)

    assert "CV EVIDENCE" in prompt and EVIDENCE[0] in prompt
    assert "WORTH FOREGROUNDING" in prompt
    assert "KNOWN GAPS" in prompt and "Kubernetes experience" in prompt
    assert "Builds data platforms." in prompt


def test_the_prompt_names_the_candidate_so_the_letter_can_be_signed(cv, job, report):
    # The live rehearsal produced "Sincerely, [Candidate]", which grounding
    # then rejected. The fix is upstream: give the model the real name and
    # forbid placeholders, rather than teaching grounding to accept them.
    prompt = build_writing_prompt(cv, job, report)

    assert "CANDIDATE NAME: Mahnoor Rauf" in prompt
    assert "[Candidate]" in WRITING_SYSTEM_PROMPT  # named as forbidden
    assert "Never write a template placeholder" in WRITING_SYSTEM_PROMPT


def test_the_prompt_refuses_to_invent_a_name_when_the_cv_has_none(job, report):
    anonymous = ParsedCV(raw_text="Built a FastAPI service.", evidence=[])

    prompt = build_writing_prompt(anonymous, job, report)

    assert "do not invent one" in prompt


def test_a_letter_signed_with_the_real_name_survives_grounding(
    make_router, cv, job, report
):
    signed = {
        **GOOD_DRAFT,
        "cover_letter": (
            "I built a research agent with tool calling and session memory. "
            "Sincerely, Mahnoor Rauf"
        ),
    }
    agent = agent_for(make_router, [json.dumps(signed)])

    _, letter = agent.write(cv, job, report)

    assert "Mahnoor Rauf" in letter.body


def test_a_plural_reference_to_real_cv_content_survives_grounding(
    make_router, cv, job, report
):
    # The exact draft the live rehearsal threw away three times.
    plural = {
        **GOOD_DRAFT,
        "bullets": ["Built REST APIs with FastAPI, SQLAlchemy and SQLite."],
        "cover_letter": "I have experience building REST APIs with FastAPI.",
    }
    agent = agent_for(make_router, [json.dumps(plural)])

    tailored, letter = agent.write(cv, job, report)

    assert tailored.bullets == ["Built REST APIs with FastAPI, SQLAlchemy and SQLite."]
    assert "REST APIs" in letter.body


def test_the_prompt_is_honest_when_there_is_no_company_research(cv, job, report):
    prompt = build_writing_prompt(cv, job, report, brief=None)

    assert "no company research available" in prompt


# --- Fabrication is rejected -------------------------------------------------


def test_an_ungrounded_draft_is_rejected_and_retried_with_the_violations(
    make_router, cv, job, report
):
    collector = TraceCollector()
    agent = agent_for(
        make_router,
        [json.dumps(FABRICATED_DRAFT), json.dumps(GOOD_DRAFT)],
        collector=collector,
    )

    tailored, _ = agent.write(cv, job, report)

    assert not any("Kubernetes" in bullet for bullet in tailored.bullets)
    # The retry told the model exactly what was wrong.
    second_prompt = agent.router.client("heavy").calls[1]["messages"][0]["content"]
    assert "not supported by the CV evidence" in second_prompt
    assert "Kubernetes" in second_prompt
    assert "5" in second_prompt


def test_schema_valid_but_ungrounded_output_never_reaches_the_caller(
    make_router, cv, job, report
):
    # Perfectly shaped JSON every time - and fabricated every time.
    agent = agent_for(make_router, [json.dumps(FABRICATED_DRAFT)] * 3)

    with pytest.raises(RetryExhaustedError) as exc:
        agent.write(cv, job, report)

    assert "unsupported claims remained" in str(exc.value)
    assert exc.value.schema_name == "WritingDraft"


def test_a_fabricated_cover_letter_sentence_is_caught(make_router, cv, job, report):
    letter_draft = {
        **GOOD_DRAFT,
        "cover_letter": (
            "I built a research agent with tool calling. "
            "I also hold an AWS Certified Solutions Architect qualification."
        ),
    }
    agent = agent_for(make_router, [json.dumps(letter_draft)] * 3)

    with pytest.raises(RetryExhaustedError) as exc:
        agent.write(cv, job, report)

    assert "AWS" in str(exc.value)


def test_the_posting_cannot_license_a_claim_the_cv_does_not_support(
    make_router, cv, job, report
):
    # The posting lists Kubernetes; that must not make claiming it acceptable.
    smuggled = {
        **GOOD_DRAFT,
        "bullets": ["Experienced with Kubernetes, as the role requires."],
    }
    agent = agent_for(make_router, [json.dumps(smuggled)] * 3)

    with pytest.raises(RetryExhaustedError):
        agent.write(cv, job, report)


def test_the_grounding_attempt_budget_is_bounded_and_respected(
    make_router, cv, job, report
):
    agent = agent_for(
        make_router, [json.dumps(FABRICATED_DRAFT)] * 2, grounding_attempts=2
    )

    with pytest.raises(RetryExhaustedError) as exc:
        agent.write(cv, job, report)

    assert exc.value.attempts == 2
    assert len(agent.router.client("heavy").calls) == 2


# --- Schema failures still use the existing guard ----------------------------


def test_malformed_output_is_retried_by_the_existing_schema_guard(
    make_router, cv, job, report
):
    agent = agent_for(
        make_router, ["here is your cover letter!", json.dumps(GOOD_DRAFT)]
    )

    tailored, _ = agent.write(cv, job, report)

    assert len(tailored.bullets) == 3
    second_prompt = agent.router.client("heavy").calls[1]["messages"][0]["content"]
    assert "rejected by the schema validator" in second_prompt


def test_persistently_malformed_output_raises(make_router, cv, job, report):
    agent = agent_for(make_router, ["nope", "nope", "nope"])

    with pytest.raises(RetryExhaustedError):
        agent.write(cv, job, report)


# --- Tracing -----------------------------------------------------------------


def test_the_model_call_is_traced(make_router, cv, job, report):
    collector = TraceCollector()
    agent_for(make_router, [json.dumps(GOOD_DRAFT)], collector=collector).write(
        cv, job, report
    )

    (event,) = collector.events
    assert event.kind == "model"
    assert event.name == "model:heavy:tailor_cv"
    assert event.agent == "writing"
    assert (event.input_tokens, event.output_tokens) == (10, 5)


def test_grounding_failures_are_traced(make_router, cv, job, report):
    collector = TraceCollector()
    agent = agent_for(
        make_router,
        [json.dumps(FABRICATED_DRAFT), json.dumps(GOOD_DRAFT)],
        collector=collector,
    )

    agent.write(cv, job, report)

    guards = [e for e in collector.events if e.name == "writing:grounding_guard"]
    assert len(guards) == 1  # the rejected first attempt, not the accepted second
    assert guards[0].arguments["unsupported"] == 1
    assert "Kubernetes" in guards[0].result


# --- Optional model-assisted verification ------------------------------------


def test_the_entailment_pass_can_reject_a_claim_the_checker_allowed(
    make_router, cv, job, report
):
    # "Led a team" carries no distinctive term or number, so only the model can
    # catch it. It is rejected, and the retry then produces clean material.
    subtle = {**GOOD_DRAFT, "bullets": ["Led a team of engineers."]}
    verdict = json.dumps(
        {"verdicts": [{"claim_index": 0, "supported": False, "reason": "not in the CV"}]}
    )
    clean_verdicts = json.dumps({"verdicts": []})
    agent = agent_for(
        make_router,
        [json.dumps(subtle), verdict, json.dumps(GOOD_DRAFT), clean_verdicts],
        verify_with_model=True,
    )

    tailored, _ = agent.write(cv, job, report)

    assert tailored.bullets == GOOD_DRAFT["bullets"]


def test_the_entailment_pass_is_off_by_default(make_router, cv, job, report):
    subtle = {**GOOD_DRAFT, "bullets": ["Led a team of engineers."]}
    agent = agent_for(make_router, [json.dumps(subtle)])

    tailored, _ = agent.write(cv, job, report)

    # One model call only - no verification round trip.
    assert len(agent.router.client("heavy").calls) == 1
    assert tailored.bullets == ["Led a team of engineers."]


# --- RunContext integration --------------------------------------------------


def test_run_writes_from_the_run_context(make_router, cv, job, report, brief):
    context = RunContext(cv_text="cv", jd_text="jd")
    context.parsed_cv, context.job, context.fit_report, context.brief = (
        cv,
        job,
        report,
        brief,
    )
    agent = agent_for(make_router, [json.dumps(GOOD_DRAFT)])

    tailored, letter = agent.run(context)

    assert tailored.company == "Arbisoft"
    assert letter.body


def test_run_refuses_without_a_fit_report(make_router, cv, job):
    context = RunContext(cv_text="cv", jd_text="jd")
    context.parsed_cv, context.job = cv, job
    agent = agent_for(make_router, [json.dumps(GOOD_DRAFT)])

    with pytest.raises(ValueError, match="fit report"):
        agent.run(context)


def test_run_refuses_without_parsed_inputs(make_router):
    agent = agent_for(make_router, [json.dumps(GOOD_DRAFT)])

    with pytest.raises(ValueError, match="parsed CV"):
        agent.run(RunContext(cv_text="cv", jd_text="jd"))

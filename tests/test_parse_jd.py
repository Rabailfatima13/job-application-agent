"""parse_jd against the fixture posting (FR-1, FR-3).

The model is a stub replaying a realistic extraction, so these tests check the
parsing *layer* - schema conformance, requirement/responsibility separation,
raw-text preservation, routing, retry - rather than a model's extraction skill,
and they cost nothing to run.
"""

import json

import pytest

from job_agent.models import JobDescription
from job_agent.tools.parsing import JD_MAX_TOKENS, UNKNOWN_COMPANY, UNKNOWN_ROLE, parse_jd
from job_agent.validation import RetryExhaustedError

FIXTURE_EXTRACTION = {
    "role": "Junior AI Engineer",
    "company": "Arbisoft",
    "location": "Lahore, hybrid",
    "requirements": [
        {"text": "Strong Python", "must_have": True},
        {"text": "Experience building REST APIs (FastAPI or similar)", "must_have": True},
        {"text": "Familiarity with LLM APIs and prompt engineering", "must_have": True},
        {"text": "Testing discipline (pytest)", "must_have": True},
        {"text": "Comfortable with SQL databases", "must_have": True},
        {
            "text": "Experience with agent frameworks or the Model Context Protocol",
            "must_have": False,
        },
        {"text": "Kubernetes / cloud deployment experience", "must_have": False},
        {"text": "Front-end familiarity (React)", "must_have": False},
    ],
    "responsibilities": ["Help build LLM-backed product features"],
    "skills": ["Python", "FastAPI", "pytest", "SQL", "Kubernetes", "React"],
}


@pytest.fixture
def parsed(make_router, sample_jd_text) -> JobDescription:
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])
    return parse_jd(sample_jd_text, router)


def test_role_and_company_are_extracted(parsed):
    assert parsed.role == "Junior AI Engineer"
    assert parsed.company == "Arbisoft"
    assert parsed.location == "Lahore, hybrid"


def test_requirements_keep_the_must_have_distinction(parsed):
    must_have = [r.text for r in parsed.requirements if r.must_have]
    nice_to_have = [r.text for r in parsed.requirements if not r.must_have]

    assert "Strong Python" in must_have
    assert len(must_have) == 5
    # "Nice to have" items must not be scored as hard requirements.
    assert "Kubernetes / cloud deployment experience" in nice_to_have


def test_responsibilities_are_kept_separate_from_requirements(parsed):
    assert parsed.responsibilities == ["Help build LLM-backed product features"]
    requirement_texts = [r.text for r in parsed.requirements]
    assert "Help build LLM-backed product features" not in requirement_texts


def test_named_technologies_are_represented(parsed):
    assert {"Python", "FastAPI", "pytest"} <= set(parsed.skills)


def test_no_requirement_is_absent_from_the_posting(parsed, sample_jd_text):
    # Every extracted requirement must be traceable to the source posting.
    source = sample_jd_text.lower()
    for requirement in parsed.requirements:
        head = requirement.text.split("(")[0].strip().lower()
        assert head in source, f"fabricated requirement: {requirement.text}"


def test_raw_text_is_the_source_verbatim(parsed, sample_jd_text):
    # raw_text comes from the file, not from the model's reply.
    assert parsed.raw_text == sample_jd_text.strip()


def test_result_is_a_validated_job_description(parsed):
    assert isinstance(parsed, JobDescription)
    assert JobDescription.model_validate(parsed.model_dump()) == parsed


def test_parsing_runs_on_the_cheap_tier(make_router, sample_jd_text):
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])
    parse_jd(sample_jd_text, router)

    light, heavy = router.for_step("parse_jd"), router.client("heavy")
    assert light.name == "stub-light"
    assert len(light.calls) == 1
    assert heavy.calls == []  # the expensive model is never touched by parsing


def test_the_request_stays_under_groqs_output_token_rate_limit(
    make_router, sample_jd_text
):
    """Regression: a live run against qwen/qwen3.8-27b (the light-tier
    model) was rejected outright by Groq's 1000 output-tokens-per-minute
    cap - the request fails before it runs at all once max_tokens reaches
    1000, regardless of the actual reply size. JD_MAX_TOKENS must stay
    safely under that ceiling."""
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])

    parse_jd(sample_jd_text, router)

    assert JD_MAX_TOKENS < 1000
    calls = router.for_step("parse_jd").calls
    assert calls[0]["max_tokens"] == JD_MAX_TOKENS


def test_a_posting_without_a_company_is_not_guessed(make_router):
    reply = json.dumps({**FIXTURE_EXTRACTION, "company": None})
    router = make_router(light_replies=[reply])

    parsed = parse_jd("Junior AI Engineer\nRequirements: Python", router)

    assert parsed.company == UNKNOWN_COMPANY


def test_a_posting_without_a_clear_title_does_not_crash_the_run(make_router):
    """Regression: a live run had the extractor honestly return role=null
    for a terse posting. `JDExtraction.role` used to require a string, so
    that null failed schema validation on every one of the 3 retries and
    `RetryExhaustedError` crashed the whole analysis with a raw pydantic
    error on screen - instead of degrading the same way a missing company
    already does."""
    reply = json.dumps({**FIXTURE_EXTRACTION, "role": None})
    router = make_router(light_replies=[reply])

    parsed = parse_jd("Requirements: Python, SQL", router)

    assert parsed.role == UNKNOWN_ROLE


def test_a_null_role_is_accepted_on_the_first_attempt_not_retried(make_router):
    """The schema itself must accept null - not merely survive it after
    retries are exhausted - so a single clean reply is never rejected."""
    reply = json.dumps({**FIXTURE_EXTRACTION, "role": None})
    router = make_router(light_replies=[reply])

    parse_jd("Requirements: Python, SQL", router)

    assert len(router.for_step("parse_jd").calls) == 1


def test_malformed_output_is_retried_then_accepted(make_router, sample_jd_text):
    router = make_router(
        light_replies=[
            "sorry, here are the details in prose",
            json.dumps(FIXTURE_EXTRACTION),
        ]
    )

    parsed = parse_jd(sample_jd_text, router)

    assert parsed.role == "Junior AI Engineer"
    stub = router.for_step("parse_jd")
    assert len(stub.calls) == 2
    # The retry carries the validation error back to the model.
    assert "rejected by the schema validator" in stub.calls[1]["messages"][0]["content"]


def test_persistently_bad_output_never_reaches_the_caller(make_router, sample_jd_text):
    router = make_router(light_replies=["nope", "still nope", "nope again"])

    with pytest.raises(RetryExhaustedError):
        parse_jd(sample_jd_text, router)

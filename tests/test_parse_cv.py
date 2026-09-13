"""parse_cv against the fixture CV (FR-1, FR-3).

The load-bearing tests here are the wording ones: everything Week 7's
no-fabrication check compares against is produced by this parser, so evidence
that is not the candidate's own text must not survive parsing.
"""

import json

import pytest

from job_agent.models import ParsedCV
from job_agent.observability import TraceCollector
from job_agent.tools.parsing import CV_MAX_TOKENS, is_supported_by, parse_cv

FIXTURE_EXTRACTION = {
    "candidate_name": "Mahnoor Rauf",
    "evidence": [
        {
            "text": (
                "Built a Task Management REST API with FastAPI, SQLAlchemy and "
                "SQLite, with Pydantic validation and pytest coverage."
            ),
            "section": "Experience",
            "skills": ["FastAPI", "SQLAlchemy", "SQLite", "Pydantic", "pytest"],
        },
        {
            "text": (
                "Built a research agent with tool calling, session memory and "
                "pre/post-tool hooks."
            ),
            "section": "Experience",
            "skills": ["tool calling"],
        },
        {
            "text": (
                "Task Manager API: JWT authentication, per-user task "
                "authorization, integration tests."
            ),
            "section": "Projects",
            "skills": ["JWT"],
        },
        {"text": "BS Computer Science", "section": "Education", "skills": []},
    ],
    "skills": [
        "Python",
        "FastAPI",
        "SQLAlchemy",
        "SQLite",
        "Pydantic",
        "pytest",
        "Docker",
        "MCP",
    ],
}


@pytest.fixture
def parsed(make_router, sample_cv_text) -> ParsedCV:
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])
    return parse_cv(sample_cv_text, router)


def test_candidate_and_skills_are_extracted(parsed):
    assert parsed.candidate_name == "Mahnoor Rauf"
    assert {"Python", "FastAPI", "pytest"} <= set(parsed.skills)


def test_experience_projects_and_education_are_all_captured(parsed):
    sections = {item.section for item in parsed.evidence}
    assert {"Experience", "Projects", "Education"} <= sections


def test_evidence_text_is_the_candidate_s_own_wording(parsed, sample_cv_text):
    for item in parsed.evidence:
        assert is_supported_by(item.text, sample_cv_text), (
            f"evidence not present in the source CV: {item.text}"
        )


def test_raw_text_is_the_source_verbatim(parsed, sample_cv_text):
    assert parsed.raw_text == sample_cv_text.strip()


def test_paraphrased_or_invented_evidence_is_dropped(make_router, sample_cv_text):
    embellished = {
        **FIXTURE_EXTRACTION,
        "evidence": [
            *FIXTURE_EXTRACTION["evidence"],
            # Never written in the CV - the classic fabrication.
            {"text": "5 years of professional Kubernetes experience.", "section": None,
             "skills": ["Kubernetes"]},
            # A rewrite of a real bullet: still not the candidate's wording.
            {"text": "Led the design of a large-scale task platform.",
             "section": "Experience", "skills": []},
        ],
    }
    collector = TraceCollector()
    router = make_router(light_replies=[json.dumps(embellished)])

    parsed = parse_cv(sample_cv_text, router, collector=collector)

    texts = [item.text for item in parsed.evidence]
    assert not any("Kubernetes" in t for t in texts)
    assert not any("Led the design" in t for t in texts)
    assert len(texts) == len(FIXTURE_EXTRACTION["evidence"])
    # The drop is recorded rather than silent.
    guard_events = [e for e in collector.events if e.name == "parse_cv:verbatim_guard"]
    assert guard_events and guard_events[0].arguments["dropped"] == 2


def test_bullet_markers_and_spacing_do_not_count_as_rewriting():
    source = "EXPERIENCE\n- Built  a research agent with tool calling.\n"
    assert is_supported_by("Built a research agent with tool calling.", source)
    assert not is_supported_by("Built a recommendation engine.", source)


def test_result_is_a_validated_parsed_cv(parsed):
    assert isinstance(parsed, ParsedCV)
    assert ParsedCV.model_validate(parsed.model_dump()) == parsed


def test_parsing_runs_on_the_cheap_tier_and_is_traced(make_router, sample_cv_text):
    collector = TraceCollector()
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])

    parse_cv(sample_cv_text, router, collector=collector)

    assert router.client("heavy").calls == []
    (event,) = collector.events
    assert event.kind == "model"
    assert event.name == "model:light:parse_cv"
    assert event.input_tokens == 10 and event.output_tokens == 5


def test_the_request_stays_under_groqs_output_token_rate_limit(
    make_router, sample_cv_text
):
    """Regression: a live run against qwen/qwen3.8-27b (the light-tier
    model) was rejected outright by Groq's 1000 output-tokens-per-minute
    cap - the request fails before it runs at all once max_tokens reaches
    1000, regardless of the actual reply size. CV_MAX_TOKENS must stay
    safely under that ceiling."""
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])

    parse_cv(sample_cv_text, router)

    assert CV_MAX_TOKENS < 1000
    calls = router.for_step("parse_cv").calls
    assert calls[0]["max_tokens"] == CV_MAX_TOKENS


def test_reads_a_cv_from_a_file_path(make_router, fixtures_dir, sample_cv_text):
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])

    parsed = parse_cv(fixtures_dir / "sample_cv.txt", router)

    assert parsed.raw_text == sample_cv_text.strip()


# --- caching (efficiency: the same baseline CV, re-parsed for nothing on ----
# every single analysis a user runs against it, unless a cache is supplied) --


def test_omitting_the_cache_reproduces_the_exact_pre_caching_behaviour(
    make_router, sample_cv_text
):
    router = make_router(
        light_replies=[json.dumps(FIXTURE_EXTRACTION), json.dumps(FIXTURE_EXTRACTION)]
    )

    parse_cv(sample_cv_text, router)
    parse_cv(sample_cv_text, router)

    # No cache supplied - every call parses fresh, exactly as before caching
    # existed.
    assert len(router.for_step("parse_cv").calls) == 2


def test_a_second_call_with_the_same_text_is_a_cache_hit_and_skips_the_model(
    tmp_path, make_router, sample_cv_text
):
    from job_agent.memory import ParsedCVCache

    cache = ParsedCVCache(tmp_path / "app.db")
    router = make_router(light_replies=[json.dumps(FIXTURE_EXTRACTION)])

    first = parse_cv(sample_cv_text, router, cache=cache)
    second = parse_cv(sample_cv_text, router, cache=cache)

    # Only one reply was ever queued - a second model call would raise
    # IndexError popping from an empty list, so this also proves no request
    # was made the second time, not merely that the result happens to match.
    assert len(router.for_step("parse_cv").calls) == 1
    assert second.candidate_name == first.candidate_name
    assert [e.text for e in second.evidence] == [e.text for e in first.evidence]


def test_a_changed_baseline_cv_is_not_served_the_old_parse(
    tmp_path, make_router, sample_cv_text
):
    """Editing or replacing the baseline CV must never return a stale
    parse - proven here by a second, different CV text triggering a real
    second model call rather than reusing the first result."""
    from job_agent.memory import ParsedCVCache

    cache = ParsedCVCache(tmp_path / "app.db")
    second_extraction = {**FIXTURE_EXTRACTION, "candidate_name": "A Different Name"}
    router = make_router(
        light_replies=[json.dumps(FIXTURE_EXTRACTION), json.dumps(second_extraction)]
    )

    parse_cv(sample_cv_text, router, cache=cache)
    changed = parse_cv(sample_cv_text + "\nNew line.", router, cache=cache)

    assert len(router.for_step("parse_cv").calls) == 2
    assert changed.candidate_name == "A Different Name"

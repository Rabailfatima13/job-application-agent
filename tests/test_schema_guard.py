"""Validation + bounded retry (FR-9): bad output is never returned, and the
retry loop terminates."""

import json

import pytest

from job_agent.models import FitReport
from job_agent.observability import TraceCollector
from job_agent.validation import (
    RetryExhaustedError,
    generate_validated,
    parse_structured,
)

VALID = {
    "role": "ML Engineer",
    "company": "Arbisoft",
    "overall_fit": 0.6,
    "requirement_matches": [],
    "gaps": [],
    "recommended_emphasis": [],
}


def test_parses_json_wrapped_in_prose_and_fences():
    raw = f"Here you go:\n```json\n{json.dumps(VALID)}\n```\nHope that helps!"
    assert parse_structured(raw, FitReport).company == "Arbisoft"


def test_retries_with_the_previous_error_then_succeeds():
    seen: list[str | None] = []

    def produce(feedback):
        seen.append(feedback)
        return "not json at all" if len(seen) == 1 else json.dumps(VALID)

    collector = TraceCollector()
    report = generate_validated(produce, FitReport, collector=collector, agent="scoring")

    assert report.overall_fit == 0.6
    assert seen[0] is None
    assert seen[1] and "json" in seen[1].lower()  # the retry was told what broke
    assert [e.status for e in collector.events] == ["error"]  # only the failure traced


def test_gives_up_after_max_attempts_instead_of_looping():
    attempts = []

    def produce(feedback):
        attempts.append(feedback)
        return {**VALID, "overall_fit": 3.0}  # out of range every time

    with pytest.raises(RetryExhaustedError) as exc:
        generate_validated(produce, FitReport, max_attempts=2)

    assert len(attempts) == 2
    assert exc.value.schema_name == "FitReport"

"""Tracing (FR-12): every call is recorded with latency and tokens, failures
included."""

import pytest

from job_agent.observability import TraceCollector, traced_tool_call


def test_records_a_successful_call_with_tokens_and_latency():
    collector = TraceCollector()

    with traced_tool_call(collector, "research", "web_search", {"q": "Arbisoft"}) as out:
        out["result"] = "3 results"
        out["input_tokens"] = 120
        out["output_tokens"] = 40

    (event,) = collector.events
    assert (event.agent, event.name) == ("research", "web_search")
    assert event.status == "success"
    assert event.arguments == {"q": "Arbisoft"}
    assert event.duration_ms >= 0
    assert collector.totals()["input_tokens"] == 120


def test_records_a_failure_and_re_raises():
    collector = TraceCollector()

    with pytest.raises(RuntimeError):
        with traced_tool_call(collector, "research", "web_search"):
            raise RuntimeError("search API down")

    (event,) = collector.events
    assert event.status == "error"
    assert "search API down" in event.error
    assert collector.totals()["errors"] == 1


def test_mirrors_events_to_a_log_file_when_configured(tmp_path):
    log = tmp_path / "logs" / "tool_calls.log"
    collector = TraceCollector(log_path=log)

    with traced_tool_call(collector, "scoring", "score_fit") as out:
        out["result"] = "0.7"

    assert '"name": "score_fit"' in log.read_text(encoding="utf-8")

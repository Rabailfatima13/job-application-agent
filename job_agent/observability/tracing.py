"""Structured tracing for every tool and model call (FR-12, NFR-5).

Carries forward the Week 5 TraceEvent/TraceCollector shape, with two changes
the proposal needs here: token counts are first-class (so per-run cost is
reportable, not just latency), and the recording point is a context manager
rather than a hook subclass, because this project's agents call tools directly
instead of going through Week 4's HookManager.

A LangSmith/Langfuse exporter can be added later by reading `collector.events`
- nothing else has to change.
"""

import datetime
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TraceEvent:
    """One traced call: a tool invocation or a model completion."""

    agent: str
    name: str  # tool name, or "model:<tier>"
    kind: str = "tool"  # "tool" | "model"
    arguments: dict[str, Any] = field(default_factory=dict)
    status: str = "success"  # "success" | "error"
    result: str | None = None
    error: str | None = None
    started_at: str = ""
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class TraceCollector:
    """Ordered in-memory trace for one run, optionally mirrored to a log file."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.events: list[TraceEvent] = []
        self.log_path = log_path

    def record(self, event: TraceEvent) -> None:
        self.events.append(event)
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event)) + "\n")

    def as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(e) for e in self.events]

    def totals(self) -> dict[str, float | int]:
        """Run-level cost/latency summary - what the Week 8 trace view reports."""
        return {
            "calls": len(self.events),
            "errors": sum(1 for e in self.events if e.status == "error"),
            "duration_ms": round(sum(e.duration_ms for e in self.events), 1),
            "input_tokens": sum(e.input_tokens for e in self.events),
            "output_tokens": sum(e.output_tokens for e in self.events),
        }

    def clear(self) -> None:
        self.events.clear()


@contextmanager
def traced_tool_call(
    collector: TraceCollector | None,
    agent: str,
    name: str,
    arguments: dict[str, Any] | None = None,
    kind: str = "tool",
):
    """Time a call and record exactly one TraceEvent for it.

    Yields a mutable dict the caller can write `result`/`input_tokens`/
    `output_tokens` into. Failures are recorded as error events and then
    re-raised - tracing never swallows an exception.
    """
    event = TraceEvent(
        agent=agent,
        name=name,
        kind=kind,
        arguments=dict(arguments or {}),
        started_at=datetime.datetime.now().isoformat(timespec="milliseconds"),
    )
    outcome: dict[str, Any] = {}
    start = time.perf_counter()
    try:
        yield outcome
    except Exception as exc:
        event.status = "error"
        event.error = str(exc)
        raise
    else:
        event.result = (
            None if outcome.get("result") is None else str(outcome["result"])[:500]
        )
        event.input_tokens = int(outcome.get("input_tokens", 0))
        event.output_tokens = int(outcome.get("output_tokens", 0))
    finally:
        event.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        if collector is not None:
            collector.record(event)

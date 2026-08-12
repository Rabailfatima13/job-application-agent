"""Schema guard + bounded retry (FR-9, NFR-3).

A model asked for JSON will occasionally return prose around it, a missing
field, or an out-of-range score. The rule here is: reject it, tell the model
precisely what was wrong, ask again, and give up after a bounded number of
attempts rather than looping. Range and enum checks come free - they are
declared on the Pydantic models themselves (e.g. overall_fit is ge=0, le=1).

Deliberately model-agnostic: `generate_validated` takes any callable that
produces raw output, so it is tested without an API key and reused by every
agent.
"""

import json
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..observability.tracing import TraceCollector, TraceEvent

T = TypeVar("T", bound=BaseModel)

DEFAULT_MAX_ATTEMPTS = 3


class RetryExhaustedError(RuntimeError):
    """Raised when every attempt produced output that failed validation.

    Callers are expected to catch this and fall back to something safe (for
    tailoring, that means returning the original CV plus the gap list) rather
    than emitting an unvalidated result.
    """

    def __init__(self, schema_name: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"{schema_name} failed validation after {attempts} attempt(s): {last_error}"
        )
        self.schema_name = schema_name
        self.attempts = attempts
        self.last_error = last_error


def _extract_json(raw: str) -> str:
    """Pull the outermost JSON object out of a response that may be wrapped in
    prose or a ``` fence. Returns `raw` unchanged when no object is found, so
    the caller still gets a real validation error rather than a silent pass."""
    start = raw.find("{")
    end = raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end > start else raw


def parse_structured(raw: str | dict, schema: type[T]) -> T:
    """Validate raw model output into `schema`, raising ValidationError if it
    does not conform."""
    if isinstance(raw, str):
        try:
            raw = json.loads(_extract_json(raw))
        except json.JSONDecodeError as exc:
            raise ValidationError.from_exception_data(
                schema.__name__,
                [
                    {
                        "type": "json_invalid",
                        "loc": (),
                        "input": raw,
                        "ctx": {"error": str(exc)},
                    }
                ],
            ) from exc
    return schema.model_validate(raw)


def generate_validated(
    produce: Callable[[str | None], str | dict],
    schema: type[T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    collector: TraceCollector | None = None,
    agent: str = "",
) -> T:
    """Call `produce` until its output validates against `schema`.

    `produce` receives the previous validation error on every retry (None on
    the first attempt) so the prompt can tell the model what to fix. Each
    failed attempt is recorded as a trace event, so a retry is visible in the
    same place as every other call rather than hidden.
    """
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        raw = produce(last_error or None)
        try:
            return parse_structured(raw, schema)
        except ValidationError as exc:
            last_error = str(exc)
            if collector is not None:
                collector.record(
                    TraceEvent(
                        agent=agent,
                        name=f"validate:{schema.__name__}",
                        kind="tool",
                        arguments={"attempt": attempt},
                        status="error",
                        error=last_error,
                    )
                )
    raise RetryExhaustedError(schema.__name__, max_attempts, last_error)

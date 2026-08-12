"""Output validation: nothing unvalidated ever reaches the user or the store.

Two independent guards, matching the proposal's S14:
  schema_guard  - shape/range/enum correctness, with bounded retry (FR-9)
  grounding     - the no-fabrication check on tailored text (FR-10, Week 7)
"""

from .schema_guard import (
    RetryExhaustedError,
    generate_validated,
    parse_structured,
)

__all__ = ["RetryExhaustedError", "generate_validated", "parse_structured"]

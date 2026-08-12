"""Output validation: nothing unvalidated ever reaches the user or the store.

Two independent guards, matching the proposal's S14:
  schema_guard  - shape/range/enum correctness, with bounded retry (FR-9)
  grounding     - the no-fabrication check on tailored text (FR-10, Week 7)
"""

from .grounding import (
    EntailmentVerdicts,
    GroundingIssue,
    GroundingResult,
    apply_verdicts,
    check_grounding,
    split_sentences,
)
from .schema_guard import (
    RetryExhaustedError,
    generate_validated,
    parse_structured,
)

__all__ = [
    "EntailmentVerdicts",
    "GroundingIssue",
    "GroundingResult",
    "RetryExhaustedError",
    "apply_verdicts",
    "check_grounding",
    "generate_validated",
    "parse_structured",
    "split_sentences",
]

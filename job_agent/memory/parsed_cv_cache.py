"""Persistent cache for parsed baseline-CV evidence: the same light-tier
extraction call, repeated for nothing, every time a second (or third, or
tenth) job application is started against a baseline CV that has not
changed since the last one (see `agents/supervisor.py`'s `_parse_cv` node,
which calls the parsing tool unconditionally on every run).

Same idea as `CompanyResearchCache` (a company's own name already fully
determines its brief), applied to the other side of a run: a CV's own text
already fully determines its parse. Keyed by the text's own content hash
rather than a user id or a baseline-CV row id, so correctness needs no
explicit invalidation at all - editing or replacing the baseline CV changes
the text, which changes the hash, which is a guaranteed cache miss. No TTL
either: unlike a company's public profile, a fixed piece of text does not go
stale with the passage of time.

Reuses the same SQLite file every other store in this project writes to
(`Settings.tracker_db_path`) - one more table, not a second database, not a
second cache service. Plain sqlite3, connection-per-operation, the same
pattern `BaselineCVStore`/`TailoredCVVersionStore`/`CompanyResearchCache`
already use.
"""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..models import CVEvidence, ParsedCV

SCHEMA = """
CREATE TABLE IF NOT EXISTS parsed_cv_cache (
    text_hash      TEXT PRIMARY KEY,
    candidate_name TEXT,
    evidence       TEXT NOT NULL,
    skills         TEXT NOT NULL
)
"""


def _hash_text(text: str) -> str:
    """The cache key: a hash of the exact CV text, not the text itself - two
    different users pasting byte-identical text would (correctly, safely)
    share a cache entry, but the table never has to store the CV text a
    second time next to `baseline_cvs`."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ParsedCVCache:
    """One row per distinct CV text - a cache miss (nothing stored) is
    always safe: the caller just parses again, exactly as if this class did
    not exist."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commits on success, rolls back on exception
                yield connection
        finally:
            connection.close()

    def get(self, raw_text: str) -> ParsedCV | None:
        """The cached parse of `raw_text`, or None on a miss. `raw_text` is
        the caller's own copy either way - a hit never needs to hand it
        back, only the two things the extraction actually produced."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT candidate_name, evidence, skills FROM parsed_cv_cache "
                "WHERE text_hash = ?",
                (_hash_text(raw_text),),
            ).fetchone()
        if row is None:
            return None
        return ParsedCV(
            candidate_name=row["candidate_name"],
            raw_text=raw_text,
            evidence=[CVEvidence(**e) for e in json.loads(row["evidence"])],
            skills=json.loads(row["skills"]),
        )

    def set(self, raw_text: str, parsed: ParsedCV) -> None:
        """Cache `parsed` under `raw_text`'s hash, replacing whatever was
        cached for it before (there is nothing to replace in practice - the
        key is the text itself - but a byte-identical re-parse should still
        just overwrite cleanly rather than raise)."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO parsed_cv_cache "
                "(text_hash, candidate_name, evidence, skills) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(text_hash) DO UPDATE SET "
                "candidate_name = excluded.candidate_name, "
                "evidence = excluded.evidence, skills = excluded.skills",
                (
                    _hash_text(raw_text),
                    parsed.candidate_name,
                    json.dumps([e.model_dump() for e in parsed.evidence]),
                    json.dumps(parsed.skills),
                ),
            )

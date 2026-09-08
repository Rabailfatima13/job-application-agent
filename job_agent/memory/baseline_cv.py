"""Persistent storage for a user's baseline CV (mentor-requested "baseline CV
system"): the original CV a user uploads once, reused for every application
rather than re-supplied per job.

Reuses the same SQLite file every other store in this project writes to
(`Settings.tracker_db_path`) - one more table, not a second database. Plain
sqlite3, connection-per-operation - the same pattern `UserStore`
(memory/users.py) and `SQLiteApplicationTracker` (memory/tracker.py) already
use, deliberately kept consistent rather than inventing a second persistence
style.

Stores raw CV text only - never a `ParsedCV`, never a `TailoredCV`. The
existing `parse_cv` tool still does the real parsing, fresh, from this text,
every time a run uses it; nothing here duplicates or replaces that. This is
also what makes "a tailored CV must never overwrite the baseline" true by
construction rather than by convention: `save` only ever accepts a plain
string, so a `TailoredCV` object has nowhere to go even if a caller tried,
and the only callers in this codebase are the two explicit UI actions a human
takes - save the first baseline, replace it later. The pipeline and the
writing agent never call this module at all.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ..models.baseline_cv import BaselineCV

SCHEMA = """
CREATE TABLE IF NOT EXISTS baseline_cvs (
    user_id    INTEGER PRIMARY KEY,
    cv_text    TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


class BaselineCVStore:
    """One baseline CV per user - `user_id` is the primary key, so saving
    again for the same user always replaces it in place, never adds a second
    row and never touches any other user's row."""

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

    def save(self, user_id: int, cv_text: str) -> BaselineCV:
        """Save this user's first baseline CV, or replace their existing one.

        A blank CV is refused rather than silently stored - the same
        discipline `load_document_text` already applies to a pipeline run.
        `created_at` is preserved across a replace; only `updated_at` moves,
        so "when was this first set up" survives later edits.
        """
        cv_text = cv_text.strip()
        if not cv_text:
            raise ValueError("A CV is required.")

        existing = self.get(user_id)
        now = datetime.now()
        created_at = existing.created_at if existing is not None else now
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO baseline_cvs (user_id, cv_text, created_at, updated_at) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET "
                "cv_text = excluded.cv_text, updated_at = excluded.updated_at",
                (user_id, cv_text, created_at.isoformat(), now.isoformat()),
            )
        return BaselineCV(
            user_id=user_id, cv_text=cv_text, created_at=created_at, updated_at=now
        )

    def get(self, user_id: int) -> BaselineCV | None:
        """This user's baseline CV, or None if they have never saved one."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT user_id, cv_text, created_at, updated_at "
                "FROM baseline_cvs WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return None if row is None else BaselineCV(**dict(row))

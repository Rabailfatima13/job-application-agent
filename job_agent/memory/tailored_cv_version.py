"""Persistent storage for tailored CV versions (mentor-requested "multiple
tailored CV versions"): one immutable row per application a user has
actually completed a run for, kept entirely apart from their baseline CV.

Reuses the same SQLite file every other store in this project writes to
(`Settings.tracker_db_path`) - one more table, not a second database. Plain
sqlite3, connection-per-operation - the same pattern `BaselineCVStore`,
`UserStore` and `SQLiteApplicationTracker` already use.

Unlike `BaselineCVStore.save` (one row per user, replaced in place), `save`
here always inserts a new row: a second application to Sephora, or an
application to Nexus, must never touch an earlier version, so there is no
upsert key to conflict on - only an always-fresh id (the same
`uuid4().hex[:12]` shape `Supervisor.new_application_id` already uses for
application ids).
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from ..models.tailored_cv_version import TailoredCVVersion

SCHEMA = """
CREATE TABLE IF NOT EXISTS tailored_cv_versions (
    id         TEXT PRIMARY KEY,
    user_id    INTEGER NOT NULL,
    company    TEXT NOT NULL,
    role       TEXT NOT NULL,
    cv_text    TEXT NOT NULL,
    created_at TEXT NOT NULL
)
"""

_COLUMNS = "id, user_id, company, role, cv_text, created_at, baseline_cv_updated_at"


class TailoredCVVersionStore:
    """Every tailored CV a user has generated - newest first, never merged."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(SCHEMA)
            self._ensure_baseline_reference_column(connection)

    @staticmethod
    def _ensure_baseline_reference_column(connection: sqlite3.Connection) -> None:
        """`baseline_cv_updated_at` was added after this table was already in
        real use - `CREATE TABLE IF NOT EXISTS` alone never touches a table
        that already exists, so an existing database file would otherwise be
        missing this column forever. Checked (and, at most once, fixed) on
        every connect - idempotent, since the second check always finds the
        column already there.
        """
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(tailored_cv_versions)")
        }
        if "baseline_cv_updated_at" not in columns:
            connection.execute(
                "ALTER TABLE tailored_cv_versions ADD COLUMN baseline_cv_updated_at TEXT"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            with connection:  # commits on success, rolls back on exception
                yield connection
        finally:
            connection.close()

    def save(
        self,
        user_id: int,
        company: str,
        role: str,
        cv_text: str,
        baseline_cv_updated_at: datetime | None = None,
    ) -> TailoredCVVersion:
        """Record one more tailored CV for this user - always a new row.

        No upsert key: two applications to the same company are two
        separate rows, exactly like `ApplicationRecord` already allows two
        applications with the same role and company (see
        `Supervisor.new_application_id`). `baseline_cv_updated_at` is
        optional so existing callers that predate it keep working unchanged.
        """
        version = TailoredCVVersion(
            id=uuid4().hex[:12],
            user_id=user_id,
            company=company,
            role=role,
            cv_text=cv_text,
            created_at=datetime.now(),
            baseline_cv_updated_at=baseline_cv_updated_at,
        )
        with self._connect() as connection:
            connection.execute(
                f"INSERT INTO tailored_cv_versions ({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    version.id,
                    version.user_id,
                    version.company,
                    version.role,
                    version.cv_text,
                    version.created_at.isoformat(),
                    version.baseline_cv_updated_at.isoformat()
                    if version.baseline_cv_updated_at
                    else None,
                ),
            )
        return version

    def list_for_user(self, user_id: int) -> list[TailoredCVVersion]:
        """This user's tailored CVs, newest first - never another user's."""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT {_COLUMNS} FROM tailored_cv_versions "
                "WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [TailoredCVVersion.model_validate(dict(row)) for row in rows]

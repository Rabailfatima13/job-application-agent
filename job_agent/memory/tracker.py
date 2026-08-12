"""The application tracker (FR-7, FR-13, US-6).

`ApplicationTracker` is the interface the supervisor and the MCP server both
talk to. Two implementations:

  InMemoryApplicationTracker - non-persistent; used by tests and by callers
                               that want a throwaway pipeline.
  SQLiteApplicationTracker   - the real one, backed by a SQLite file, so the
                               list survives restarts (FR-7).

Both satisfy the same four methods and are covered by the same contract tests.
Keeping the interface this narrow is what lets the MCP server expose the
tracker as a resource without any agent code knowing where the rows live.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import ApplicationRecord, ApplicationStatus


@runtime_checkable
class ApplicationTracker(Protocol):
    def add(self, record: ApplicationRecord) -> ApplicationRecord:
        """Persist a new application."""
        ...

    def get(self, application_id: str) -> ApplicationRecord | None: ...

    def list(self, status: ApplicationStatus | None = None) -> list[ApplicationRecord]:
        """All applications, newest first, optionally filtered by status (US-8)."""
        ...

    def set_status(
        self, application_id: str, status: ApplicationStatus
    ) -> ApplicationRecord:
        """Move an application along the pipeline. Only ever called on behalf of
        the human - the agent does not mark anything submitted (FR-14)."""
        ...


class InMemoryApplicationTracker:
    """Non-persistent reference implementation of the interface above."""

    def __init__(self) -> None:
        self._records: dict[str, ApplicationRecord] = {}

    def add(self, record: ApplicationRecord) -> ApplicationRecord:
        if record.application_id in self._records:
            raise ValueError(f"Application '{record.application_id}' already tracked.")
        self._records[record.application_id] = record
        return record

    def get(self, application_id: str) -> ApplicationRecord | None:
        return self._records.get(application_id)

    def list(self, status: ApplicationStatus | None = None) -> list[ApplicationRecord]:
        records = sorted(
            self._records.values(), key=lambda r: r.created_at, reverse=True
        )
        return [r for r in records if status is None or r.status == status]

    def set_status(
        self, application_id: str, status: ApplicationStatus
    ) -> ApplicationRecord:
        record = self._records.get(application_id)
        if record is None:
            raise KeyError(f"Unknown application '{application_id}'.")
        updated = record.model_copy(update={"status": ApplicationStatus(status)})
        self._records[application_id] = updated
        return updated


# The status column is constrained to the enum's own values, so the schema
# cannot drift from ApplicationStatus if a status is ever added.
_STATUS_VALUES = ", ".join(f"'{s.value}'" for s in ApplicationStatus)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS applications (
    application_id TEXT PRIMARY KEY,
    role           TEXT NOT NULL,
    company        TEXT NOT NULL,
    fit_score      REAL NOT NULL CHECK (fit_score >= 0.0 AND fit_score <= 1.0),
    status         TEXT NOT NULL CHECK (status IN ({_STATUS_VALUES})),
    created_at     TEXT NOT NULL
)
"""

_COLUMNS = "application_id, role, company, fit_score, status, created_at"


class SQLiteApplicationTracker:
    """The persistent tracker: one SQLite file, one table, six columns.

    Deliberately plain sqlite3 - no ORM. The tracker stores six flat values and
    is read by the UI and (in Week 7) by the MCP server; an ORM would add a
    dependency and a mapping layer without removing a single line of this file.

    Connections are opened per operation rather than held. That keeps the
    tracker safe to use from Streamlit (which reruns scripts across threads),
    and at this scale - tens of applications - costs nothing.

    Validation stays where it already lives: every row goes in as an
    `ApplicationRecord` and comes back out through `model_validate`, so a bad
    score or an unknown status cannot enter or leave the database silently.
    The CHECK constraints are a second line of defence against a row written by
    something other than this class.
    """

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

    @staticmethod
    def _to_record(row: sqlite3.Row) -> ApplicationRecord:
        return ApplicationRecord.model_validate(dict(row))

    def add(self, record: ApplicationRecord) -> ApplicationRecord:
        # `record` is already an ApplicationRecord, so range/enum/required-field
        # validation has happened before we get here.
        try:
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO applications ({_COLUMNS}) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.application_id,
                        record.role,
                        record.company,
                        record.fit_score,
                        record.status.value,
                        record.created_at.isoformat(),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Could not track application '{record.application_id}': {exc}"
            ) from exc
        return record

    def get(self, application_id: str) -> ApplicationRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT {_COLUMNS} FROM applications WHERE application_id = ?",
                (application_id,),
            ).fetchone()
        return None if row is None else self._to_record(row)

    def list(self, status: ApplicationStatus | None = None) -> list[ApplicationRecord]:
        query = f"SELECT {_COLUMNS} FROM applications"
        parameters: tuple = ()
        if status is not None:
            query += " WHERE status = ?"
            parameters = (ApplicationStatus(status).value,)
        # created_at is stored ISO-formatted, so lexical order is chronological.
        query += " ORDER BY created_at DESC"

        with self._connect() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._to_record(row) for row in rows]

    def set_status(
        self, application_id: str, status: ApplicationStatus
    ) -> ApplicationRecord:
        status = ApplicationStatus(status)  # rejects an unknown status value
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE applications SET status = ? WHERE application_id = ?",
                (status.value, application_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Unknown application '{application_id}'.")
        updated = self.get(application_id)
        if updated is None:  # pragma: no cover - only reachable on a concurrent delete
            raise KeyError(f"Unknown application '{application_id}'.")
        return updated

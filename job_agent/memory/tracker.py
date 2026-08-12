"""The application tracker (FR-7, FR-13, US-6).

`ApplicationTracker` is the interface the supervisor and the MCP server both
talk to. Two implementations are planned:

  InMemoryApplicationTracker - here; used by tests and by the pipeline before
                               persistence lands.
  SQLiteApplicationTracker   - Week 6; same four methods, backed by the file at
                               Settings.tracker_db_path so the list survives
                               restarts.

Keeping the interface this narrow is what lets the MCP server expose the
tracker as a resource without any agent code knowing where the rows live.
"""

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
        updated = record.model_copy(update={"status": status})
        self._records[application_id] = updated
        return updated

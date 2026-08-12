"""Tracker behaviour (FR-7, FR-13, US-6, US-8).

The contract tests run against *both* implementations: the SQLite tracker is
only correct if it behaves exactly like the in-memory one, so the same tests
are parametrised over the two rather than duplicated. SQLite-specific concerns
- surviving a restart, creating its own schema - are tested separately below.

Every test uses a tmp_path database; nothing here can touch a real one.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from job_agent.memory import (
    ApplicationTracker,
    InMemoryApplicationTracker,
    SQLiteApplicationTracker,
)
from job_agent.models import ApplicationRecord, ApplicationStatus


@pytest.fixture(params=["in_memory", "sqlite"])
def tracker(request, tmp_path) -> ApplicationTracker:
    if request.param == "in_memory":
        return InMemoryApplicationTracker()
    return SQLiteApplicationTracker(tmp_path / "applications.db")


# --- Contract: both implementations -----------------------------------------


def test_tracker_satisfies_the_interface(tracker):
    assert isinstance(tracker, ApplicationTracker)


def test_added_applications_are_retrievable(tracker, application_record):
    tracker.add(application_record)

    stored = tracker.get("app-1")
    assert stored.company == "Arbisoft"
    assert stored.role == "ML Engineer"
    assert stored.fit_score == 0.75
    assert stored.status is ApplicationStatus.draft
    assert stored.created_at == datetime(2026, 8, 12, 9, 0)
    assert tracker.get("missing") is None


def test_lists_newest_first_and_filters_by_status(tracker, application_record):
    older = application_record.model_copy(
        update={"application_id": "app-0", "created_at": datetime(2026, 8, 1)}
    )
    tracker.add(application_record)
    tracker.add(older)

    assert [r.application_id for r in tracker.list()] == ["app-1", "app-0"]
    assert tracker.list(status=ApplicationStatus.submitted) == []
    assert len(tracker.list(status=ApplicationStatus.draft)) == 2


def test_status_moves_along_the_pipeline(tracker, application_record):
    tracker.add(application_record)

    updated = tracker.set_status("app-1", ApplicationStatus.ready)

    assert updated.status is ApplicationStatus.ready
    assert tracker.get("app-1").status is ApplicationStatus.ready
    # Only the status changes.
    assert tracker.get("app-1").fit_score == 0.75


def test_status_filter_follows_an_update(tracker, application_record):
    tracker.add(application_record)
    tracker.set_status("app-1", ApplicationStatus.submitted)

    assert tracker.list(status=ApplicationStatus.draft) == []
    assert [r.application_id for r in tracker.list(ApplicationStatus.submitted)] == [
        "app-1"
    ]


def test_duplicate_ids_are_rejected(tracker, application_record):
    tracker.add(application_record)
    with pytest.raises(ValueError):
        tracker.add(application_record)


def test_unknown_application_cannot_change_status(tracker):
    with pytest.raises(KeyError):
        tracker.set_status("nope", ApplicationStatus.ready)


def test_an_unknown_status_value_is_rejected(tracker, application_record):
    tracker.add(application_record)
    with pytest.raises(ValueError):
        tracker.set_status("app-1", "interviewing")


def test_an_empty_tracker_lists_nothing(tracker):
    assert tracker.list() == []


# --- Schema constraints, enforced by the Pydantic model ----------------------


@pytest.mark.parametrize("score", [-0.2, 1.5])
def test_record_requires_a_bounded_fit_score(application_record, score):
    with pytest.raises(ValidationError):
        ApplicationRecord.model_validate(
            {**application_record.model_dump(), "fit_score": score}
        )


def test_record_requires_a_known_status(application_record):
    with pytest.raises(ValidationError):
        ApplicationRecord.model_validate(
            {**application_record.model_dump(), "status": "interviewing"}
        )


def test_record_requires_its_core_fields(application_record):
    incomplete = application_record.model_dump()
    del incomplete["company"]
    with pytest.raises(ValidationError):
        ApplicationRecord.model_validate(incomplete)


def test_an_unvalidated_record_cannot_be_tracked(tracker):
    # The tracker takes ApplicationRecord, so a raw dict never reaches SQL.
    with pytest.raises(AttributeError):
        tracker.add({"application_id": "app-9", "fit_score": 4.0})


# --- SQLite specifics --------------------------------------------------------


def test_records_survive_a_new_tracker_instance(tmp_path, application_record):
    db_path = tmp_path / "applications.db"

    first = SQLiteApplicationTracker(db_path)
    first.add(application_record)
    first.set_status("app-1", ApplicationStatus.ready)
    del first  # the process that wrote the record is gone

    second = SQLiteApplicationTracker(db_path)
    reopened = second.get("app-1")

    assert reopened is not None
    assert reopened.role == "ML Engineer"
    assert reopened.fit_score == 0.75
    assert reopened.status is ApplicationStatus.ready
    assert reopened.created_at == datetime(2026, 8, 12, 9, 0)


def test_multiple_applications_survive_and_keep_their_order(tmp_path, application_record):
    db_path = tmp_path / "applications.db"
    first = SQLiteApplicationTracker(db_path)
    first.add(application_record)
    first.add(
        application_record.model_copy(
            update={
                "application_id": "app-2",
                "role": "AI Engineer",
                "created_at": datetime(2026, 8, 20),
            }
        )
    )

    reopened = SQLiteApplicationTracker(db_path).list()

    assert [r.application_id for r in reopened] == ["app-2", "app-1"]


def test_database_and_parent_directory_are_created_on_demand(tmp_path):
    db_path = tmp_path / "nested" / "store" / "applications.db"

    SQLiteApplicationTracker(db_path)

    assert db_path.exists()


def test_opening_an_existing_database_keeps_its_rows(tmp_path, application_record):
    db_path = tmp_path / "applications.db"
    SQLiteApplicationTracker(db_path).add(application_record)

    # Re-running the schema setup must not wipe anything.
    assert SQLiteApplicationTracker(db_path).get("app-1") is not None


def test_the_database_rejects_a_row_written_around_the_model(tmp_path):
    # Second line of defence: even a direct INSERT cannot store a bad score.
    import sqlite3

    db_path = tmp_path / "applications.db"
    SQLiteApplicationTracker(db_path)

    with sqlite3.connect(db_path) as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?)",
                ("app-x", "Role", "Co", 4.0, "draft", "2026-08-12T09:00:00"),
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO applications VALUES (?, ?, ?, ?, ?, ?)",
                ("app-y", "Role", "Co", 0.5, "interviewing", "2026-08-12T09:00:00"),
            )


def test_the_tracker_uses_the_path_it_is_given(tmp_path, application_record, settings):
    # No production path is baked in - the caller (or Settings) decides.
    tracker = SQLiteApplicationTracker(settings.tracker_db_path)
    tracker.add(application_record)

    assert settings.tracker_db_path.exists()
    assert settings.tracker_db_path.parent == tmp_path

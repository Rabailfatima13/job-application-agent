"""Tracker behaviour (FR-7, US-6, US-8) against the in-memory implementation.

These tests are written against the ApplicationTracker interface on purpose:
Week 6's SQLite implementation must pass the same ones.
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from job_agent.memory import ApplicationTracker, InMemoryApplicationTracker
from job_agent.models import ApplicationRecord, ApplicationStatus


@pytest.fixture
def tracker() -> InMemoryApplicationTracker:
    return InMemoryApplicationTracker()


def test_in_memory_tracker_satisfies_the_interface(tracker):
    assert isinstance(tracker, ApplicationTracker)


def test_added_applications_are_retrievable(tracker, application_record):
    tracker.add(application_record)
    assert tracker.get("app-1").company == "Arbisoft"
    assert tracker.get("missing") is None


def test_lists_newest_first_and_filters_by_status(tracker, application_record):
    older = application_record.model_copy(
        update={"application_id": "app-0", "created_at": datetime(2026, 8, 1)}
    )
    tracker.add(application_record)
    tracker.add(older)

    assert [r.application_id for r in tracker.list()] == ["app-1", "app-0"]
    assert tracker.list(status=ApplicationStatus.submitted) == []


def test_status_moves_along_the_pipeline(tracker, application_record):
    tracker.add(application_record)
    updated = tracker.set_status("app-1", ApplicationStatus.ready)

    assert updated.status is ApplicationStatus.ready
    assert tracker.get("app-1").status is ApplicationStatus.ready


def test_duplicate_ids_are_rejected(tracker, application_record):
    tracker.add(application_record)
    with pytest.raises(ValueError):
        tracker.add(application_record)


def test_unknown_application_cannot_change_status(tracker):
    with pytest.raises(KeyError):
        tracker.set_status("nope", ApplicationStatus.ready)


def test_record_requires_a_bounded_fit_score(application_record):
    with pytest.raises(ValidationError):
        ApplicationRecord.model_validate(
            {**application_record.model_dump(), "fit_score": 1.5}
        )

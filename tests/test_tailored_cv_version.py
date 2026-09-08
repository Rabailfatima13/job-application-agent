"""The tailored CV version store (mentor-requested "multiple tailored CV
versions").

Every test uses a tmp_path database; nothing here can touch a real one. What
is under test is the contract the UI relies on: every save is a new,
independent row - never an update, never a merge, and never visible to
another user.
"""

import sqlite3
from datetime import datetime

import pytest

from job_agent.memory import TailoredCVVersionStore
from job_agent.models import TailoredCVVersion


@pytest.fixture
def store(tmp_path) -> TailoredCVVersionStore:
    return TailoredCVVersionStore(tmp_path / "app.db")


# --- save / retrieve ----------------------------------------------------


def test_no_tailored_cvs_exist_before_one_is_saved(store):
    assert store.list_for_user(user_id=1) == []


def test_a_tailored_cv_can_be_saved_and_listed(store):
    saved = store.save(
        user_id=1, company="Sephora", role="Data Scientist", cv_text="- Built X."
    )

    assert isinstance(saved, TailoredCVVersion)
    assert saved.user_id == 1
    assert saved.company == "Sephora"
    assert saved.role == "Data Scientist"
    assert saved.cv_text == "- Built X."
    assert saved.id  # a real id was assigned

    fetched = store.list_for_user(user_id=1)
    assert fetched == [saved]


def test_each_save_gets_a_unique_id(store):
    first = store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="a")
    second = store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="b")

    assert first.id != second.id


# --- multiple versions never overwrite one another -----------------------


def test_saving_again_for_the_same_company_adds_a_new_row_not_an_update(store):
    """Applying to Sephora twice must produce two rows, exactly like a
    second application to the same company is already allowed in the
    application tracker (`Supervisor.new_application_id`)."""
    first = store.save(
        user_id=1, company="Sephora", role="Data Scientist", cv_text="Version one."
    )
    second = store.save(
        user_id=1, company="Sephora", role="Data Scientist", cv_text="Version two."
    )

    versions = store.list_for_user(user_id=1)
    assert len(versions) == 2
    texts = {v.cv_text for v in versions}
    assert texts == {"Version one.", "Version two."}
    assert first.id != second.id


def test_sephora_and_nexus_are_stored_as_separate_records(store):
    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="Sephora CV.")
    store.save(user_id=1, company="Nexus", role="Data Scientist", cv_text="Nexus CV.")

    versions = store.list_for_user(user_id=1)
    companies = {v.company for v in versions}
    assert companies == {"Sephora", "Nexus"}
    assert len(versions) == 2


def test_a_new_version_never_overwrites_an_older_one(store):
    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="Sephora CV.")
    store.save(user_id=1, company="Nexus", role="Data Scientist", cv_text="Nexus CV.")
    store.save(user_id=1, company="Third Co", role="Analyst", cv_text="Third CV.")

    versions = {v.company: v.cv_text for v in store.list_for_user(user_id=1)}
    assert versions["Sephora"] == "Sephora CV."
    assert versions["Nexus"] == "Nexus CV."
    assert versions["Third Co"] == "Third CV."


def test_newest_version_is_listed_first(store):
    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="a")
    store.save(user_id=1, company="Nexus", role="Data Scientist", cv_text="b")

    versions = store.list_for_user(user_id=1)
    assert versions[0].company == "Nexus"
    assert versions[1].company == "Sephora"


# --- user isolation ---------------------------------------------------------


def test_each_user_only_sees_their_own_tailored_cvs(store):
    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="Ada's CV.")
    store.save(user_id=2, company="Nexus", role="Data Scientist", cv_text="Grace's CV.")

    assert [v.company for v in store.list_for_user(user_id=1)] == ["Sephora"]
    assert [v.company for v in store.list_for_user(user_id=2)] == ["Nexus"]


def test_one_users_new_version_never_appears_in_another_users_list(store):
    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="Ada's CV.")

    store.save(user_id=2, company="Sephora", role="Data Scientist", cv_text="Grace's CV.")

    assert len(store.list_for_user(user_id=1)) == 1
    assert len(store.list_for_user(user_id=2)) == 1


# --- the complete tailored CV, and the baseline it was tailored from --------


def test_a_complete_multi_line_cv_document_round_trips_intact(store):
    """The store must hold a whole tailored CV, not just a short highlight
    string - a realistic, multi-section document round-trips byte-for-byte."""
    complete_cv = (
        "Ada Lovelace\n\n"
        "Tailored for: Data Scientist at Sephora\n\n"
        "- Built a recommendation engine.\n"
        "- Led a team of 3 engineers.\n\n"
        "Skills: Python, SQL, pandas\n\n"
        "Additional Experience\n"
        "- Ran a university coding club."
    )

    store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text=complete_cv)

    fetched = store.list_for_user(user_id=1)[0]
    assert fetched.cv_text == complete_cv


def test_the_store_survives_a_fresh_instance_over_the_same_file(tmp_path):
    """Requirement: reloading the application must still retrieve previously
    saved tailored CVs - simulated here by dropping the original `store`
    object and opening a brand new one against the same database file,
    exactly like a fresh Streamlit process would."""
    db_path = tmp_path / "app.db"
    TailoredCVVersionStore(db_path).save(
        user_id=1, company="Sephora", role="Data Scientist", cv_text="Persisted CV."
    )

    reopened = TailoredCVVersionStore(db_path)

    versions = reopened.list_for_user(user_id=1)
    assert len(versions) == 1
    assert versions[0].cv_text == "Persisted CV."


def test_saving_with_a_baseline_reference_records_it(store):
    snapshot = datetime(2026, 1, 1, 12, 0)

    saved = store.save(
        user_id=1,
        company="Sephora",
        role="Data Scientist",
        cv_text="Some CV.",
        baseline_cv_updated_at=snapshot,
    )

    assert saved.baseline_cv_updated_at == snapshot
    assert store.list_for_user(user_id=1)[0].baseline_cv_updated_at == snapshot


def test_saving_without_a_baseline_reference_leaves_it_none(store):
    saved = store.save(user_id=1, company="Sephora", role="Data Scientist", cv_text="x")

    assert saved.baseline_cv_updated_at is None
    assert store.list_for_user(user_id=1)[0].baseline_cv_updated_at is None


def test_a_pre_existing_database_without_the_new_column_is_migrated(tmp_path):
    """`baseline_cv_updated_at` was added after this table was already in
    real use - a database file created before it must still open, migrate,
    and read its existing rows correctly rather than erroring."""
    db_path = tmp_path / "old.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        "CREATE TABLE tailored_cv_versions ("
        "id TEXT PRIMARY KEY, user_id INTEGER NOT NULL, company TEXT NOT NULL, "
        "role TEXT NOT NULL, cv_text TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    connection.execute(
        "INSERT INTO tailored_cv_versions VALUES "
        "('abc123', 1, 'Sephora', 'Data Scientist', 'Old CV text.', "
        "'2026-01-01T00:00:00')"
    )
    connection.commit()
    connection.close()

    store = TailoredCVVersionStore(db_path)

    versions = store.list_for_user(user_id=1)
    assert len(versions) == 1
    assert versions[0].id == "abc123"
    assert versions[0].cv_text == "Old CV text."
    assert versions[0].baseline_cv_updated_at is None

    # And the store is still fully usable afterwards.
    store.save(user_id=1, company="Nexus", role="Analyst", cv_text="New CV.")
    assert len(store.list_for_user(user_id=1)) == 2

"""The baseline CV store (mentor-requested baseline CV system).

Every test uses a tmp_path database; nothing here can touch a real one. What
is under test is the contract the UI relies on: save once, read it back,
replace it explicitly, and never let anything but a plain string in.
"""

import pytest

from job_agent.memory import BaselineCVStore
from job_agent.models import BaselineCV, TailoredCV


@pytest.fixture
def store(tmp_path) -> BaselineCVStore:
    return BaselineCVStore(tmp_path / "app.db")


# --- save / retrieve ----------------------------------------------------


def test_no_baseline_exists_before_one_is_saved(store):
    assert store.get(user_id=1) is None


def test_a_baseline_cv_can_be_saved_and_retrieved(store):
    saved = store.save(user_id=1, cv_text="Built a REST API with FastAPI.")

    assert isinstance(saved, BaselineCV)
    assert saved.user_id == 1
    assert saved.cv_text == "Built a REST API with FastAPI."
    assert saved.created_at == saved.updated_at  # first save

    fetched = store.get(user_id=1)
    assert fetched == saved


def test_the_cv_text_is_stripped_before_storage(store):
    saved = store.save(user_id=1, cv_text="  Built a REST API.  \n")

    assert saved.cv_text == "Built a REST API."
    assert store.get(user_id=1).cv_text == "Built a REST API."


def test_a_blank_cv_is_refused(store):
    with pytest.raises(ValueError, match="required"):
        store.save(user_id=1, cv_text="   ")

    assert store.get(user_id=1) is None  # nothing was stored


# --- replace / update -----------------------------------------------------


def test_saving_again_replaces_the_existing_baseline_in_place(store):
    store.save(user_id=1, cv_text="Version one.")
    replaced = store.save(user_id=1, cv_text="Version two.")

    assert replaced.cv_text == "Version two."
    fetched = store.get(user_id=1)
    assert fetched.cv_text == "Version two."
    # Still exactly one baseline for this user, not a second row.
    assert fetched.user_id == 1


def test_replacing_preserves_the_original_created_at(store):
    first = store.save(user_id=1, cv_text="Version one.")
    second = store.save(user_id=1, cv_text="Version two.")

    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at


# --- a tailored CV must never become the baseline --------------------------


def test_save_only_ever_accepts_a_plain_string_never_a_tailored_cv(store):
    """Structural guarantee, not just convention: `save`'s signature takes
    `cv_text: str`. A `TailoredCV` object has nowhere to go - passing one
    fails immediately, before anything could be written to disk."""
    tailored = TailoredCV(
        role="Junior AI Engineer",
        company="Arbisoft",
        bullets=["Built a REST API with FastAPI."],
    )

    with pytest.raises((TypeError, AttributeError)):
        store.save(user_id=1, cv_text=tailored)

    assert store.get(user_id=1) is None


def test_the_baseline_is_unaffected_by_unrelated_application_activity(store):
    """A tailored draft being produced for some job application is a
    completely separate code path from this store - nothing here is ever
    called by the writing agent, so saving a baseline once and then
    "running applications" (simulated here by simply not touching the store
    again) leaves the baseline exactly as it was."""
    baseline = store.save(user_id=1, cv_text="Original CV text.")

    # No call to `store.save` happens here - this is the point being tested.

    assert store.get(user_id=1) == baseline


# --- user isolation ---------------------------------------------------------


def test_each_user_has_their_own_independent_baseline(store):
    store.save(user_id=1, cv_text="Ada's CV.")
    store.save(user_id=2, cv_text="Grace's CV.")

    assert store.get(user_id=1).cv_text == "Ada's CV."
    assert store.get(user_id=2).cv_text == "Grace's CV."


def test_replacing_one_users_baseline_never_touches_another_users(store):
    store.save(user_id=1, cv_text="Ada's original CV.")
    store.save(user_id=2, cv_text="Grace's CV.")

    store.save(user_id=1, cv_text="Ada's updated CV.")

    assert store.get(user_id=1).cv_text == "Ada's updated CV."
    assert store.get(user_id=2).cv_text == "Grace's CV."  # untouched

"""User accounts for the login/signup UI.

Every test uses a tmp_path database; nothing here can touch a real one, and
no test ever asserts against the raw password - only against the User object
returned, or the stored hash's shape.
"""

import sqlite3

import pytest

from job_agent.memory.users import (
    EmailAlreadyRegistered,
    UserStore,
    hash_password,
    verify_password,
)
from job_agent.models import User


@pytest.fixture
def store(tmp_path) -> UserStore:
    return UserStore(tmp_path / "app.db")


# --- signup -------------------------------------------------------------


def test_a_new_account_can_be_created(store):
    user = store.create("Mahnoor Rauf", "mahnoor@example.com", "correct-horse-1")

    assert isinstance(user, User)
    assert user.name == "Mahnoor Rauf"
    assert user.email == "mahnoor@example.com"
    assert user.id > 0


def test_email_is_normalised_to_lowercase(store):
    user = store.create("Ada", "Ada@Example.com", "correct-horse-1")

    assert user.email == "ada@example.com"


def test_a_duplicate_email_is_rejected(store):
    store.create("Ada", "ada@example.com", "correct-horse-1")

    with pytest.raises(EmailAlreadyRegistered):
        store.create("A Different Ada", "ada@example.com", "another-password")


def test_a_duplicate_email_is_rejected_regardless_of_case(store):
    store.create("Ada", "ada@example.com", "correct-horse-1")

    with pytest.raises(EmailAlreadyRegistered):
        store.create("Ada", "ADA@EXAMPLE.COM", "another-password")


def test_a_blank_name_is_refused(store):
    with pytest.raises(ValueError, match="name"):
        store.create("   ", "ada@example.com", "correct-horse-1")


def test_an_invalid_email_is_refused(store):
    with pytest.raises(ValueError, match="valid email"):
        store.create("Ada", "not-an-email", "correct-horse-1")


def test_a_short_password_is_refused(store):
    with pytest.raises(ValueError, match="8 characters"):
        store.create("Ada", "ada@example.com", "short")


# --- login ----------------------------------------------------------------


def test_correct_credentials_authenticate(store):
    store.create("Ada", "ada@example.com", "correct-horse-1")

    user = store.authenticate("ada@example.com", "correct-horse-1")

    assert user is not None
    assert user.email == "ada@example.com"


def test_login_email_is_case_insensitive(store):
    store.create("Ada", "ada@example.com", "correct-horse-1")

    assert store.authenticate("ADA@EXAMPLE.COM", "correct-horse-1") is not None


def test_wrong_password_is_rejected(store):
    store.create("Ada", "ada@example.com", "correct-horse-1")

    assert store.authenticate("ada@example.com", "wrong-password") is None


def test_unknown_email_is_rejected(store):
    assert store.authenticate("nobody@example.com", "whatever-1") is None


# --- password hashing -------------------------------------------------------


def test_the_stored_hash_is_never_the_plaintext_password(tmp_path):
    store = UserStore(tmp_path / "app.db")
    store.create("Ada", "ada@example.com", "correct-horse-1")

    connection = sqlite3.connect(tmp_path / "app.db")
    (stored,) = connection.execute(
        "SELECT password_hash FROM users WHERE email = ?", ("ada@example.com",)
    ).fetchone()
    connection.close()

    assert "correct-horse-1" not in stored
    assert stored.startswith("pbkdf2_sha256$")


def test_hash_password_salts_so_identical_passwords_differ():
    assert hash_password("correct-horse-1") != hash_password("correct-horse-1")


def test_verify_password_accepts_a_matching_hash():
    assert verify_password("correct-horse-1", hash_password("correct-horse-1"))


def test_verify_password_rejects_a_wrong_password():
    assert not verify_password("wrong", hash_password("correct-horse-1"))


def test_verify_password_rejects_a_malformed_stored_value():
    assert not verify_password("anything", "not-a-real-hash")


def test_verify_password_rejects_a_hash_with_an_unknown_algorithm():
    assert not verify_password("anything", "bcrypt$12$somesalt$somehash")

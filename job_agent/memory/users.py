"""Persistent user accounts for the login/signup UI.

Reuses the same SQLite file the application tracker already writes to
(`Settings.tracker_db_path`) rather than introducing a second database or a
different engine - the project's storage story stays "one local SQLite
file," just with a second table in it. Plain sqlite3, no ORM, connections
opened per operation - the same pattern `SQLiteApplicationTracker` already
uses (memory/tracker.py), deliberately kept consistent rather than inventing
a second persistence style.

Passwords are never stored in plain text. Hashing uses PBKDF2-HMAC-SHA256
(`hashlib.pbkdf2_hmac`, standard library only - no new dependency needed)
with a random per-user salt and a deliberately slow iteration count. The
stored value is self-describing (`pbkdf2_sha256$<iterations>$<salt>$<hash>`)
so a future algorithm change would not invalidate hashes already on disk.
"""

import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from ..models.user import User

PBKDF2_ITERATIONS = 260_000
_ALGORITHM = "pbkdf2_sha256"


def hash_password(password: str) -> str:
    """A salted, slow hash - never the password itself, never a fast digest."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    )
    return f"{_ALGORITHM}${PBKDF2_ITERATIONS}${salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Whether `password` matches `stored_hash`, using a constant-time
    comparison so a mismatch never leaks timing information about how much
    of the hash matched."""
    try:
        algorithm, iterations, salt, hex_digest = stored_hash.split("$")
    except ValueError:
        return False
    if algorithm != _ALGORITHM:
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), int(iterations)
    )
    return hmac.compare_digest(candidate.hex(), hex_digest)


class EmailAlreadyRegistered(ValueError):
    """Raised by `UserStore.create` when the email is already taken."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
)
"""


class UserStore:
    """Create accounts and authenticate against them. Nothing here ever
    returns a password or a password_hash - only the safe-to-display `User`."""

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

    def create(self, name: str, email: str, password: str) -> User:
        """Register a new account, or raise if the email is already taken.

        Email is normalised (trimmed, lower-cased) before storage and lookup,
        so "Ada@Example.com" and "ada@example.com" are the same account -
        SQLite's UNIQUE constraint is otherwise case-sensitive on TEXT.
        """
        name = name.strip()
        email = email.strip().lower()
        if not name:
            raise ValueError("A name is required.")
        if "@" not in email or not email.split("@")[-1]:
            raise ValueError("Enter a valid email address.")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")

        created_at = datetime.now()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    "INSERT INTO users (name, email, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name, email, hash_password(password), created_at.isoformat()),
                )
                user_id = cursor.lastrowid
        except sqlite3.IntegrityError as exc:
            raise EmailAlreadyRegistered(
                f"An account with '{email}' already exists."
            ) from exc
        return User(id=user_id, name=name, email=email, created_at=created_at)

    def authenticate(self, email: str, password: str) -> User | None:
        """The matching `User` if the credentials are correct, else None -
        deliberately the same outcome whether the email is unknown or the
        password is wrong, so a caller cannot enumerate registered emails."""
        email = email.strip().lower()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, name, email, password_hash, created_at "
                "FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None or not verify_password(password, row["password_hash"]):
            return None
        return User(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            created_at=row["created_at"],
        )

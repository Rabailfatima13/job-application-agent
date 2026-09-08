"""The account behind a login session (mentor-requested login/signup UI).

Deliberately has no password or password_hash field - a `User` is only ever
constructed from data already safe to display, so there is no field here
that could accidentally be serialised or shown on screen. Password handling
lives entirely in `memory/users.py`, never in this model.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    email: str
    created_at: datetime

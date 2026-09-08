"""One saved tailored CV (mentor-requested "multiple tailored CV versions").

Deliberately not named `TailoredCV` - that name is already the writing
agent's in-memory output (`models/outputs.py`), which has no id, no user,
and no notion of being one of several. This is the persisted record built
*from* one, after it has already passed grounding: one immutable row per
successful run, kept apart from the user's baseline CV (`models/baseline_cv.py`)
by construction - a version can be produced only from a real `TailoredCV`,
never from a plain string, so it can never be mistaken for (or promoted
into) the baseline.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class TailoredCVVersion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    user_id: int
    company: str
    role: str
    cv_text: str = Field(
        description="The complete tailored CV, as generated for this role."
    )
    created_at: datetime
    # Identifies which snapshot of the user's baseline CV this version was
    # tailored from - baseline CVs have no separate version number (see
    # BaselineCVStore, one row per user, replaced in place), so its own
    # `updated_at` at generation time is what "the original CV/version"
    # means here. None only for a version saved without that context (e.g.
    # a directly-constructed test record), never for one saved through the
    # real UI flow (see `ui.save_tailored_cv_version`).
    baseline_cv_updated_at: datetime | None = None

"""The user's baseline CV (mentor-requested baseline CV system).

The one persistent, original CV every tailored application is built from -
uploaded once, reused for every job. Deliberately holds only raw text, never
a `TailoredCV`: nothing about this model's shape allows a tailored output to
be mistaken for (or accidentally stored as) the baseline.
"""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class BaselineCV(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int
    cv_text: str = Field(description="The original CV text, verbatim.")
    created_at: datetime
    updated_at: datetime

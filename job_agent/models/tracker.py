"""The persisted application record (FR-7, proposal S13).

Kept separate from outputs.py because this is the one object that outlives a
run: it is what the tracker stores and what the pipeline view lists.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ApplicationStatus(StrEnum):
    draft = "draft"
    ready = "ready"
    submitted = "submitted"
    archived = "archived"


class ApplicationRecord(BaseModel):
    """One tracked application. `submitted` is set by the human, never by the
    agent - the system does not auto-submit anything (FR-14)."""

    model_config = ConfigDict(extra="forbid")

    application_id: str
    role: str
    company: str
    fit_score: float = Field(ge=0.0, le=1.0)
    status: ApplicationStatus = ApplicationStatus.draft
    created_at: datetime

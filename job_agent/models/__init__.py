"""Structured objects exchanged between agents.

Every step of the pipeline emits a schema-constrained object rather than free
text (proposal S13), so the UI, the tracker, and the validation layer all work
against guaranteed shapes.
"""

from .baseline_cv import BaselineCV
from .inputs import CVEvidence, JobDescription, ParsedCV, RoleRequirement
from .outputs import (
    CompanyBrief,
    CoverLetter,
    FitReport,
    MatchLevel,
    RequirementMatch,
    Source,
    TailoredCV,
)
from .tailored_cv_version import TailoredCVVersion
from .tracker import ApplicationRecord, ApplicationStatus
from .user import User

__all__ = [
    "ApplicationRecord",
    "ApplicationStatus",
    "BaselineCV",
    "CVEvidence",
    "CompanyBrief",
    "CoverLetter",
    "FitReport",
    "JobDescription",
    "MatchLevel",
    "ParsedCV",
    "RequirementMatch",
    "RoleRequirement",
    "Source",
    "TailoredCV",
    "TailoredCVVersion",
    "User",
]

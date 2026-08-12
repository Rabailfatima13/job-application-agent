"""Structured objects exchanged between agents.

Every step of the pipeline emits a schema-constrained object rather than free
text (proposal S13), so the UI, the tracker, and the validation layer all work
against guaranteed shapes.
"""

from .inputs import CVEvidence, JobDescription, ParsedCV, RoleRequirement
from .outputs import (
    CompanyBrief,
    CoverLetter,
    FitReport,
    RequirementMatch,
    Source,
    TailoredCV,
)
from .tracker import ApplicationRecord, ApplicationStatus

__all__ = [
    "ApplicationRecord",
    "ApplicationStatus",
    "CVEvidence",
    "CompanyBrief",
    "CoverLetter",
    "FitReport",
    "JobDescription",
    "ParsedCV",
    "RequirementMatch",
    "RoleRequirement",
    "Source",
    "TailoredCV",
]

"""The structured-output contract from the proposal (S13) holds, including the
range and enum guarantees the validation strategy relies on."""

import pytest
from pydantic import ValidationError

from job_agent.models import ApplicationRecord, ApplicationStatus, FitReport


def _fit_report(**overrides) -> dict:
    base = {
        "role": "ML Engineer",
        "company": "Arbisoft",
        "overall_fit": 0.8,
        "requirement_matches": [
            {
                "requirement": "Python",
                "evidence": "Built a FastAPI service",
                "match_level": "match",
                "met": True,
            }
        ],
        "gaps": ["No Kubernetes experience"],
        "recommended_emphasis": ["FastAPI work"],
    }
    return {**base, **overrides}


def test_fit_report_accepts_the_proposal_shape():
    report = FitReport.model_validate(_fit_report())
    assert report.overall_fit == 0.8
    assert report.requirement_matches[0].met is True
    assert report.gaps == ["No Kubernetes experience"]


@pytest.mark.parametrize("score", [-0.1, 1.4])
def test_fit_score_must_be_within_zero_to_one(score):
    with pytest.raises(ValidationError):
        FitReport.model_validate(_fit_report(overall_fit=score))


def test_unexpected_fields_are_rejected():
    # extra="forbid" is what stops a model from inventing its own fields.
    with pytest.raises(ValidationError):
        FitReport.model_validate(_fit_report(verdict="hire"))


def test_tracker_status_must_be_a_known_enum_value(application_record):
    assert application_record.status is ApplicationStatus.draft
    with pytest.raises(ValidationError):
        ApplicationRecord.model_validate(
            {**application_record.model_dump(), "status": "maybe"}
        )

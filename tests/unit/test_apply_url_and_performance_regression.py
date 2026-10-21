import pytest
from unittest.mock import MagicMock, patch
from app.services.url_validator import validate_job_url, validate_jobs_batch, is_synthetic_or_stale_url
from app.services.jobs import get_jobs_from_db, get_job_by_id_from_db
from app.services.dashboard_service import (
    assemble_dashboard_overview,
    get_user_state,
    _USER_DASHBOARD_STATE,
    _OVERVIEW_CACHE
)

def test_url_validator_distinguishes_valid_and_synthetic():
    """Verify that synthetic /job/120 pattern URLs are flagged invalid, while valid HTTPS URLs pass."""
    # Stale synthetic patterns must be rejected
    is_valid, reason = validate_job_url("https://adobe.wd5.myworkdayjobs.com/job/120", check_network=False)
    assert not is_valid
    assert "synthetic" in reason or "pattern" in reason or "stale" in reason

    is_valid_tcs, _ = validate_job_url("https://ibegin.tcs.com/job/111", check_network=False)
    assert not is_valid_tcs

    # Real HTTPS URLs pass syntax/domain checks
    is_valid_real, _ = validate_job_url("https://careers.google.com/jobs/results/123456789/", check_network=False)
    assert is_valid_real

    # Garbage / non-http URLs fail
    is_valid_none, _ = validate_job_url("Application link unavailable", check_network=False)
    assert not is_valid_none

def test_expired_or_unavailable_job_fallback_in_catalog():
    """Verify that static/stale catalog records are marked unavailable and do not present dead URLs."""
    res = get_jobs_from_db(limit=5)
    assert res["total"] > 0
    for job in res["jobs"]:
        # If application_url is synthetic or unavailable, is_available must be False
        if "job/" in str(job.get("application_url", "")).lower() or job.get("application_url") == "Application link unavailable":
            assert job.get("is_available") is False
            assert job["application_url"] == "Application link unavailable"

def test_get_job_by_id_preserves_identity():
    """Verify job identity fields are maintained when fetching details."""
    from app.api.v1.jobs import get_job_by_id
    from fastapi import HTTPException

    # Test static job
    job = get_job_by_id("1")
    assert job["id"] == 1
    assert "job_title" in job
    assert "company" in job
    assert "required_skills" in job
    assert "is_available" in job

    # Test non-existent job returns 404 with helpful message
    with pytest.raises(HTTPException) as exc_info:
        get_job_by_id("99999999")
    assert exc_info.value.status_code == 404
    assert "expired" in exc_info.value.detail or "not found" in exc_info.value.detail.lower()

def test_batch_url_validation_for_displayed_jobs():
    """Verify batch validation processes candidate job lists without excessive waiting."""
    test_jobs = [
        {"job_id": "1", "company": "Adobe", "apply_url": "https://adobe.wd5.myworkdayjobs.com/job/120", "source_type": "static_catalog"},
        {"job_id": "2", "company": "Valid Co", "apply_url": "https://careers.google.com/jobs/results/999", "source_type": "rapidapi_live"},
        {"job_id": "3", "company": "TCS", "apply_url": "https://ibegin.tcs.com/job/111", "source_type": "static_catalog"},
    ]
    validated = validate_jobs_batch(test_jobs)
    assert len(validated) == 3
    # Adobe and TCS synthetic URLs must be marked unavailable
    assert validated[0]["is_available"] is False
    assert validated[2]["is_available"] is False

def test_dashboard_overview_preserves_job_identity_and_skills():
    """Verify assemble_dashboard_overview maps all identity fields, availability, and skills."""
    test_user_id = 8888
    _OVERVIEW_CACHE.clear()
    user_state = get_user_state(test_user_id)
    user_state.clear()
    user_state.update({
        "resume_name": "Test_Engineer.pdf",
        "resume_text": "Python FastAPI AWS PostgreSQL Docker Git Redis Linux CI/CD PyTest SQL",
        "matches": [
            {
                "job_id": "job_123",
                "external_job_id": "ext_987",
                "source": "JSearch",
                "source_type": "rapidapi_live",
                "job_title": "Senior Python Engineer",
                "company": "Tech Corp",
                "location": "Bengaluru, India",
                "match_percentage": 88.5,
                "application_url": "https://techcorp.com/apply/123",
                "matched_skills": ["Python", "AWS", "FastAPI"],
                "missing_skills": ["Docker", "Kubernetes"],
                "is_available": True
            }
        ],
        "discovery_stats": {
            "total_jobs_analyzed": 10,
            "jobs_retrieved": 10,
            "relevant_jobs": 1,
            "matched_jobs": 1,
            "source": "live_jsearch"
        },
        "pipeline_stages": {"embedding": "completed", "hybrid_matching": "completed"}
    })

    mock_db = MagicMock()
    mock_resume = MagicMock()
    mock_resume.id = 1
    mock_resume.filename = "Test_Engineer.pdf"
    mock_resume.extracted_text = "Python FastAPI AWS PostgreSQL Docker Git Redis Linux CI/CD PyTest SQL"
    mock_resume.created_at = None
    mock_resume.updated_at = None
    mock_db.query().filter().order_by().first.return_value = mock_resume

    mock_user = MagicMock()
    mock_user.id = test_user_id

    mock_profile = MagicMock()
    mock_profile.target_role = "Backend Engineer"
    mock_profile.experience_level = "Mid Level"
    mock_profile.min_match_score = 65.0

    with patch("app.services.dashboard_service.get_or_create_user_profile", return_value=mock_profile):
        overview = assemble_dashboard_overview(mock_db, mock_user)

    assert overview.resume.filename == "Test_Engineer.pdf"
    assert len(overview.matches) == 1
    m = overview.matches[0]
    assert m.job_id == "job_123"
    assert m.external_job_id == "ext_987"
    assert m.source == "JSearch"
    assert m.title == "Senior Python Engineer"
    assert m.company == "Tech Corp"
    assert m.location == "Bengaluru, India"
    assert m.score == 88.5
    assert m.is_available is True
    assert m.apply_url == "https://techcorp.com/apply/123"

    # Verify extracted skills in profile
    assert len(overview.profile.skills) >= 5
    assert "python" in overview.profile.skills

def test_early_job_availability_state():
    """Verify that dashboard state exposes matches immediately after hybrid matching without waiting for enrichment."""
    test_user_id = 9999
    _OVERVIEW_CACHE.clear()
    user_state = get_user_state(test_user_id)
    user_state.clear()
    user_state.update({
        "is_processing": True,
        "pipeline_stages": {
            "parsing": "completed",
            "skill_extraction": "completed",
            "candidate_retrieval": "completed",
            "hybrid_matching": "completed",
            "rag_enrichment": "in_progress"
        },
        "matches": [
            {
                "job_id": "job_fast",
                "job_title": "Fast Retrieval Engineer",
                "company": "Fast Corp",
                "match_percentage": 90.0,
                "is_available": True
            }
        ]
    })

    mock_db = MagicMock()
    mock_resume = MagicMock()
    mock_resume.id = 2
    mock_resume.filename = "Fast_Resume.pdf"
    mock_resume.extracted_text = "Fast Retrieval Engineer Python"
    mock_resume.created_at = None
    mock_resume.updated_at = None
    mock_db.query().filter().order_by().first.return_value = mock_resume

    mock_user = MagicMock()
    mock_user.id = test_user_id

    mock_profile = MagicMock()
    mock_profile.target_role = "Fast Retrieval Engineer"
    mock_profile.experience_level = "Senior"
    mock_profile.min_match_score = 65.0

    with patch("app.services.dashboard_service.get_or_create_user_profile", return_value=mock_profile):
        overview = assemble_dashboard_overview(mock_db, mock_user)

    # Even when rag_enrichment is in_progress, matches must already be present and available
    assert len(overview.matches) == 1
    assert overview.matches[0].title == "Fast Retrieval Engineer"
    assert overview.is_processing is True  # still enriching in background

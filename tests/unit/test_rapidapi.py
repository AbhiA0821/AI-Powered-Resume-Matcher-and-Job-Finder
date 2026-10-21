import pytest
from unittest.mock import patch, MagicMock
from app.services.job_sources.rapidapi_job_service import (
    normalize_rapidapi_job,
    deduplicate_jobs,
    RapidAPIJobService,
    to_hireagent_job_dict,
    NormalizedJob
)
from app.core.config import settings

def test_rapidapi_configuration():
    """Validates RapidAPI configuration keys exist in settings."""
    assert hasattr(settings, "RAPIDAPI_KEY")
    assert hasattr(settings, "RAPIDAPI_HOST")
    assert hasattr(settings, "RAPIDAPI_JOB_ENDPOINT")
    assert settings.RAPIDAPI_HOST == "jsearch.p.rapidapi.com"

def test_rapidapi_response_normalization():
    """Validates correct normalization of standard RapidAPI/JSearch job records."""
    raw_job = {
        "job_id": "ext-12345",
        "job_title": "AI Engineer",
        "employer_name": "Tech Corp",
        "job_city": "Bengaluru",
        "job_country": "IN",
        "job_description": "Building state of the art generative AI systems with Python and PyTorch.",
        "job_required_skills": ["Python", "PyTorch", "FastAPI"],
        "job_apply_link": "https://linkedin.com/jobs/view/12345",
        "job_posted_at_datetime_utc": "2026-03-20T10:00:00Z"
    }

    normalized = normalize_rapidapi_job(raw_job)
    assert normalized is not None
    assert normalized.external_job_id == "ext-12345"
    assert normalized.title == "AI Engineer"
    assert normalized.company == "Tech Corp"
    assert "Bengaluru" in normalized.location
    assert any("python" in s.lower() for s in normalized.skills)
    assert normalized.source == "LinkedIn"
    assert normalized.source_url == "https://linkedin.com/jobs/view/12345"
    assert normalized.is_fresh is True
    assert normalized.source_type == "live_api"

def test_missing_api_response_fields():
    """Validates robust handling when API omits optional fields."""
    raw_job = {
        "job_id": "minimal-001",
        "job_title": "Python Developer",
        "employer_name": "Startup Inc"
    }

    normalized = normalize_rapidapi_job(raw_job)
    assert normalized is not None
    assert normalized.title == "Python Developer"
    assert normalized.company == "Startup Inc"
    assert normalized.location == "India"
    assert normalized.source_url is None
    assert normalized.skills == []
    assert normalized.source_type == "live_api"

def test_url_safety_no_fabrication():
    """Ensures application URLs are NEVER fabricated when missing from response."""
    raw_job = {
        "job_id": "no-url-job",
        "job_title": "Data Scientist",
        "employer_name": "Analytics Co"
    }
    normalized = normalize_rapidapi_job(raw_job)
    assert normalized.source_url is None

    h_job = to_hireagent_job_dict(normalized)
    assert h_job["application_url"] == "Application link unavailable"

def test_job_deduplication():
    """Validates multi-tier deduplication prevents duplicate listings."""
    job1 = NormalizedJob(
        external_job_id="j1",
        title="ML Engineer",
        company="Alpha AI",
        location="Bengaluru, India",
        description="ML desc",
        skills=["Python"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/j1",
        source_type="live_api"
    )

    # Duplicate by external_job_id and source
    job2 = NormalizedJob(
        external_job_id="j1",
        title="ML Engineer (Different Title)",
        company="Alpha AI",
        location="Bengaluru, India",
        description="ML desc",
        skills=["Python"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/other",
        source_type="live_api"
    )

    # Duplicate by source_url
    job3 = NormalizedJob(
        external_job_id="j3",
        title="Machine Learning Engineer",
        company="Alpha AI",
        location="Bengaluru, India",
        description="ML desc",
        skills=["Python"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/j1",
        source_type="live_api"
    )

    # Duplicate by normalized title + company + location
    job4 = NormalizedJob(
        external_job_id="j4",
        title="ml engineer",
        company="alpha ai",
        location="Bengaluru, India",
        description="ML desc",
        skills=["Python"],
        source="Naukri",
        source_url="https://naukri.com/jobs/j4",
        source_type="live_api"
    )

    # Distinct job
    job5 = NormalizedJob(
        external_job_id="j5",
        title="Backend Developer",
        company="Beta Corp",
        location="Pune, India",
        description="Backend desc",
        skills=["FastAPI"],
        source="Indeed",
        source_url="https://indeed.com/jobs/j5",
        source_type="live_api"
    )

    unique = deduplicate_jobs([job1, job2, job3, job4, job5])
    assert len(unique) == 2
    assert unique[0].external_job_id == "j1"
    assert unique[1].external_job_id == "j5"

def test_rapidapi_fallback_when_api_fails():
    """Validates that RapidAPIService returns empty list gracefully on network/auth error."""
    service = RapidAPIJobService(api_key="mock_invalid_key")
    with patch("requests.get", side_effect=Exception("Connection refused")):
        results = service.search_jobs(query="Software Engineer", location="India")
        assert results == []

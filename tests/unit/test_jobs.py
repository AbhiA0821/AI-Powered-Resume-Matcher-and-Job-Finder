import pytest
from app.services.job_service import get_jobs_from_db, get_job_by_id_from_db

def test_jobs_database_retrieval():
    """Verifies that jobs catalog returns records with required fields including application_url."""
    result = get_jobs_from_db(limit=10)
    assert "total" in result
    assert "jobs" in result
    assert isinstance(result["jobs"], list)
    assert len(result["jobs"]) > 0

    job = result["jobs"][0]
    assert "id" in job
    assert "job_title" in job
    assert "company" in job
    assert "required_skills" in job
    assert "application_url" in job
    assert job["application_url"] is not None

def test_get_job_by_id():
    """Verifies retrieval of a specific job by integer ID."""
    job = get_job_by_id_from_db(1)
    assert job is not None
    assert job["id"] == 1
    assert "job_title" in job

def test_get_nonexistent_job():
    """Verifies safe handling of non-existent job ID."""
    job = get_job_by_id_from_db(999999)
    assert job is None

from typing import Optional
from fastapi import APIRouter, Query
from app.schemas.job import JobListResponse
from app.services.jobs import get_jobs_from_db

router = APIRouter(prefix="/jobs", tags=["Jobs"])

@router.get("", response_model=JobListResponse, summary="List Job Opportunities")
def list_jobs(
    search: Optional[str] = Query(None, description="Search term for job title, skills, or description"),
    role: Optional[str] = Query(None, description="Target job role filter"),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of job listings to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Fetches job listings from underlying database with search and role filtering."""
    return get_jobs_from_db(search=search, role=role, limit=limit, offset=offset)

@router.get("/quota", summary="JSearch API Quota & Usage Status")
def get_jsearch_quota():
    """Returns real-time monthly usage, limits, and threshold warnings for JSearch API."""
    from app.services.job_sources.rapidapi_job_service import get_jsearch_quota_status
    return get_jsearch_quota_status()

@router.get("/{job_id}", summary="Get Job by ID")
def get_job_by_id(job_id: str):
    """Retrieves specific job listing details by id across local catalog, live cache, and active matches.
    Gracefully handles expired/unavailable jobs with explicit status.
    """
    from fastapi import HTTPException
    from app.services.jobs import get_job_by_id_from_db
    from app.services.job_sources.rapidapi_job_service import find_cached_rapidapi_job, to_hireagent_job_dict
    from app.services.dashboard_service import _USER_DASHBOARD_STATE

    target_id = str(job_id).strip()
    job = None

    # Tier 1: Check SQLite local catalog if id is numeric
    if target_id.isdigit():
        try:
            job = get_job_by_id_from_db(int(target_id))
        except Exception:
            job = None

    # Tier 2: Check RapidAPI live job cache
    if not job:
        cached_rapid = find_cached_rapidapi_job(target_id)
        if cached_rapid:
            job = to_hireagent_job_dict(cached_rapid)

    # Tier 3: Check active user pipeline match states
    if not job:
        for u_id, u_state in _USER_DASHBOARD_STATE.items():
            for m in u_state.get("matches", []):
                m_jid = str(m.get("job_id", "")).strip()
                m_ext = str(m.get("external_job_id", "")).strip()
                m_id = str(m.get("id", "")).strip()
                if target_id in [m_jid, m_ext, m_id] and m_jid:
                    job = {
                        "id": m.get("id") or (int(target_id) if target_id.isdigit() else 0),
                        "job_id": m.get("job_id") or target_id,
                        "external_job_id": m.get("external_job_id") or target_id,
                        "job_title": m.get("job_title", "Software Engineer"),
                        "company": m.get("company", "Company"),
                        "required_skills": ", ".join(m.get("required_skills", [])) if isinstance(m.get("required_skills"), list) else str(m.get("required_skills", "")),
                        "experience_level": m.get("experience_level") or "Not specified",
                        "job_role": m.get("job_role") or m.get("job_title"),
                        "description": m.get("recommendation") or f"Position at {m.get('company')}",
                        "application_url": m.get("application_url") or m.get("apply_url") or "Application link unavailable",
                        "canonical_job_url": m.get("canonical_job_url") or m.get("application_url"),
                        "location": m.get("location") or "India",
                        "source": m.get("source") or "HireAgent Test Catalog",
                        "source_type": m.get("source_type") or "static_catalog",
                        "posted_at": m.get("posted_at") or ""
                    }
                    break
            if job:
                break

    from app.services.url_validator import validate_job_url

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job {job_id} not found or no longer available (expired)"
        )

    # Ensure application_url and is_available are verified
    app_url = job.get("application_url") or job.get("apply_url")
    is_avail = job.get("is_available")
    if is_avail is None:
        if job.get("source_type") == "static_catalog":
            is_avail = False
        else:
            is_avail, _ = validate_job_url(app_url, check_network=True)
    
    job["is_available"] = bool(is_avail)
    if not is_avail:
        job["application_url"] = "Application link unavailable"

    return job

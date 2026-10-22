import pytest
from app.services.job_sources.rapidapi_job_service import (
    extract_best_apply_url,
    normalize_rapidapi_job,
    to_hireagent_job_dict,
    find_cached_rapidapi_job,
    RapidAPIJob,
    _JOB_CACHE
)
from app.services.matching_service import match_and_rank_jobs, apply_employer_diversity
from app.api.v1.jobs import get_job_by_id
from fastapi import HTTPException

def test_apply_url_priority_direct_ats():
    """Verifies that extract_best_apply_url prefers direct ATS application link over generic links."""
    raw_job = {
        "job_id": "test-ats-1",
        "job_title": "Senior Python Developer",
        "employer_name": "InnoCorp",
        "job_apply_link": "https://innocorp.com/careers",
        "job_google_link": "https://www.google.com/search?q=python+job",
        "apply_options": [
            {"publisher": "Google", "apply_link": "https://www.google.com/search?q=python+job", "is_direct": False},
            {"publisher": "Greenhouse", "apply_link": "https://boards.greenhouse.io/innocorp/jobs/567890", "is_direct": True}
        ]
    }
    best_url, _ = extract_best_apply_url(raw_job)
    assert best_url == "https://boards.greenhouse.io/innocorp/jobs/567890"

def test_apply_url_direct_posting_over_generic_homepage():
    """Verifies that specific job posting URL is preferred over root domain homepage."""
    raw_job = {
        "job_id": "test-posting-1",
        "job_title": "AI Engineer",
        "employer_name": "StartupLab",
        "job_apply_link": "https://startuplab.ai",  # root domain
        "apply_options": [
            {"publisher": "Lever", "apply_link": "https://jobs.lever.co/startuplab/ai-engineer-101", "is_direct": False}
        ]
    }
    best_url, _ = extract_best_apply_url(raw_job)
    assert best_url == "https://jobs.lever.co/startuplab/ai-engineer-101"

def test_job_id_consistency_across_layers():
    """Verifies that external job ID and identity are preserved across dictionary transforms."""
    raw_job = {
        "job_id": "jsearch-india-9988",
        "job_title": "Backend Engineer",
        "employer_name": "FinTech Startup",
        "job_city": "Bengaluru",
        "job_country": "IN",
        "job_apply_link": "https://jobs.lever.co/fintech/9988"
    }
    normalized = normalize_rapidapi_job(raw_job)
    assert normalized.external_job_id == "jsearch-india-9988"

    h_dict = to_hireagent_job_dict(normalized)
    assert h_dict["external_job_id"] == "jsearch-india-9988"
    assert h_dict["job_id"] == "jsearch-india-9988"
    assert h_dict["application_url"] == "https://jobs.lever.co/fintech/9988"
    assert h_dict["id"] > 0

    # Pass through matching
    ranked = match_and_rank_jobs(
        resume_text="Experienced Backend Engineer proficient in Python and FastAPI.",
        jobs=[h_dict],
        top_k=1
    )
    assert len(ranked) == 1
    assert ranked[0]["job_id"] == "jsearch-india-9988"
    assert ranked[0]["external_job_id"] == "jsearch-india-9988"
    assert ranked[0]["application_url"] == "https://jobs.lever.co/fintech/9988"

def test_company_diversity_cap():
    """Verifies that apply_employer_diversity prevents a single company from dominating Top 10."""
    candidates = []
    # 5 jobs from GiantCorp
    for i in range(5):
        candidates.append({
            "id": i + 1,
            "job_id": f"giant-{i}",
            "job_title": f"Software Engineer {i}",
            "company": "GiantCorp",
            "match_percentage": 95.0 - i,
            "required_skills": "Python, SQL",
            "application_url": f"https://giantcorp.com/job/{i}"
        })
    # 2 jobs from StartupAlpha
    for i in range(2):
        candidates.append({
            "id": 10 + i,
            "job_id": f"alpha-{i}",
            "job_title": f"Python Developer {i}",
            "company": "StartupAlpha",
            "match_percentage": 88.0 - i,
            "required_skills": "Python",
            "application_url": f"https://startupalpha.io/job/{i}"
        })
    # 2 jobs from ProductBeta
    for i in range(2):
        candidates.append({
            "id": 20 + i,
            "job_id": f"beta-{i}",
            "job_title": f"Backend Engineer {i}",
            "company": "ProductBeta",
            "match_percentage": 85.0 - i,
            "required_skills": "Python",
            "application_url": f"https://productbeta.com/job/{i}"
        })
    # 2 jobs from ServiceGamma
    for i in range(2):
        candidates.append({
            "id": 30 + i,
            "job_id": f"gamma-{i}",
            "job_title": f"Full Stack Engineer {i}",
            "company": "ServiceGamma",
            "match_percentage": 80.0 - i,
            "required_skills": "Python",
            "application_url": f"https://servicegamma.com/job/{i}"
        })

    diverse = apply_employer_diversity(candidates, top_k=7, max_per_employer=2)

    # GiantCorp must have at most 2 entries in top 7 despite having the 5 highest scores
    giant_count = sum(1 for j in diverse if j["company"] == "GiantCorp")
    assert giant_count <= 2

    # Result should include StartupAlpha and ProductBeta
    companies_present = {j["company"] for j in diverse}
    assert "StartupAlpha" in companies_present
    assert "ProductBeta" in companies_present

def test_duplicate_company_handling_relevance_first():
    """Verifies that if candidate pool has few distinct companies, deferred jobs backfill slots."""
    candidates = []
    # 4 jobs from OnlyTwoCorp A
    for i in range(4):
        candidates.append({
            "id": i + 1,
            "job_id": f"a-{i}",
            "job_title": f"Role A{i}",
            "company": "Company A",
            "match_percentage": 90.0 - i,
            "required_skills": "Python"
        })
    # 2 jobs from Company B
    for i in range(2):
        candidates.append({
            "id": 10 + i,
            "job_id": f"b-{i}",
            "job_title": f"Role B{i}",
            "company": "Company B",
            "match_percentage": 85.0 - i,
            "required_skills": "Python"
        })

    diverse = apply_employer_diversity(candidates, top_k=5, max_per_employer=2)
    # Total selected must be 5 (2 from A, 2 from B, plus 1 backfilled from A)
    assert len(diverse) == 5
    assert diverse[0]["company"] == "Company A"
    assert diverse[1]["company"] == "Company A"

def test_same_job_search_consistency():
    """Verifies that the same job can be looked up by external ID or cached ID."""
    cached_job = RapidAPIJob(
        external_job_id="live-consist-1234",
        title="Senior Python Architect",
        company="PayTech Solutions",
        location="Pune, India",
        description="Core payments engine architecture.",
        skills=["Python", "FastAPI"],
        source="LinkedIn",
        source_url="https://paytech.com/careers/1234",
        source_type="live_api"
    )
    _JOB_CACHE["test:key:1"] = (9999999999.0, [cached_job])

    found = find_cached_rapidapi_job("live-consist-1234")
    assert found is not None
    assert found.title == "Senior Python Architect"
    assert found.company == "PayTech Solutions"

    # API lookup
    api_job = get_job_by_id("live-consist-1234")
    assert api_job is not None
    assert api_job["company"] == "PayTech Solutions"
    assert api_job["application_url"] == "https://paytech.com/careers/1234"

def test_expired_job_handling_graceful_404():
    """Verifies that requesting an unavailable/expired job returns HTTP 404 with clear status."""
    with pytest.raises(HTTPException) as exc_info:
        get_job_by_id("non-existent-expired-job-id-99999")
    assert exc_info.value.status_code == 404
    assert "not found or no longer available" in exc_info.value.detail

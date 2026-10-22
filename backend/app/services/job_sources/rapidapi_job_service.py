import os
import re
import time
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field
import requests

from app.core.config import settings
from app.services.skill_service import extract_skills
from .usage_tracker import get_usage_tracker, get_jsearch_quota_status, JSearchUsageTracker

logger = logging.getLogger("hireagent.rapidapi")

class RapidAPIJob(BaseModel):
    """Normalized internal representation of a live external job from RapidAPI."""
    external_job_id: str
    title: str
    company: str
    location: str
    description: str
    skills: List[str] = Field(default_factory=list)
    experience: str = ""
    employment_type: str = "Full-time"
    salary: str = "Not specified"
    source: str = "other"  # linkedin, naukri, indeed, glassdoor, etc.
    source_url: Optional[str] = None
    is_direct: bool = True
    posted_at: Optional[str] = None
    fetched_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_fresh: bool = True
    source_type: str = "live_api"

# Simple in-memory cache with TTL (15 minutes)
_JOB_CACHE: Dict[str, Tuple[float, List[RapidAPIJob]]] = {}
CACHE_TTL_SECONDS = 900  # 15 minutes

def _get_cache_key(query: str, location: str, source: Optional[str], page: int) -> str:
    q = (query or "").strip().lower()
    loc = (location or "").strip().lower()
    src = (source or "all").strip().lower()
    return f"{q}:{loc}:{src}:{page}"

def _detect_source_platform(raw_job: Dict[str, Any]) -> str:
    """Identifies the originating job portal (LinkedIn, Naukri, Indeed, etc.) from provider metadata."""
    publisher = str(raw_job.get("job_publisher") or "").lower()
    apply_link = str(raw_job.get("job_apply_link") or "").lower()
    source_field = str(raw_job.get("source") or "").lower()

    combined = f"{publisher} {apply_link} {source_field}"
    if "linkedin" in combined:
        return "LinkedIn"
    elif "naukri" in combined:
        return "Naukri"
    elif "indeed" in combined:
        return "Indeed"
    elif "glassdoor" in combined:
        return "Glassdoor"
    elif "internshala" in combined:
        return "Internshala"
    elif publisher.strip():
        return publisher.strip().capitalize()
    return "Web"

def extract_best_apply_url(raw_job: Dict[str, Any]) -> Tuple[Optional[str], bool]:
    """Extracts the most direct application URL or valid job-listing URL.
    Returns (url: Optional[str], is_direct: bool).
    - Direct employer application link (is_direct == True) or known ATS -> (url, True)
    - Valid job listing portal URL -> (url, False)
    - None if unavailable -> (None, False)
    """
    ats_domains = (
        "boards.greenhouse.io", "greenhouse.io",
        "jobs.lever.co", "lever.co",
        "myworkdayjobs.com", "workday.com",
        "ashbyhq.com",
        "jobs.smartrecruiters.com", "smartrecruiters.com",
        "bamboohr.com",
        "apply.workable.com", "workable.com",
        "jobs.jobvite.com", "jobvite.com",
        "icims.com",
        "recruitee.com",
        "breezy.hr",
        "freshteam.com"
    )

    def is_valid_http(url: Any) -> bool:
        if not url or not isinstance(url, str):
            return False
        u = url.strip()
        return u.startswith("http://") or u.startswith("https://")

    def is_generic_root(url: str) -> bool:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path.strip("/")
            return not path and not parsed.query
        except Exception:
            return False

    apply_options = raw_job.get("apply_options") or []
    if isinstance(apply_options, list):
        # Priority 1: is_direct == True in apply_options
        for opt in apply_options:
            if isinstance(opt, dict) and opt.get("is_direct") is True:
                link = opt.get("apply_link")
                if is_valid_http(link):
                    return str(link).strip(), True

        # Priority 2: Known ATS platforms in apply_options
        for opt in apply_options:
            if isinstance(opt, dict):
                link = opt.get("apply_link")
                if is_valid_http(link):
                    link_str = str(link).strip().lower()
                    if any(ats in link_str for ats in ats_domains):
                        return str(link).strip(), True

    # Priority 3: job_apply_is_direct flag
    job_apply_link = raw_job.get("job_apply_link")
    if raw_job.get("job_apply_is_direct") is True and is_valid_http(job_apply_link):
        return str(job_apply_link).strip(), True

    # Priority 4: Specific job links in apply_options (not generic homepages)
    if isinstance(apply_options, list):
        for opt in apply_options:
            if isinstance(opt, dict):
                link = opt.get("apply_link")
                if is_valid_http(link) and not is_generic_root(str(link).strip()):
                    link_str = str(link).strip()
                    if "google.com/search" not in link_str.lower():
                        return link_str, False

    # Priority 5: job_apply_link
    if is_valid_http(job_apply_link) and not is_generic_root(str(job_apply_link).strip()):
        return str(job_apply_link).strip(), False

    # Priority 6: Fallback candidate fields
    for field in ["apply_link", "url", "link"]:
        candidate = raw_job.get(field)
        if is_valid_http(candidate) and not is_generic_root(str(candidate).strip()):
            return str(candidate).strip(), False

    if is_valid_http(job_apply_link):
        return str(job_apply_link).strip(), False

    if isinstance(apply_options, list):
        for opt in apply_options:
            if isinstance(opt, dict) and is_valid_http(opt.get("apply_link")):
                return str(opt.get("apply_link")).strip(), False

    # Last resort fallback: google link
    google_link = raw_job.get("job_google_link")
    if is_valid_http(google_link):
        return str(google_link).strip(), False

    return None, False

def normalize_rapidapi_job(raw_job: Dict[str, Any]) -> RapidAPIJob:
    """Maps diverse RapidAPI job provider formats (e.g. JSearch, Job Search API) to the unified schema.
    Never invents URLs, companies, or salary data.
    """
    now_iso = datetime.now(timezone.utc).isoformat()

    # Extract ID
    external_id = (
        raw_job.get("job_id") or
        raw_job.get("id") or
        raw_job.get("jobId") or
        str(hash(f"{raw_job.get('job_title', '')}{raw_job.get('employer_name', '')}"))
    )

    # Extract Title & Company
    title = (
        raw_job.get("job_title") or
        raw_job.get("title") or
        raw_job.get("jobTitle") or
        "Job Title Unavailable"
    ).strip()

    company = (
        raw_job.get("employer_name") or
        raw_job.get("company_name") or
        raw_job.get("company") or
        raw_job.get("employerName") or
        "Company Name Unavailable"
    ).strip()

    # Location handling (India default)
    city = raw_job.get("job_city") or raw_job.get("city") or ""
    state = raw_job.get("job_state") or raw_job.get("state") or ""
    country = raw_job.get("job_country") or raw_job.get("country") or "India"
    loc_parts = [p for p in [city, state, country] if p and p.strip()]
    location = ", ".join(loc_parts) if loc_parts else (raw_job.get("location") or "India")

    # Description
    description = (
        raw_job.get("job_description") or
        raw_job.get("description") or
        raw_job.get("snippet") or
        ""
    ).strip()

    # Verified application link (prioritizes direct job posting, never hallucinated)
    clean_url, is_direct = extract_best_apply_url(raw_job)

    # Source Platform
    source = _detect_source_platform(raw_job)

    # Skills extraction from job highlights or description
    parsed_skills: List[str] = []
    highlights = raw_job.get("job_highlights") or {}
    if isinstance(highlights, dict):
        qualifications = highlights.get("Qualifications") or []
        if isinstance(qualifications, list):
            qual_text = " ".join(qualifications)
            parsed_skills.extend(extract_skills(qual_text))
    
    # Extract skills directly from description using canonical skill extractor
    if description:
        parsed_skills.extend(extract_skills(description))
    
    # Deduplicate skills preserving case
    unique_skills = sorted(list(set(parsed_skills)))

    # Experience level
    exp_required = raw_job.get("job_required_experience") or {}
    experience = "Fresher / 0-2 years"
    if isinstance(exp_required, dict):
        months = exp_required.get("required_experience_in_months")
        if months is not None:
            years = months // 12
            experience = f"{years} years" if years > 0 else "Fresher (<1 year)"
    elif raw_job.get("experience"):
        experience = str(raw_job.get("experience"))

    # Employment type
    employment_type = (
        raw_job.get("job_employment_type") or
        raw_job.get("employment_type") or
        "Full-time"
    )

    # Salary
    salary = "Not disclosed"
    min_sal = raw_job.get("job_min_salary")
    max_sal = raw_job.get("job_max_salary")
    currency = raw_job.get("job_salary_currency") or "INR"
    if min_sal and max_sal:
        salary = f"{currency} {min_sal:,.0f} - {max_sal:,.0f}"
    elif min_sal:
        salary = f"{currency} {min_sal:,.0f}+"

    # Posted date
    posted_at = (
        raw_job.get("job_posted_at_datetime_utc") or
        raw_job.get("posted_at") or
        raw_job.get("job_posted_at_timestamp")
    )
    if posted_at:
        posted_at = str(posted_at)

    return RapidAPIJob(
        external_job_id=str(external_id),
        title=title,
        company=company,
        location=location,
        description=description,
        skills=unique_skills,
        experience=experience,
        employment_type=str(employment_type).replace("_", " ").title(),
        salary=salary,
        source=source,
        source_url=clean_url,
        is_direct=is_direct,
        posted_at=posted_at,
        fetched_at=now_iso,
        is_fresh=True,
        source_type="live_api"
    )

def deduplicate_jobs(jobs: List[RapidAPIJob]) -> List[RapidAPIJob]:
    """Applies a multi-tier deduplication strategy:
    1. external_job_id + source
    2. source_url (if valid)
    3. normalized (company, title, location)
    """
    seen_ids = set()
    seen_urls = set()
    seen_triples = set()
    unique: List[RapidAPIJob] = []

    for j in jobs:
        # Tier 1: external_job_id + source
        id_key = f"{j.source}:{j.external_job_id}"
        if id_key in seen_ids:
            continue

        # Tier 2: source_url
        if j.source_url and j.source_url in seen_urls:
            continue

        # Tier 3: company + title + location
        clean_comp = re.sub(r'[^a-zA-Z0-9]', '', j.company.lower())
        clean_title = re.sub(r'[^a-zA-Z0-9]', '', j.title.lower())
        clean_loc = re.sub(r'[^a-zA-Z0-9]', '', j.location.lower()[:15])
        triple_key = f"{clean_comp}:{clean_title}:{clean_loc}"
        if triple_key in seen_triples:
            continue

        seen_ids.add(id_key)
        if j.source_url:
            seen_urls.add(j.source_url)
        seen_triples.add(triple_key)
        unique.append(j)

    return unique

def search_live_jobs(
    query: str,
    location: str = "India",
    source: Optional[str] = None,
    page: int = 1,
    num_pages: int = 1,
    date_posted: str = "all"
) -> List[RapidAPIJob]:
    """Fetches live India job postings from RapidAPI provider with deduplication and caching.
    Falls back cleanly to empty list if API key is missing or request fails.
    """
    api_key = settings.RAPIDAPI_KEY or os.getenv("RAPIDAPI_KEY")
    if not api_key or not api_key.strip():
        logger.info("RapidAPI key not configured; skipping live external search.")
        return []

    cache_key = _get_cache_key(query, location, source, page)
    now = time.time()
    if cache_key in _JOB_CACHE:
        cached_time, cached_results = _JOB_CACHE[cache_key]
        if now - cached_time < CACHE_TTL_SECONDS:
            logger.info(f"Returning {len(cached_results)} cached RapidAPI jobs for query '{query}'")
            return cached_results

    # Check JSearch monthly quota tracking (200 requests/month limit)
    tracker = get_usage_tracker()
    allowed, usage_info, _ = tracker.acquire_request_slot()
    if not allowed:
        logger.warning(
            f"Monthly JSearch quota reached ({usage_info.get('requests_used')}/{usage_info.get('monthly_limit')}). "
            "Blocking outbound request and using fallback."
        )
        return []

    # Build query string tailored for India jobs
    full_query = query.strip()
    if location and location.lower() not in full_query.lower():
        full_query = f"{full_query} in {location.strip()}"
    if source and source.lower() not in ["all", "any", "other"]:
        full_query = f"{full_query} site:{source.lower().strip()}.com"

    endpoint = settings.RAPIDAPI_JOB_ENDPOINT or "https://jsearch.p.rapidapi.com/search"
    host = settings.RAPIDAPI_HOST or "jsearch.p.rapidapi.com"

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host
    }

    params = {
        "query": full_query,
        "page": str(page),
        "num_pages": str(num_pages),
        "date_posted": date_posted,
        "country": "in"
    }

    try:
        rapid_timeout = float(os.getenv("RAPIDAPI_TIMEOUT", "20.0"))
        resp = requests.get(endpoint, headers=headers, params=params, timeout=rapid_timeout)
        
        if resp.status_code == 429:
            logger.warning("RapidAPI rate limit exceeded (HTTP 429). Falling back gracefully.")
            return []
        
        if resp.status_code != 200:
            logger.warning(f"RapidAPI responded with HTTP {resp.status_code}. Falling back gracefully.")
            return []

        data = resp.json()
        raw_jobs = []
        if isinstance(data, dict):
            raw_jobs = data.get("data") or data.get("jobs") or data.get("results") or []
        elif isinstance(data, list):
            raw_jobs = data

        normalized = [normalize_rapidapi_job(item) for item in raw_jobs if isinstance(item, dict)]
        deduped = deduplicate_jobs(normalized)

        # Update cache
        _JOB_CACHE[cache_key] = (now, deduped)
        logger.info(f"Successfully retrieved and normalized {len(deduped)} fresh live jobs from RapidAPI")
        return deduped

    except requests.exceptions.Timeout:
        logger.warning("RapidAPI request timed out. Using fallback.")
        return []
    except Exception as e:
        logger.warning(f"RapidAPI job retrieval failed: {e}. Using fallback.")
        return []

def to_hireagent_job_dict(job: RapidAPIJob, auto_id: int = 0, role_category: Optional[str] = None) -> Dict[str, Any]:
    """Converts a normalized RapidAPIJob into the dictionary format expected by HireAgent's matching engine.
    Preserves original external job ID, canonical job URL, and direct application link.
    """
    skills_str = ", ".join(job.skills) if job.skills else "Software Engineering"
    
    # Stable integer ID for database and schema compliance
    if auto_id > 0:
        stable_id = auto_id
    elif job.external_job_id.isdigit():
        stable_id = int(job.external_job_id)
    else:
        stable_id = (abs(hash(job.external_job_id)) % 900000) + 100000

    has_url = bool(job.source_url and str(job.source_url).strip().startswith(("http://", "https://")) and job.source_url != "Application link unavailable")
    app_url = job.source_url if has_url else "Application link unavailable"

    return {
        "id": stable_id,
        "job_id": job.external_job_id,
        "external_job_id": job.external_job_id,
        "job_title": job.title,
        "title": job.title,
        "company": job.company,
        "required_skills": skills_str,
        "experience_level": job.experience,
        "job_role": role_category or job.title,
        "role_category": role_category or job.title,
        "description": job.description,
        "canonical_job_url": job.source_url,
        "application_url": app_url,
        "apply_url": app_url,
        "source": job.source,
        "source_type": job.source_type,
        "location": job.location,
        "posted_at": job.posted_at or "",
        "is_available": has_url,
        "is_direct": getattr(job, "is_direct", True),
        "url_status": "active" if has_url else "unavailable"
    }

def find_cached_rapidapi_job(job_id: str) -> Optional[RapidAPIJob]:
    """Searches memory cache for a job matching external_job_id, stable id, or title/company."""
    if not job_id:
        return None
    target_id = str(job_id).strip()
    for _, (_, jobs_list) in _JOB_CACHE.items():
        for j in jobs_list:
            if j.external_job_id == target_id:
                return j
            computed_id = str((abs(hash(j.external_job_id)) % 900000) + 100000)
            if computed_id == target_id:
                return j
    return None

NormalizedJob = RapidAPIJob

class RapidAPIJobService:
    """Service wrapper for interacting with RapidAPI job endpoints."""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.RAPIDAPI_KEY or os.getenv("RAPIDAPI_KEY")

    def search_jobs(self, query: str, location: str = "India", **kwargs) -> List[RapidAPIJob]:
        # Temporarily use self.api_key if passed
        orig_key = settings.RAPIDAPI_KEY
        try:
            if self.api_key:
                settings.RAPIDAPI_KEY = self.api_key
            return search_live_jobs(query=query, location=location, **kwargs)
        finally:
            settings.RAPIDAPI_KEY = orig_key

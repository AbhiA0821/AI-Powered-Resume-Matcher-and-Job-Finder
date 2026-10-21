"""
Job URL Validation and Usability Service for HireAgent.
Adheres strictly to production requirements:
- Validates that application URLs point to usable, active job application pages.
- Detects obvious 404/not-found pages, expired postings, and stale synthetic links.
- Uses fast, lightweight HTTP checks with realistic headers and strict timeouts.
- Caches validation results to avoid repetitive network calls.
- Validates only top final displayed jobs concurrently (never hundreds).
- Never fabricates or guesses URLs.
"""

import time
import logging
import concurrent.futures
from urllib.parse import urlparse
from typing import Dict, Any, List, Optional, Tuple
import requests

logger = logging.getLogger("hireagent.url_validator")

# In-memory cache for URL validation results: url -> (timestamp, is_valid, reason)
_URL_VALIDATION_CACHE: Dict[str, Tuple[float, bool, str]] = {}
URL_CACHE_TTL_SECONDS = 3600.0  # 1 hour cache

# Known synthetic placeholder patterns (e.g. from static catalog seeds)
SYNTHETIC_PATTERNS = [
    "/job/101", "/job/102", "/job/103", "/job/104", "/job/105",
    "/job/106", "/job/107", "/job/108", "/job/109", "/job/110",
    "/job/111", "/job/112", "/job/113", "/job/114", "/job/115",
    "/job/116", "/job/117", "/job/118", "/job/119", "/job/120",
    "/job/201", "/job/202", "/job/203", "/job/204", "/job/205",
    "/job/206", "/job/207", "/job/208", "/job/209", "/job/210",
    "example.com", "test.com"
]

def is_synthetic_or_stale_url(url: str) -> bool:
    """Checks if a URL matches known synthetic test catalog patterns."""
    if not url or not isinstance(url, str):
        return True
    u_lower = url.strip().lower()
    for pattern in SYNTHETIC_PATTERNS:
        if pattern in u_lower:
            return True
    return False

def is_generic_homepage(url: str) -> bool:
    """Determines if a URL points only to a generic root homepage rather than a job posting."""
    try:
        parsed = urlparse(url.strip())
        path = parsed.path.strip("/")
        return not path and not parsed.query
    except Exception:
        return True

def validate_job_url(url: Optional[str], check_network: bool = True) -> Tuple[bool, str]:
    """
    Validates whether a job application URL is active and usable.
    Returns (is_valid: bool, status_reason: str).
    """
    if not url or not isinstance(url, str):
        return False, "missing_url"

    clean_url = url.strip()
    if not (clean_url.startswith("http://") or clean_url.startswith("https://")):
        return False, "invalid_scheme"

    if clean_url == "Application link unavailable":
        return False, "unavailable"

    # Block generic root homepages (requirement: NOT company homepage)
    if is_generic_homepage(clean_url):
        return False, "generic_homepage"

    # Block known stale synthetic patterns
    if is_synthetic_or_stale_url(clean_url):
        return False, "stale_placeholder"

    # Check validation cache
    now = time.time()
    if clean_url in _URL_VALIDATION_CACHE:
        cached_ts, cached_valid, cached_reason = _URL_VALIDATION_CACHE[clean_url]
        if now - cached_ts < URL_CACHE_TTL_SECONDS:
            return cached_valid, cached_reason

    if not check_network:
        return True, "format_valid"

    # Fast network verification with realistic browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
    }

    try:
        # First attempt HEAD request with short 2.0s timeout
        try:
            resp = requests.head(clean_url, headers=headers, timeout=2.0, allow_redirects=True)
            status_code = resp.status_code
        except requests.exceptions.RequestException:
            # Fall back to quick stream GET
            resp = requests.get(clean_url, headers=headers, timeout=2.0, allow_redirects=True, stream=True)
            status_code = resp.status_code

        # Obvious 404 / 410 not found
        if status_code in [404, 410]:
            _URL_VALIDATION_CACHE[clean_url] = (now, False, f"http_{status_code}")
            return False, f"http_{status_code}"

        # 403 / 999 are common anti-bot responses from LinkedIn/Glassdoor when accessed via script,
        # but the URLs are valid and accessible for human users in a browser.
        # Status 200, 201, 301, 302, 307, 308, 403 are treated as usable.
        if status_code in [200, 201, 204, 301, 302, 307, 308, 403]:
            # Check if redirected to a generic error page
            final_url = getattr(resp, "url", clean_url).lower()
            if any(err_term in final_url for err_term in ["/404", "not-found", "pagenotfound", "job-not-found"]):
                _URL_VALIDATION_CACHE[clean_url] = (now, False, "redirected_to_404")
                return False, "redirected_to_404"

            _URL_VALIDATION_CACHE[clean_url] = (now, True, "active")
            return True, "active"

        # Any other 5xx or unhandled code
        if status_code >= 500:
            _URL_VALIDATION_CACHE[clean_url] = (now, False, f"http_{status_code}")
            return False, f"http_{status_code}"

        _URL_VALIDATION_CACHE[clean_url] = (now, True, f"http_{status_code}")
        return True, f"http_{status_code}"

    except requests.exceptions.Timeout:
        # On network timeout, do not block user; allow with cautious status if structure is valid
        logger.debug(f"URL validation timed out for {clean_url}; keeping formatted link.")
        _URL_VALIDATION_CACHE[clean_url] = (now, True, "timeout_fallback")
        return True, "timeout_fallback"
    except Exception as e:
        logger.debug(f"URL validation error for {clean_url}: {e}")
        _URL_VALIDATION_CACHE[clean_url] = (now, False, "connection_failed")
        return False, "connection_failed"

def validate_jobs_batch(jobs: List[Dict[str, Any]], max_to_validate: int = 10) -> List[Dict[str, Any]]:
    """
    Validates URLs for the top N jobs concurrently in a ThreadPoolExecutor.
    Updates each job with:
    - 'is_available': bool
    - 'url_status': str
    - if unavailable, leaves application_url intact or None so UI can show 'JOB NO LONGER AVAILABLE'.
    """
    if not jobs:
        return []

    target_slice = jobs[:max_to_validate]
    remaining = jobs[max_to_validate:]

    def _validate_single(job: Dict[str, Any]) -> Dict[str, Any]:
        job_copy = dict(job)
        url = job_copy.get("application_url") or job_copy.get("apply_url")
        
        # If source is static catalog with synthetic url, explicitly mark unavailable
        source_type = job_copy.get("source_type", "")
        if source_type == "static_catalog" and is_synthetic_or_stale_url(str(url)):
            job_copy["is_available"] = False
            job_copy["url_status"] = "stale_catalog_record"
            return job_copy

        is_valid, reason = validate_job_url(url, check_network=True)
        job_copy["is_available"] = is_valid
        job_copy["url_status"] = reason
        return job_copy

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(target_slice))) as executor:
            validated_top = list(executor.map(_validate_single, target_slice))
    except Exception as e:
        logger.warning(f"Batch URL validation error: {e}")
        validated_top = [_validate_single(j) for j in target_slice]

    # For remaining jobs beyond top N, perform fast structural check without network overhead
    processed_remaining = []
    for j in remaining:
        j_copy = dict(j)
        u = j_copy.get("application_url") or j_copy.get("apply_url")
        if j_copy.get("source_type") == "static_catalog" and is_synthetic_or_stale_url(str(u)):
            j_copy["is_available"] = False
            j_copy["url_status"] = "stale_catalog_record"
        else:
            is_valid, reason = validate_job_url(u, check_network=False)
            j_copy["is_available"] = is_valid
            j_copy["url_status"] = reason
        processed_remaining.append(j_copy)

    return validated_top + processed_remaining

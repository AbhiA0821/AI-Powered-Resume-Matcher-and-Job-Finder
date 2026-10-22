import os
from typing import Dict, Any, Optional, List, Tuple
import requests

def _get_api_base_url() -> str:
    url = os.environ.get("BACKEND_URL") or os.environ.get("HIREAGENT_API_URL") or "http://127.0.0.1:8000"
    url = url.strip().rstrip("/")
    # On Render Free tier, internal service hostnames (e.g. 'hireagent-backend:10000' or 'hireagent-backend')
    # cannot be resolved via DNS because private networking is disabled.
    # Automatically resolve to the public production backend URL.
    if "hireagent-backend" in url and "onrender.com" not in url:
        return "https://hireagent-backend.onrender.com"
    if not url.startswith("http://") and not url.startswith("https://"):
        # If it contains onrender.com or a non-IP domain, default to https
        if "onrender.com" in url or ("." in url and not url.split(":")[0].replace(".", "").isdigit()):
            url = f"https://{url}"
        else:
            url = f"http://{url}"
    # Strip invalid port suffix on public onrender.com domains if present
    if "onrender.com" in url and ":10000" in url:
        url = url.replace(":10000", "")
    return url

def get_api_base_url() -> str:
    """Returns dynamically resolved backend base URL."""
    return _get_api_base_url()

API_BASE_URL = _get_api_base_url()

def _get_headers(token: Optional[str] = None) -> Dict[str, str]:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers

def _safe_json_response(r: requests.Response, default_error: str) -> Tuple[Optional[Any], Optional[str]]:
    """Safely extracts JSON response or user-friendly error message, preventing JSONDecodeError."""
    try:
        data = r.json()
    except Exception:
        # Non-JSON response (e.g. Render 502/504 gateway timeout, HTML error page, or empty body)
        if 200 <= r.status_code < 300:
            return None, "Received invalid response from server."
        if r.status_code == 502:
            return None, "Backend service is temporarily unavailable (502 Bad Gateway). Please try again shortly."
        if r.status_code == 504:
            return None, "Backend request timed out (504 Gateway Timeout). Please try again shortly."
        if r.status_code == 500:
            return None, "Internal server error. Please try again shortly."
        return None, f"Server returned error HTTP {r.status_code}."

    if 200 <= r.status_code < 300:
        return data, None

    if isinstance(data, dict):
        err = data.get("detail") or data.get("error") or data.get("message") or default_error
        return None, str(err)
    return None, default_error

def check_backend_health() -> bool:
    """Checks if the FastAPI backend is running and healthy."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            try:
                return r.json().get("status") == "ok"
            except Exception:
                return False
        return False
    except Exception:
        return False

def get_jsearch_quota() -> Dict[str, Any]:
    """Retrieves JSearch monthly usage, limits, and threshold warnings from backend."""
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            try:
                return r.json().get("jsearch_quota", {})
            except Exception:
                pass
    except Exception:
        pass
    return {}

# ── Auth APIs ──

def register_user(name: str, email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Registers a new HireAgent user."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json={"name": name, "email": email, "password": password},
            timeout=5
        )
        return _safe_json_response(r, "Registration failed")
    except requests.exceptions.Timeout:
        return None, "Registration request timed out. Please try again."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def login_user(email: str, password: str) -> Tuple[Optional[str], Optional[str]]:
    """Logs in user and returns JWT access token."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            json={"email": email, "password": password},
            timeout=5
        )
        data, err = _safe_json_response(r, "Invalid credentials")
        if data and isinstance(data, dict):
            return data.get("access_token"), None
        return None, err or "Invalid credentials"
    except requests.exceptions.Timeout:
        return None, "Login request timed out. Please try again."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def login_firebase_user(id_token: str) -> Tuple[Optional[str], Optional[str]]:
    """Authenticates user using Firebase Google ID token via POST /api/v1/auth/firebase."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/auth/firebase",
            json={"id_token": id_token},
            timeout=25
        )
        data, err = _safe_json_response(r, "Google Firebase authentication failed")
        if data and isinstance(data, dict):
            return data.get("access_token"), None
        return None, err or "Google Firebase authentication failed"
    except requests.exceptions.Timeout:
        return None, "Backend authentication timed out. Please try again."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_current_user(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves current user details via JWT."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/auth/me",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch user details")
    except requests.exceptions.Timeout:
        return None, "User details request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Resume APIs ──

def upload_resume(
    token: str,
    file_bytes: bytes,
    filename: str,
    target_role: Optional[str] = None,
    preferred_location: Optional[str] = None,
    preferred_company: Optional[str] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Uploads resume file to FastAPI backend with optional preferences."""
    try:
        files = {"file": (filename, file_bytes)}
        data = {}
        if target_role:
            data["target_role"] = target_role
        if preferred_location:
            data["preferred_location"] = preferred_location
        if preferred_company:
            data["preferred_company"] = preferred_company

        r = requests.post(
            f"{API_BASE_URL}/api/v1/resumes",
            headers=_get_headers(token),
            files=files,
            data=data if data else None,
            timeout=60
        )
        return _safe_json_response(r, "Resume upload failed")
    except requests.exceptions.Timeout:
        return None, "Job matching is taking longer than expected. Please try again."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_user_resumes(token: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Fetches user's saved resumes."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/resumes",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch resumes")
    except requests.exceptions.Timeout:
        return None, "Request timed out while fetching resumes."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_resume_detail(token: str, resume_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches specific resume details."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/resumes/{resume_id}",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch resume details")
    except requests.exceptions.Timeout:
        return None, "Request timed out while fetching resume details."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def delete_resume(token: str, resume_id: int) -> Tuple[bool, Optional[str]]:
    """Deletes specific resume."""
    try:
        r = requests.delete(
            f"{API_BASE_URL}/api/v1/resumes/{resume_id}",
            headers=_get_headers(token),
            timeout=5
        )
        data, err = _safe_json_response(r, "Failed to delete resume")
        if err:
            return False, err
        return True, None
    except requests.exceptions.Timeout:
        return False, "Delete request timed out."
    except Exception as e:
        return False, f"Connection error: {str(e)}"

# ── Profile APIs ──

def get_user_profile(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches user's career profile."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/profile",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch user profile")
    except requests.exceptions.Timeout:
        return None, "Profile request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def update_user_profile(token: str, profile_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Updates user's career profile."""
    try:
        r = requests.put(
            f"{API_BASE_URL}/api/v1/profile",
            headers=_get_headers(token),
            json=profile_data,
            timeout=5
        )
        return _safe_json_response(r, "Failed to update profile")
    except requests.exceptions.Timeout:
        return None, "Profile update request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Job Preferences APIs ──

def get_user_preferences(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches user's job search & auto-apply preferences."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/preferences",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch job preferences")
    except requests.exceptions.Timeout:
        return None, "Preferences request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def update_user_preferences(token: str, prefs_data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Updates user's job search & auto-apply preferences."""
    try:
        r = requests.put(
            f"{API_BASE_URL}/api/v1/preferences",
            headers=_get_headers(token),
            json=prefs_data,
            timeout=5
        )
        return _safe_json_response(r, "Failed to update job preferences")
    except requests.exceptions.Timeout:
        return None, "Preferences update request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Jobs APIs ──

def get_jobs(
    token: Optional[str] = None,
    search: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches job listings from FastAPI backend with optional search and role filtering."""
    try:
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if search:
            params["search"] = search
        if role:
            params["role"] = role
        r = requests.get(
            f"{API_BASE_URL}/api/v1/jobs",
            headers=_get_headers(token),
            params=params,
            timeout=5
        )
        return _safe_json_response(r, "Failed to fetch jobs")
    except requests.exceptions.Timeout:
        return None, "Job catalog request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Semantic & Vector Search APIs (Phase 8) ──

def index_resume(token: str, resume_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Indexes or re-indexes user's resume into Qdrant Cloud."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/semantic/resumes/{resume_id}/index",
            headers=_get_headers(token),
            timeout=30
        )
        return _safe_json_response(r, "Failed to index resume vectors")
    except requests.exceptions.Timeout:
        return None, "Resume vector indexing timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def search_resume_semantic(
    token: str,
    query: str,
    limit: int = 5,
    resume_id: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Performs semantic vector search across authenticated user's resume chunks."""
    try:
        payload: Dict[str, Any] = {"query": query, "limit": limit}
        if resume_id is not None:
            payload["resume_id"] = resume_id
        r = requests.post(
            f"{API_BASE_URL}/api/v1/semantic/resumes/search",
            headers=_get_headers(token),
            json=payload,
            timeout=15
        )
        return _safe_json_response(r, "Failed to search resume vectors")
    except requests.exceptions.Timeout:
        return None, "Resume search timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def index_all_jobs_semantic(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Indexes static job listings into Qdrant Cloud."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/v1/semantic/jobs/index-all",
            headers=_get_headers(token),
            timeout=60
        )
        return _safe_json_response(r, "Failed to index jobs into Qdrant")
    except requests.exceptions.Timeout:
        return None, "Job vector indexing timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def search_jobs_semantic(token: str, query: str, limit: int = 5) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Performs semantic vector search across indexed job listings."""
    try:
        payload = {"query": query, "limit": limit}
        r = requests.post(
            f"{API_BASE_URL}/api/v1/semantic/jobs/search",
            headers=_get_headers(token),
            json=payload,
            timeout=15
        )
        return _safe_json_response(r, "Failed to search jobs vector index")
    except requests.exceptions.Timeout:
        return None, "Job vector search timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_resume_matched_jobs_semantic(token: str, resume_id: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves job listings semantically matched against candidate resume vectors."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/semantic/resumes/{resume_id}/matched-jobs",
            headers=_get_headers(token),
            timeout=15
        )
        return _safe_json_response(r, "Failed to retrieve semantically matched jobs")
    except requests.exceptions.Timeout:
        return None, "Vector matching request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Hybrid Matching & Recommendations (LangGraph) ──

def get_job_recommendations(
    token: str,
    resume_id: Optional[int] = None,
    target_role: Optional[str] = None,
    top_k: int = 10
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves Top K explainable job recommendations using the LangGraph hybrid matching workflow."""
    try:
        payload: Dict[str, Any] = {"top_k": top_k}
        if resume_id is not None:
            payload["resume_id"] = resume_id
        if target_role:
            payload["target_role"] = target_role

        r = requests.post(
            f"{API_BASE_URL}/api/v1/matching/recommendations",
            headers=_get_headers(token),
            json=payload,
            timeout=30
        )
        return _safe_json_response(r, "Failed to retrieve job recommendations")
    except requests.exceptions.Timeout:
        return None, "Recommendation request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Phase 2 & 3: AI Assistant & Career Insights ──

def get_job_by_id(token: Optional[str], job_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Retrieves specific job listing details by id."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/jobs/{job_id}",
            headers=_get_headers(token),
            timeout=5
        )
        return _safe_json_response(r, f"Job {job_id} not found")
    except requests.exceptions.Timeout:
        return None, "Job lookup timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def chat_with_assistant(token: str, message: str, history: Optional[List[Dict[str, str]]] = None) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Sends user query and history to AI Career Assistant endpoint."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/assistant/chat",
            headers=_get_headers(token),
            json={"message": message, "history": history or []},
            timeout=30
        )
        return _safe_json_response(r, "Assistant failed to generate response")
    except requests.exceptions.Timeout:
        return None, "AI assistant request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def get_career_insights(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Triggers CrewAI Multi-Agent career insights report."""
    try:
        r = requests.post(
            f"{API_BASE_URL}/api/career/insights",
            headers=_get_headers(token),
            timeout=45
        )
        return _safe_json_response(r, "Failed to generate career insights")
    except requests.exceptions.Timeout:
        return None, "Career insights request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

# ── Dashboard Workspace API ──

def get_dashboard_overview(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches the aggregated read-only dashboard overview for the authenticated user."""
    try:
        r = requests.get(
            f"{API_BASE_URL}/api/v1/dashboard/overview",
            headers=_get_headers(token),
            timeout=15
        )
        return _safe_json_response(r, "Failed to fetch dashboard overview")
    except requests.exceptions.Timeout:
        return None, "Dashboard overview request timed out."
    except Exception as e:
        return None, f"Connection error: {str(e)}"

def rerun_dashboard_pipeline(
    token: str,
    target_role: Optional[str] = None,
    preferred_location: Optional[str] = None,
    preferred_company: Optional[str] = None,
    resume_id: Optional[int] = None
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Explicitly re-runs the end-to-end analysis pipeline for the current user."""
    try:
        payload: Dict[str, Any] = {}
        if target_role:
            payload["target_role"] = target_role
        if preferred_location:
            payload["preferred_location"] = preferred_location
        if preferred_company:
            payload["preferred_company"] = preferred_company
        if resume_id is not None:
            payload["resume_id"] = resume_id

        r = requests.post(
            f"{API_BASE_URL}/api/v1/dashboard/rerun",
            headers=_get_headers(token),
            json=payload if payload else None,
            timeout=60
        )
        return _safe_json_response(r, "Failed to re-run pipeline analysis")
    except requests.exceptions.Timeout:
        return None, "Job matching is taking longer than expected. Please try again."
    except Exception as e:
        return None, f"Connection error: {str(e)}"





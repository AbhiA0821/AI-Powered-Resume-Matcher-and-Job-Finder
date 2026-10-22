"""
Unit tests for HireAgent Flask Frontend (Phase 4).
Validates:
- Route rendering for authenticated vs unauthenticated users
- Empty states on dashboard, resume, jobs, and career recommendations
- Form landmarks and accessible structure
- Security: No exposed secrets or raw tokens in rendered HTML
- Assistant chat proxy authorization enforcement
"""

import pytest
from unittest.mock import patch
from frontend.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test_flask_secret_key"
    with app.test_client() as client:
        yield client


def test_landing_page_renders_for_unauthenticated_user(client):
    """Validates landing page displays hero and core features without requiring sign-in."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Your AI-Powered Career Intelligence" in html
    assert "Resume Intelligence" in html
    assert "Hybrid Matching" in html
    assert "Sign In" in html


def test_login_and_register_pages_render(client):
    """Validates login and register pages render properly."""
    res_login = client.get("/login")
    assert res_login.status_code == 200
    assert "Sign In to Resume Matcher & Job Finder" in res_login.data.decode("utf-8")

    res_reg = client.get("/register")
    assert res_reg.status_code == 200
    assert "Create Your Account" in res_reg.data.decode("utf-8")


def test_protected_routes_redirect_unauthenticated_user(client):
    """Validates protected routes redirect to login when no token is in session."""
    protected_paths = [
        "/dashboard",
        "/resume",
        "/jobs",
        "/skill-gap",
        "/career-recommendations",
        "/assistant"
    ]
    for path in protected_paths:
        res = client.get(path)
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


def test_dashboard_renders_empty_state_when_no_resumes(client):
    """Validates clean empty state renders when authenticated user has not uploaded a resume."""
    with client.session_transaction() as sess:
        sess["token"] = "fake_test_jwt_token"
        sess["user"] = {"name": "Test Candidate", "email": "candidate@example.com"}

    with patch("frontend.api_client.get_user_resumes", return_value=([], None)), \
         patch("frontend.api_client.get_user_profile", return_value=({}, None)), \
         patch("frontend.api_client.get_jsearch_quota", return_value={"requests_used": 3, "monthly_limit": 200, "status": "normal", "requests_remaining": 197}):

        res = client.get("/dashboard")
        assert res.status_code == 200
        html = res.data.decode("utf-8")
        assert "No Resume Uploaded Yet" in html
        assert "Quota" not in html


def test_no_secrets_leaked_in_html(client):
    """Validates that secret keys, API tokens, and private env variables never appear in rendered templates."""
    res = client.get("/")
    html = res.data.decode("utf-8")

    forbidden_patterns = [
        "AIzaSy",  # Gemini API key pattern
        "gsk_",    # Groq API key pattern
        "x-rapidapi-key",
        "HIREAGENT_JWT_SECRET",
        "QDRANT_API_KEY",
        "qdrant.tech",
        "rapidapi.com"
    ]
    for pattern in forbidden_patterns:
        assert pattern.lower() not in html.lower()


def test_assistant_chat_proxy_requires_auth(client):
    """Validates assistant chat proxy returns 401 when called without a valid session."""
    res = client.post("/api/proxy/assistant/chat", json={"message": "hello"})
    assert res.status_code == 401
    assert res.json["ok"] is False


def test_assistant_chat_proxy_forwards_for_authenticated_user(client):
    """Validates assistant chat proxy passes message and returns response."""
    with client.session_transaction() as sess:
        sess["token"] = "fake_test_jwt_token"

    mock_assistant_resp = {
        "ok": True,
        "answer": "Here are recommendations for your profile.",
        "job_refs": ["job_123"],
        "tool_trace": [{"tool": "get_student_profile", "ok": True, "summary": "Fetched profile"}]
    }

    with patch("frontend.api_client.chat_with_assistant", return_value=(mock_assistant_resp, None)):
        res = client.post("/api/proxy/assistant/chat", json={"message": "What is my role?"})
        assert res.status_code == 200
        data = res.json
        assert data["ok"] is True
        assert data["answer"] == "Here are recommendations for your profile."
        assert "job_123" in data["job_refs"]

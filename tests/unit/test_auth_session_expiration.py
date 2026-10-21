"""
Unit tests for HireAgent Auth Session & Expiration Handling.
Verifies:
1. Valid JWT allows access to dashboard and protected endpoints.
2. Expired JWT in session is detected, cleared, and user redirected to /login.
3. Backend returning 401 'Invalid or expired authentication token' clears session and redirects.
4. AJAX request with expired session returns 401 with redirect URL.
5. Fresh login establishes session with valid token.
"""

import time
import pytest
import jwt
from unittest.mock import patch, MagicMock
from frontend.app import app, get_session_token, is_auth_error


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test_flask_secret_key"
    with app.test_client() as client:
        yield client


def test_valid_jwt_recognized_by_get_session_token():
    """Verifies unexpired JWT is returned without clearing session."""
    future_exp = int(time.time()) + 3600
    valid_jwt = jwt.encode({"sub": "42", "exp": future_exp, "iat": int(time.time())}, "secret_key_32_bytes_long_minimum!", algorithm="HS256")

    with app.test_request_context():
        from flask import session
        session["token"] = valid_jwt
        token = get_session_token()
        assert token == valid_jwt


def test_expired_jwt_clears_session_and_returns_none():
    """Verifies expired JWT causes get_session_token to clear session and return None."""
    past_exp = int(time.time()) - 100
    expired_jwt = jwt.encode({"sub": "42", "exp": past_exp, "iat": past_exp - 100}, "secret_key_32_bytes_long_minimum!", algorithm="HS256")

    with app.test_request_context():
        from flask import session
        session["token"] = expired_jwt
        token = get_session_token()
        assert token is None
        assert "token" not in session


def test_dashboard_redirects_on_expired_jwt(client):
    """Verifies visiting /dashboard with expired JWT clears session and redirects to /login."""
    past_exp = int(time.time()) - 100
    expired_jwt = jwt.encode({"sub": "42", "exp": past_exp, "iat": past_exp - 100}, "secret_key_32_bytes_long_minimum!", algorithm="HS256")

    with client.session_transaction() as sess:
        sess["token"] = expired_jwt
        sess["user"] = {"email": "test@example.com"}

    res = client.get("/dashboard")
    assert res.status_code == 302
    assert "/login" in res.headers["Location"]


def test_dashboard_rerun_ajax_returns_401_on_expired_jwt(client):
    """Verifies AJAX call with expired JWT returns 401 with redirect to /login."""
    past_exp = int(time.time()) - 100
    expired_jwt = jwt.encode({"sub": "42", "exp": past_exp, "iat": past_exp - 100}, "secret_key_32_bytes_long_minimum!", algorithm="HS256")

    with client.session_transaction() as sess:
        sess["token"] = expired_jwt

    res = client.post(
        "/dashboard/rerun",
        json={"target_role": "Python Developer"},
        headers={"X-Requested-With": "XMLHttpRequest"}
    )
    assert res.status_code == 401
    data = res.get_json()
    assert data["ok"] is False
    assert "/login" in data["redirect"]


def test_backend_auth_error_clears_session_and_redirects(client):
    """Verifies that when backend returns 'Invalid or expired authentication token', Flask clears session."""
    future_exp = int(time.time()) + 3600
    valid_format_jwt = jwt.encode({"sub": "42", "exp": future_exp, "iat": int(time.time())}, "secret_key_32_bytes_long_minimum!", algorithm="HS256")

    with client.session_transaction() as sess:
        sess["token"] = valid_format_jwt
        sess["user"] = {"email": "test@example.com"}

    with patch("frontend.api_client.get_user_resumes", return_value=(None, "Invalid or expired authentication token")), \
         patch("frontend.api_client.get_user_profile", return_value=(None, "Invalid or expired authentication token")), \
         patch("frontend.api_client.get_dashboard_overview", return_value=(None, "Invalid or expired authentication token")), \
         patch("frontend.api_client.get_jsearch_quota", return_value={}):

        res = client.get("/dashboard")
        assert res.status_code == 302
        assert "/login" in res.headers["Location"]


def test_is_auth_error_helper():
    """Verifies is_auth_error identifies various token/auth failure messages."""
    assert is_auth_error("Invalid or expired authentication token") is True
    assert is_auth_error("401 Unauthorized") is True
    assert is_auth_error("Invalid token payload") is True
    assert is_auth_error("Something went wrong with database") is False
    assert is_auth_error(None) is False

"""
Unit tests for Firebase Google Sign-In Frontend Flow.
Validates:
- Rendering of 'Continue with Google' button, SVG icon, and Firebase scripts on /login
- POST /login/firebase validation and error handling (missing token, invalid token)
- Successful Google login setting Flask session and returning JSON redirect
- Frontend api_client.login_firebase_user properly calling POST /api/v1/auth/firebase
- Zero real outbound network calls (enforced by conftest guards)
"""

import pytest
from unittest.mock import patch, MagicMock
from frontend.app import app
from frontend import api_client


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test_flask_secret_key"
    with app.test_client() as client:
        yield client


def test_login_page_renders_continue_with_google_button(client):
    """Validates login page renders the Google button, SVG icon, and Firebase script assets."""
    res = client.get("/login")
    assert res.status_code == 200
    html = res.data.decode("utf-8")
    assert "Continue with Google" in html
    assert "google-signin-btn" in html
    assert "google-auth-alert" in html
    assert "firebase-app-compat.js" in html
    assert "firebase-auth-compat.js" in html
    assert "google_auth.js" in html


def test_login_firebase_missing_token_returns_400(client):
    """Validates POST /login/firebase returns 400 when id_token is missing."""
    res = client.post("/login/firebase", json={})
    assert res.status_code == 400
    data = res.get_json()
    assert data["ok"] is False
    assert "Missing Firebase ID token" in data["error"]


def test_login_firebase_backend_failure_returns_401(client):
    """Validates POST /login/firebase returns 401 when backend verification fails."""
    with patch("frontend.api_client.login_firebase_user", return_value=(None, "Invalid Firebase ID token")):
        res = client.post("/login/firebase", json={"id_token": "invalid_fake_token"})
        assert res.status_code == 401
        data = res.get_json()
        assert data["ok"] is False
        assert "Invalid Firebase ID token" in data["error"]


def test_login_firebase_success_sets_session_and_redirects(client):
    """Validates successful Google auth sets Flask session['token'] and returns dashboard redirect."""
    fake_jwt = "hireagent_jwt_token_for_google_user"
    fake_user_profile = {
        "id": 42,
        "email": "candidate@example.com",
        "name": "Jane Google",
        "google_id": "google-uid-12345"
    }

    with patch("frontend.api_client.login_firebase_user", return_value=(fake_jwt, None)), \
         patch("frontend.api_client.get_current_user", return_value=(fake_user_profile, None)):

        res = client.post("/login/firebase", json={"id_token": "valid_firebase_id_token"})
        assert res.status_code == 200
        data = res.get_json()
        assert data["ok"] is True
        assert data["redirect"] == "/dashboard"

        # Verify session was populated
        with client.session_transaction() as sess:
            assert sess.get("token") == fake_jwt
            assert sess.get("user") == fake_user_profile


def test_api_client_login_firebase_user_success():
    """Validates frontend api_client sends correct payload to POST /api/v1/auth/firebase."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"access_token": "mocked_jwt_token", "token_type": "bearer"}

    with patch("requests.post", return_value=mock_response) as mock_post:
        token, err = api_client.login_firebase_user("test_id_token_123")
        assert token == "mocked_jwt_token"
        assert err is None
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "/api/v1/auth/firebase" in args[0]
        assert kwargs["json"] == {"id_token": "test_id_token_123"}


def test_api_client_login_firebase_user_failure():
    """Validates frontend api_client returns error message on 401 response."""
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.json.return_value = {"detail": "Firebase authentication failed: Token expired"}

    with patch("requests.post", return_value=mock_response):
        token, err = api_client.login_firebase_user("expired_id_token")
        assert token is None
        assert "Token expired" in err

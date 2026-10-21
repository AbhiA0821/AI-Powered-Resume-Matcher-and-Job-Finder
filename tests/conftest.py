import os
import sys
import tempfile
import pytest
from unittest.mock import patch, MagicMock

# Ensure backend directory is on sys.path so tests can do `from app...`
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Ensure root directory is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Disallow live tests by default
os.environ.setdefault("HIREAGENT_LIVE_TESTS", "0")


@pytest.fixture(autouse=True)
def protect_jsearch_usage_tracker(tmp_path, monkeypatch):
    """
    Autouse safety net ensuring NO test ever modifies the production
    data/jsearch_usage.json or data/resume_matcher.db usage files.
    """
    from app.services.job_sources.usage_tracker import JSearchUsageTracker
    import app.services.job_sources.usage_tracker as tracker_mod

    temp_db = str(tmp_path / "test_safety_usage.db")
    temp_json = str(tmp_path / "test_safety_usage.json")
    safe_tracker = JSearchUsageTracker(db_path=temp_db, snapshot_path=temp_json)

    # Route default tracker to test-isolated safe_tracker
    monkeypatch.setattr(tracker_mod, "_default_tracker", safe_tracker)
    monkeypatch.setattr(tracker_mod, "DEFAULT_DB_PATH", temp_db)
    monkeypatch.setattr(tracker_mod, "DEFAULT_SNAPSHOT_PATH", temp_json)


@pytest.fixture(autouse=True)
def network_and_api_safety_guard(monkeypatch):
    """
    Global autouse guard that strictly blocks any real outbound HTTP
    requests to RapidAPI/JSearch and external LLM/vector providers.
    """
    import requests

    original_send = requests.Session.send

    def guarded_send(session_self, request, **kwargs):
        url = getattr(request, "url", "") or ""
        url_lower = url.lower()

        # Hard stop on RapidAPI / JSearch
        if "rapidapi.com" in url_lower or "jsearch" in url_lower:
            pytest.fail(
                f"STRICT SECURITY VIOLATION: Test attempted real outbound HTTP call to RapidAPI/JSearch: {url}"
            )

        # External AI / Vector DB block unless explicitly opted-in
        live_tests = os.environ.get("HIREAGENT_LIVE_TESTS", "0") == "1"
        if not live_tests:
            if "generativelanguage.googleapis.com" in url_lower:
                pytest.fail(
                    f"STRICT SECURITY VIOLATION: Test attempted real outbound call to Gemini: {url}"
                )
            if "groq.com" in url_lower:
                pytest.fail(
                    f"STRICT SECURITY VIOLATION: Test attempted real outbound call to Groq: {url}"
                )
            if "cloud.qdrant.io" in url_lower or ":6333" in url_lower:
                pytest.fail(
                    f"STRICT SECURITY VIOLATION: Test attempted real outbound call to Qdrant: {url}"
                )

        return original_send(session_self, request, **kwargs)

    monkeypatch.setattr(requests.Session, "send", guarded_send)

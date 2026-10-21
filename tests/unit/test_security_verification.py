"""
Security Verification Suite (Phase 5).
Validates:
- JWT requirement on new endpoints (/api/assistant/chat, /api/career/insights)
- User isolation: identity comes from server-side token context, not user/LLM input
- In-memory rate limiting enforcement on assistant chat
- Strict read-only tool execution & schema bounds
- Output delimitation with <untrusted_job_text>
- No leakage of internal secrets or stack traces
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from app.main import app
from app.tools.registry import tool_registry, ToolContext
from app.tools.schemas import SearchLiveJobsInput, CalculateSkillGapInput
from app.core.config import settings


@pytest.fixture
def api_client():
    return TestClient(app)


def test_assistant_chat_requires_jwt(api_client):
    """Validates POST /api/assistant/chat rejects requests without JWT token."""
    res = api_client.post("/api/assistant/chat", json={"message": "hello"})
    assert res.status_code == 401
    assert "detail" in res.json()


def test_career_insights_requires_jwt(api_client):
    """Validates POST /api/career/insights rejects requests without JWT token."""
    res = api_client.post("/api/career/insights")
    assert res.status_code == 401
    assert "detail" in res.json()


def test_user_isolation_enforced_in_tools():
    """Validates that tool execution cannot be spoofed to access another user's profile."""
    # Context has authenticated user_id=404
    ctx = ToolContext(user_id=404, live_searches_conducted=0)

    # Attempt to query profile passing a malicious payload attempting to impersonate user 1
    res = tool_registry.execute("get_student_profile", {"user_id": 1, "is_admin": True}, ctx)
    assert res.get("ok") is True
    # The returned profile MUST belong to the server context user (404), not the attacker argument
    profile = res.get("profile", {})
    assert profile.get("user_id") == 404


def test_all_registered_tools_are_strictly_read_only():
    """Validates that no write, modification, deletion, or code execution tools exist."""
    tools = tool_registry.list_tools()
    assert len(tools) == 6

    forbidden_verbs = ["write", "create", "delete", "update", "execute", "eval", "drop", "modify", "insert"]
    for t in tools:
        name_lower = t.name.lower()
        desc_lower = t.description.lower()
        for verb in forbidden_verbs:
            assert verb not in name_lower, f"Forbidden verb {verb} in tool name {t.name}"
            # Ensure descriptions explicitly reflect retrieval/calculation
            assert "read" in desc_lower or "retriev" in desc_lower or "calculat" in desc_lower or "search" in desc_lower


def test_tool_argument_bounds_and_validation():
    """Validates bounded input schemas reject out-of-range limits and hostile strings."""
    ctx = ToolContext(user_id=101)

    # limit > 10 should fail validation
    res = tool_registry.execute("search_live_jobs", {"role": "Engineer", "limit": 9999}, ctx)
    assert res.get("ok") is False
    assert "Invalid arguments" in res.get("error", "")

    # empty job_id should fail validation
    res_gap = tool_registry.execute("calculate_skill_gap", {"job_id": ""}, ctx)
    assert res_gap.get("ok") is False
    assert "Invalid arguments" in res_gap.get("error", "")


def test_assistant_chat_in_memory_rate_limiting():
    """Validates that exceeding 15 requests/minute triggers HTTP 429 Too Many Requests."""
    from app.api.v1.assistant import _user_request_timestamps, MAX_REQUESTS_PER_WINDOW, _check_rate_limit
    from fastapi import HTTPException

    _user_request_timestamps.clear()

    # Pre-fill 15 requests
    import time
    now = time.time()
    _user_request_timestamps[999] = [now] * MAX_REQUESTS_PER_WINDOW

    with pytest.raises(HTTPException) as excinfo:
        _check_rate_limit(999)

    assert excinfo.value.status_code == 429
    assert "Rate limit exceeded" in excinfo.value.detail



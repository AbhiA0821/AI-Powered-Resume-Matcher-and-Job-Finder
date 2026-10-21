"""
Unit tests for CrewAI Multi-Agent System (Phase 3).
Validates:
- Construction of 5 sequential agents (no Interview Agent, allow_delegation=False)
- Server-side tool binding
- Structured output validation (CareerInsightsReport)
- Deterministic fallback when AI agents are unavailable or fail
- Absence of OPENAI_API_KEY requirement (memory=False)
- In-memory insights caching
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from app.tools.schemas import ToolContext
from app.services.crew_service import (
    build_career_insights_crew,
    build_deterministic_insights_fallback,
    generate_career_insights,
    CareerInsightsReport,
    _INSIGHTS_CACHE
)


@pytest.fixture
def crew_context():
    return ToolContext(user_id=202, live_searches_conducted=0)


def test_crew_has_exactly_five_agents_and_no_interview_agent(crew_context):
    """Validates that exactly 5 agents exist, without any Interview Agent, and delegation is disabled."""
    mock_llm = "gemini/gemini-2.5-flash"
    crew = build_career_insights_crew(crew_context, mock_llm)

    assert len(crew.agents) == 5
    agent_roles = [a.role for a in crew.agents]

    # Must NOT have Interview Agent
    for r in agent_roles:
        assert "interview" not in r.lower()

    # All agents must have allow_delegation=False and bounded iterations
    for a in crew.agents:
        assert a.allow_delegation is False
        assert a.max_iter <= 2

    # Memory must be strictly False (no OpenAI embedder)
    assert crew.memory is False


def test_deterministic_insights_fallback(crew_context):
    """Validates that the fallback builds a complete CareerInsightsReport with is_degraded=True."""
    mock_jobs = [
        {
            "id": "j_test_101",
            "job_title": "AI Developer",
            "company": "Alpha Corp",
            "location": "Bengaluru, India",
            "application_url": "https://example.com/apply/101"
        }
    ]

    with patch("app.tools.registry.handle_get_student_profile", return_value={
        "ok": True,
        "profile": {"primary_role": "AI Engineer", "skills": ["Python", "PyTorch"]}
    }), patch("app.tools.registry.handle_search_live_jobs", return_value={
        "ok": True,
        "jobs": mock_jobs
    }), patch("app.tools.registry.handle_calculate_skill_gap", return_value={
        "ok": True,
        "matched_skills": ["Python"],
        "missing_skills": ["Docker", "Kubernetes"],
        "coverage_percentage": 50.0
    }):

        report = build_deterministic_insights_fallback(crew_context)

        assert isinstance(report, CareerInsightsReport)
        assert report.is_degraded is True
        assert report.target_role == "AI Engineer"
        assert len(report.top_jobs) == 1
        assert report.top_jobs[0].job_id == "j_test_101"
        assert report.top_jobs[0].company == "Alpha Corp"
        assert "Docker" in report.top_jobs[0].missing_skills
        assert len(report.learning_priorities) > 0


def test_generate_career_insights_uses_fallback_when_keys_missing(crew_context):
    """Validates graceful degradation to deterministic report when API keys are absent."""
    # Ensure cache is clear for this user
    _INSIGHTS_CACHE.pop(f"user_{crew_context.user_id}", None)

    mock_jobs = [{"job_id": "j_mock_1", "title": "Developer", "company": "Acme", "apply_url": "https://example.com"}]

    with patch("app.services.crew_service._get_crew_llm", return_value=None), \
         patch("app.tools.registry.handle_get_student_profile", return_value={"ok": True, "profile": {"primary_role": "Software Engineer", "skills": ["Python"]}}), \
         patch("app.tools.registry.handle_search_live_jobs", return_value={"ok": True, "jobs": mock_jobs}):
        report = generate_career_insights(crew_context)

        assert isinstance(report, CareerInsightsReport)
        assert report.is_degraded is True
        assert len(report.career_directions) > 0


def test_generate_career_insights_cache_hit(crew_context):
    """Validates subsequent calls within TTL return cached CareerInsightsReport."""
    _INSIGHTS_CACHE.pop(f"user_{crew_context.user_id}", None)

    mock_jobs = [{"job_id": "j_mock_1", "title": "Developer", "company": "Acme", "apply_url": "https://example.com"}]

    with patch("app.services.crew_service._get_crew_llm", return_value=None), \
         patch("app.tools.registry.handle_get_student_profile", return_value={"ok": True, "profile": {"primary_role": "Software Engineer", "skills": ["Python"]}}), \
         patch("app.tools.registry.handle_search_live_jobs", return_value={"ok": True, "jobs": mock_jobs}):
        # First call generates report and caches
        report1 = generate_career_insights(crew_context)
        # Second call returns cached version
        report2 = generate_career_insights(crew_context)

        assert report2.cached is True
        assert report1.target_role == report2.target_role

import pytest
from unittest.mock import patch
from app.services.llm_service import build_rag_prompt, generate_grounded_recommendation
from app.workflows.hireagent_graph import run_hireagent_matching_workflow
from app.services.job_sources.rapidapi_job_service import NormalizedJob

def test_rag_prompt_construction():
    """Verifies that RAG prompt contains strict anti-hallucination instructions and injected data."""
    prompt = build_rag_prompt(
        resume_skills=["Python", "FastAPI"],
        target_role="Backend Developer",
        job_title="Senior Backend Developer",
        company="Stripe",
        required_skills=["Python", "FastAPI", "Docker", "Redis"],
        matched_skills=["Python", "FastAPI"],
        missing_skills=["Docker", "Redis"],
        match_percentage=75.0
    )
    assert "STRICT INSTRUCTIONS" in prompt
    assert "Do NOT invent, assume, or hallucinate" in prompt
    assert "Candidate Target Role: Backend Developer" in prompt
    assert "Company: Stripe" in prompt
    assert "Match Score: 75.0%" in prompt

def test_grounded_recommendation_fallback():
    """Verifies that generate_grounded_recommendation falls back safely to deterministic recommendation."""
    fallback = "Strong alignment across Python and FastAPI. Bridge Docker and Redis gaps."
    rec = generate_grounded_recommendation(
        resume_skills=["Python"],
        target_role="Developer",
        job_title="Dev",
        company="Co",
        required_skills=["Python"],
        matched_skills=["Python"],
        missing_skills=[],
        match_percentage=90.0,
        fallback_recommendation=fallback
    )
    assert isinstance(rec, str)
    assert len(rec) > 0

def test_langgraph_workflow_orchestration():
    """Verifies full execution of the controlled LangGraph matching workflow with mocked live search."""
    resume_text = (
        "Alice Smith - Senior Python Developer.\n"
        "Expert in building scalable APIs with FastAPI, PostgreSQL, Redis, and Docker.\n"
        "Proficient in Git, CI/CD, and microservices architecture."
    )
    mock_job = NormalizedJob(
        external_job_id="mock-1",
        title="Python Backend Developer",
        company="Acme Cloud",
        location="Bengaluru, India",
        description="FastAPI microservices development.",
        skills=["Python", "FastAPI", "Docker"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/view/mock1",
        source_type="live_api"
    )

    with patch("app.services.job_sources.rapidapi_job_service.search_live_jobs", return_value=[mock_job]), \
         patch("app.services.job_service.search_live_jobs", return_value=[mock_job]):
        result = run_hireagent_matching_workflow(
            user_id=42,
            resume_text=resume_text,
            target_role="Backend Developer",
            top_k=5
        )

    assert result["user_id"] == 42
    assert result["target_role"] == "Backend Developer"
    assert "python" in result["extracted_skills"]
    assert "fastapi" in result["extracted_skills"]
    assert len(result["recommendations"]) <= 5
    assert len(result["recommendations"]) > 0

    top_job = result["recommendations"][0]
    assert "job_id" in top_job
    assert "job_title" in top_job
    assert "company" in top_job
    assert "match_percentage" in top_job
    assert "matched_skills" in top_job
    assert "missing_skills" in top_job
    assert "application_url" in top_job
    assert top_job["match_percentage"] > 0

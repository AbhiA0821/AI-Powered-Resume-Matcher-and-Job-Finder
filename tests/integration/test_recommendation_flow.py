import pytest
from unittest.mock import patch
from app.workflows.hireagent_graph import build_hireagent_graph
from app.services.matching_service import match_and_rank_jobs
from app.schemas.matching import JobRecommendation
from app.services.job_sources.rapidapi_job_service import NormalizedJob

def test_full_graph_recommendation_flow():
    """Validates full LangGraph execution from resume text to validated recommendations."""
    graph = build_hireagent_graph()

    sample_resume = """
    Jane Doe
    AI Engineer & Python Developer
    Bengaluru, India
    
    Skills: Python, PyTorch, Machine Learning, FastAPI, SQL, Docker
    
    Experience:
    Junior AI Developer at TechLabs (2024-2026)
    - Built computer vision and NLP models using PyTorch and Transformers
    - Developed REST APIs with FastAPI
    """

    initial_state = {
        "user_id": 999,
        "resume_id": 10,
        "resume_text": sample_resume,
        "extracted_skills": [],
        "target_role": "",
        "candidate_jobs": [],
        "ranked_jobs": [],
        "top_k": 5,
        "errors": []
    }

    mock_job = NormalizedJob(
        external_job_id="test-rec-flow-1",
        title="AI Engineer",
        company="TechCorp Labs",
        location="Bengaluru, India",
        description="Developing Python and PyTorch ML models.",
        skills=["Python", "PyTorch", "FastAPI"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/view/test-rec-1",
        source_type="live_api"
    )

    with patch("app.services.job_sources.rapidapi_job_service.search_live_jobs", return_value=[mock_job]), \
         patch("app.services.job_service.search_live_jobs", return_value=[mock_job]):
        final_state = graph.invoke(initial_state)

    assert "ranked_jobs" in final_state
    ranked = final_state["ranked_jobs"]
    assert len(ranked) > 0
    assert len(ranked) <= 5

    first_job = ranked[0]
    # Check that required schema fields are present
    assert "job_title" in first_job
    assert "company" in first_job
    assert "match_percentage" in first_job
    assert "matched_skills" in first_job
    assert "missing_skills" in first_job
    assert "recommendation" in first_job
    assert "source" in first_job
    assert "source_type" in first_job
    assert "location" in first_job

    # Validate with Pydantic model
    validated = JobRecommendation(**first_job)
    assert validated.match_percentage >= 15.0

def test_url_safety_in_matching():
    """Validates that match_and_rank_jobs preserves original URLs and does not invent URLs."""
    mock_jobs = [
        {
            "id": 1,
            "job_title": "AI Engineer",
            "company": "Real Corp",
            "required_skills": "Python, Machine Learning",
            "job_role": "AI Engineer",
            "description": "Building AI systems.",
            "application_url": "https://company.com/apply/1",
            "source": "LinkedIn",
            "source_type": "live_api",
            "location": "Bengaluru, India"
        },
        {
            "id": 2,
            "job_title": "Data Scientist",
            "company": "Another Corp",
            "required_skills": "Python, SQL",
            "job_role": "Data Scientist",
            "description": "Analyzing data.",
            "application_url": None,
            "source": "HireAgent Test Catalog",
            "source_type": "static_catalog",
            "location": "India"
        }
    ]

    resume = "Python, Machine Learning, SQL developer"
    ranked = match_and_rank_jobs(resume_text=resume, jobs=mock_jobs, top_k=2)

    assert len(ranked) == 2
    job_with_url = next(j for j in ranked if j["job_id"] == 1)
    job_without_url = next(j for j in ranked if j["job_id"] == 2)

    assert job_with_url["application_url"] == "https://company.com/apply/1"
    assert job_with_url["source_type"] == "live_api"

    assert job_without_url["application_url"] == "Application link unavailable"
    assert job_without_url["source_type"] == "static_catalog"

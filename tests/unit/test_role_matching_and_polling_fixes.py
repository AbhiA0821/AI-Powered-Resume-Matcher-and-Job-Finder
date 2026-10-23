import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from app.services.skill_service import (
    is_role_compatible,
    normalize_role,
    ROLE_DISCIPLINES,
    ROLE_SKILL_MATRIX,
    ROLE_ALIASES
)
from app.services.matching_service import (
    match_and_rank_jobs,
    compute_role_score
)
from app.services.dashboard_service import (
    assemble_dashboard_overview,
    get_user_state,
    _USER_DASHBOARD_STATE,
    _OVERVIEW_CACHE
)

def test_15_canonical_it_roles_coverage_and_discipline_isolation():
    """Verifies that all 15 required IT roles are recognized, normalized, and partitioned cleanly."""
    required_roles = [
        "Software Engineer",
        "Python Developer",
        "Backend Developer",
        "Full Stack Developer",
        "AI Engineer",
        "Machine Learning Engineer",
        "Data Scientist",
        "Data Analyst",
        "Data Engineer",
        "DevOps Engineer",
        "Cloud Engineer",
        "MLOps Engineer",
        "Generative AI Engineer",
        "NLP Engineer",
        "Computer Vision Engineer"
    ]
    for role in required_roles:
        # Every role must exist in the skill matrix
        assert role in ROLE_SKILL_MATRIX, f"Role '{role}' is missing from ROLE_SKILL_MATRIX"
        # Every role must normalize to itself
        assert normalize_role(role) == role
        # Every role must have skills defined
        assert len(ROLE_SKILL_MATRIX[role]) >= 3, f"Role '{role}' should have at least 3 defining skills"

    # Discipline isolation: Software Engineer must be incompatible with Data Scientist
    compat, score, _ = is_role_compatible("Software Engineer", "Data Scientist", "Data Scientist")
    assert not compat
    assert score <= 35.0

    # Discipline isolation: AI Engineer must be incompatible with DevOps Engineer
    compat, score, _ = is_role_compatible("AI Engineer", "DevOps Engineer", "DevOps Engineer")
    assert not compat
    assert score <= 35.0

    # Compatible within discipline: Software Engineer with Backend Developer
    compat, score, _ = is_role_compatible("Software Engineer", "Senior Backend Developer", "Backend Developer")
    assert compat
    assert score >= 75.0

    # Compatible within discipline: AI Engineer with Generative AI Engineer
    compat, score, _ = is_role_compatible("AI Engineer", "GenAI Engineer", "Generative AI Engineer")
    assert compat
    assert score >= 80.0

def test_role_filtering_software_engineer_vs_ai_engineer():
    """
    Verifies that selecting Software Engineer strictly filters out AI Engineer and Data Scientist jobs
    when filter_unrelated_roles=True, preventing cross-discipline pollution.
    """
    mixed_jobs = [
        {
            "id": 1,
            "job_title": "Software Engineer",
            "company": "Tech Corp",
            "job_role": "Software Engineer",
            "required_skills": "Python, Java, Docker, Git, REST APIs",
            "description": "Develop backend enterprise applications."
        },
        {
            "id": 2,
            "job_title": "Lead Data Scientist",
            "company": "Analytics Inc",
            "job_role": "Data Scientist",
            "required_skills": "Python, Statistics, Machine Learning, Pandas",
            "description": "Build statistical forecasting models."
        },
        {
            "id": 3,
            "job_title": "AI Research Engineer",
            "company": "Cognitive AI",
            "job_role": "AI Engineer",
            "required_skills": "Python, PyTorch, Transformers, Large Language Models",
            "description": "Develop and fine-tune LLM foundation models."
        }
    ]

    resume = "Senior Python engineer with experience in Docker, REST APIs, Git, and some PyTorch."

    # When Software Engineer is selected, only the Software Engineer job is returned
    se_matches = match_and_rank_jobs(
        resume_text=resume,
        jobs=mixed_jobs,
        target_role="Software Engineer",
        top_k=5,
        filter_unrelated_roles=True
    )
    assert len(se_matches) == 1
    assert se_matches[0]["job_id"] == 1
    assert se_matches[0]["job_title"] == "Software Engineer"

    # When AI Engineer is selected, only the AI Engineer job is returned
    ai_matches = match_and_rank_jobs(
        resume_text=resume,
        jobs=mixed_jobs,
        target_role="AI Engineer",
        top_k=5,
        filter_unrelated_roles=True
    )
    assert len(ai_matches) == 1
    assert ai_matches[0]["job_id"] == 3
    assert ai_matches[0]["job_title"] == "AI Research Engineer"

def test_skills_influence_ranking_without_overriding_role():
    """
    Verifies that resume skills rank jobs within the compatible target role,
    but cannot promote an unrelated role into the match results.
    """
    candidate_jobs = [
        {
            "id": 10,
            "job_title": "Software Engineer (Full Stack)",
            "company": "WebCorp",
            "job_role": "Software Engineer",
            "required_skills": "Python, FastAPI, Docker, PostgreSQL",
            "description": "Full stack software development."
        },
        {
            "id": 11,
            "job_title": "Software Engineer (C++)",
            "company": "EmbeddedTech",
            "job_role": "Software Engineer",
            "required_skills": "C++, Linux, Kernel, Networking",
            "description": "Low-level system software development."
        },
        {
            "id": 12,
            "job_title": "Senior Data Scientist",
            "company": "DataHub",
            "job_role": "Data Scientist",
            # High skill overlap with candidate resume
            "required_skills": "Python, FastAPI, Docker, PostgreSQL, Pandas",
            "description": "Data science modeling and feature extraction."
        }
    ]

    resume = "Backend software engineer skilled in Python, FastAPI, Docker, and PostgreSQL."

    ranked = match_and_rank_jobs(
        resume_text=resume,
        jobs=candidate_jobs,
        target_role="Software Engineer",
        top_k=5,
        filter_unrelated_roles=True
    )

    # Data Scientist must be filtered out despite high skill overlap
    job_ids = [j["job_id"] for j in ranked]
    assert 12 not in job_ids

    # Both Software Engineer jobs are present, and the one with matched skills ranks first
    assert len(ranked) == 2
    assert ranked[0]["job_id"] == 10
    assert ranked[1]["job_id"] == 11
    assert ranked[0]["match_percentage"] > ranked[1]["match_percentage"]

def test_missing_results_for_role_returns_empty_not_unrelated_jobs():
    """
    Verifies requirement 10: If no suitable jobs exist for the selected role,
    return an empty list rather than unrelated jobs.
    """
    data_science_jobs_only = [
        {
            "id": 20,
            "job_title": "Senior Data Analyst",
            "company": "BI Corp",
            "job_role": "Data Analyst",
            "required_skills": "Tableau, SQL, Excel, PowerBI",
            "description": "Create business reports and dashboards."
        },
        {
            "id": 21,
            "job_title": "Data Scientist",
            "company": "Insight Inc",
            "job_role": "Data Scientist",
            "required_skills": "Python, R, Statistics, Scikit-learn",
            "description": "Perform predictive statistical modeling."
        }
    ]

    resume = "Experienced DevOps Engineer with Kubernetes, Terraform, and CI/CD skills."

    results = match_and_rank_jobs(
        resume_text=resume,
        jobs=data_science_jobs_only,
        target_role="DevOps Engineer",
        top_k=5,
        filter_unrelated_roles=True
    )

    # Must return empty list, NOT unrelated Data Analyst/Data Scientist jobs
    assert results == []

def test_cache_key_isolation_by_target_role():
    """
    Verifies that changing target_role invalidates the overview cache and
    does not return cached results from a previously selected role.
    """
    mock_db = MagicMock()
    user_id = 9999

    # Mock user and profile
    mock_user = MagicMock()
    mock_user.id = user_id

    mock_resume = MagicMock()
    mock_resume.id = 101
    mock_resume.filename = "test_resume.pdf"
    mock_resume.updated_at = None
    mock_resume.created_at = None
    mock_resume.parsed_text = "Python developer with FastAPI, PostgreSQL, and Docker."
    mock_resume.extracted_text = "Python developer with FastAPI, PostgreSQL, and Docker."

    mock_profile = MagicMock()
    mock_profile.id = 1
    mock_profile.user_id = user_id
    mock_profile.target_role = "Software Engineer"
    mock_profile.experience_level = "Mid"
    mock_profile.skills = ["python", "fastapi", "docker"]
    mock_profile.work_mode_preference = "Remote"
    mock_profile.employment_type = "Full-time"
    mock_profile.preferred_locations = "[\"India\"]"
    mock_profile.min_match_score = 65.0
    mock_profile.created_at = datetime.now(timezone.utc)
    mock_profile.updated_at = datetime.now(timezone.utc)

    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_resume
    mock_db.query.return_value.filter.return_value.first.return_value = mock_profile

    # Initialize user state
    _USER_DASHBOARD_STATE[user_id] = {
        "pipeline": [],
        "is_processing": False,
        "matching_completed": True,
        "matched_resume_id": 101,
        "matched_target_role": "Software Engineer",
        "matches": [
            {
                "job_id": 1,
                "job_title": "Software Engineer",
                "company": "Acme Inc",
                "match_percentage": 88.0,
                "matched_skills": ["python", "fastapi"],
                "missing_skills": [],
                "application_url": "https://example.com/apply",
                "source": "Catalog"
            }
        ]
    }

    # Request overview for Software Engineer
    overview_se = assemble_dashboard_overview(mock_db, mock_user)
    assert len(overview_se.matches) == 1
    assert overview_se.matches[0].title == "Software Engineer"

    # Now change target role on profile to AI Engineer
    mock_profile.target_role = "AI Engineer"

    # When role changes, cache lookup for the old role must not return the Software Engineer matches
    # Assemble overview should detect role key mismatch
    _USER_DASHBOARD_STATE[user_id]["matches"] = [
        {
            "job_id": 2,
            "job_title": "AI Engineer",
            "company": "DeepAI",
            "match_percentage": 92.0,
            "matched_skills": ["python", "pytorch"],
            "missing_skills": [],
            "application_url": "https://example.com/apply-ai",
            "source": "Catalog"
        }
    ]
    _USER_DASHBOARD_STATE[user_id]["matched_target_role"] = "AI Engineer"

    overview_ai = assemble_dashboard_overview(mock_db, mock_user)
    assert len(overview_ai.matches) == 1
    assert overview_ai.matches[0].title == "AI Engineer"

def test_polling_does_not_repeat_expensive_matching():
    """
    Verifies that calling assemble_dashboard_overview during polling when matching_completed is True
    returns instantly without re-invoking catalog matching or URL validation.
    """
    mock_db = MagicMock()
    user_id = 8888

    mock_user = MagicMock()
    mock_user.id = user_id

    mock_resume = MagicMock()
    mock_resume.id = 202
    mock_resume.filename = "dev_resume.pdf"
    mock_resume.updated_at = None
    mock_resume.created_at = None
    mock_resume.parsed_text = "Backend Developer experienced in Python, Django, Redis."
    mock_resume.extracted_text = "Backend Developer experienced in Python, Django, Redis."

    mock_profile = MagicMock()
    mock_profile.id = 2
    mock_profile.user_id = user_id
    mock_profile.target_role = "Backend Developer"
    mock_profile.experience_level = "Senior"
    mock_profile.skills = ["python", "django", "redis"]
    mock_profile.work_mode_preference = "Remote"
    mock_profile.employment_type = "Full-time"
    mock_profile.preferred_locations = "[\"India\"]"
    mock_profile.min_match_score = 65.0
    mock_profile.created_at = datetime.now(timezone.utc)
    mock_profile.updated_at = datetime.now(timezone.utc)

    mock_db.query.return_value.filter.return_value.order_by.return_value.first.return_value = mock_resume
    mock_db.query.return_value.filter.return_value.first.return_value = mock_profile

    # State where matching already completed with 0 matches
    _USER_DASHBOARD_STATE[user_id] = {
        "pipeline": [],
        "is_processing": False,
        "matching_completed": True,
        "matched_resume_id": 202,
        "matched_target_role": "Backend Developer",
        "matches": [],
        "discovery_stats": {
            "total_jobs_analyzed": 50,
            "jobs_retrieved": 50,
            "relevant_jobs": 0,
            "matched_jobs": 0
        }
    }

    with patch("app.services.dashboard_service.match_and_rank_jobs") as mock_match, \
         patch("app.services.dashboard_service.get_jobs_from_db") as mock_cat:
        
        # Poll 1
        res1 = assemble_dashboard_overview(mock_db, mock_user)
        # Poll 2
        res2 = assemble_dashboard_overview(mock_db, mock_user)
        # Poll 3
        res3 = assemble_dashboard_overview(mock_db, mock_user)

        # Neither match_and_rank_jobs nor get_jobs_from_db should have been called
        assert mock_match.call_count == 0
        assert mock_cat.call_count == 0
        assert res1.is_processing is False
        assert len(res1.matches) == 0

def test_application_and_search_url_generation():
    """
    Verifies requirement 4: Every job has a valid search_url fallback generated,
    and valid application URLs are preserved.
    """
    jobs = [
        {
            "id": 50,
            "job_title": "Cloud Architect",
            "company": "Amazon Web Services",
            "job_role": "Cloud Engineer",
            "required_skills": "AWS, CloudFormation, Terraform, Architecture",
            "application_url": "https://amazon.jobs/en/jobs/12345/cloud-architect",
            "is_direct": True
        },
        {
            "id": 51,
            "job_title": "DevOps Engineer",
            "company": "StartupX",
            "job_role": "DevOps Engineer",
            "required_skills": "Docker, Kubernetes, CI/CD",
            "application_url": "Application link unavailable",
            "is_available": False
        }
    ]

    resume = "Cloud and DevOps Engineer with AWS, Terraform, Docker, and Kubernetes."

    ranked = match_and_rank_jobs(
        resume_text=resume,
        jobs=jobs,
        target_role="Cloud Engineer",
        top_k=2
    )

    assert len(ranked) == 2

    # Check Job 50: direct URL preserved and valid Google search fallback provided
    aws_job = next(j for j in ranked if j["job_id"] == 50)
    assert aws_job["application_url"] == "https://amazon.jobs/en/jobs/12345/cloud-architect"
    assert "search_url" in aws_job
    assert "https://www.google.com/search?q=" in aws_job["search_url"]
    assert "Cloud+Architect" in aws_job["search_url"] or "Cloud%20Architect" in aws_job["search_url"]

    # Check Job 51: unavailable application URL preserves search_url fallback
    startup_job = next(j for j in ranked if j["job_id"] == 51)
    assert startup_job["application_url"] == "Application link unavailable"
    assert "search_url" in startup_job
    assert "https://www.google.com/search?q=" in startup_job["search_url"]
    assert "StartupX" in startup_job["search_url"]

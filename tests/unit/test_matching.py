import pytest
from app.services.skill_service import (
    extract_skills,
    normalize_skill,
    parse_job_required_skills,
    analyze_skill_gap,
    detect_candidate_role
)
from app.services.matching_service import match_and_rank_jobs, compute_skill_score

def test_skill_normalization():
    """Verifies that common skill aliases map to canonical identifiers."""
    assert normalize_skill("sklearn") == "scikit-learn"
    assert normalize_skill("postgres") == "postgresql"
    assert normalize_skill("k8s") == "kubernetes"
    assert normalize_skill("nlp") == "natural language processing"

def test_deterministic_skill_extraction():
    """Verifies exact extraction of skills from natural language text."""
    text = "Full Stack developer with 4 years building Python, FastAPI, Docker, and PostgreSQL microservices."
    skills = extract_skills(text)
    assert "python" in skills
    assert "fastapi" in skills
    assert "docker" in skills
    assert "postgresql" in skills

def test_skill_gap_analysis():
    """Verifies that matched and missing skills are strictly separated without hallucination."""
    resume_skills = ["python", "fastapi", "sql", "machine learning"]
    job_required = ["Python", "FastAPI", "SQL", "Machine Learning", "Docker", "AWS"]

    matched, missing = analyze_skill_gap(resume_skills, job_required)
    
    # Matched skills
    assert any("python" in s.lower() for s in matched)
    assert any("fastapi" in s.lower() for s in matched)
    assert any("sql" in s.lower() for s in matched)
    assert any("machine learning" in s.lower() for s in matched)

    # Missing skills
    assert "Docker" in missing
    assert "AWS" in missing
    assert len(matched) == 4
    assert len(missing) == 2

def test_role_detection():
    """Verifies role detection based on skill composition."""
    ai_skills = ["python", "pytorch", "transformers", "large language models", "langchain"]
    assert detect_candidate_role(ai_skills) == "AI Engineer"

    backend_skills = ["python", "fastapi", "flask", "django", "postgresql", "mysql", "redis"]
    assert detect_candidate_role(backend_skills) == "Backend Developer"

def test_hybrid_matching_and_ranking():
    """Verifies that match_and_rank_jobs produces deterministic, bounded match percentages."""
    synthetic_jobs = [
        {
            "id": 1,
            "job_title": "AI Engineer",
            "company": "Tech AI",
            "job_role": "AI Engineer",
            "required_skills": "Python, PyTorch, LangChain, Transformers",
            "experience_level": "2-4 years",
            "description": "Develop and deploy LLM applications.",
            "application_url": "https://careers.example.com/ai"
        },
        {
            "id": 2,
            "job_title": "Classical Violinist",
            "company": "Philharmonic",
            "job_role": "Musician",
            "required_skills": "Violin, Orchestral, Music Theory",
            "experience_level": "5+ years",
            "description": "Perform classical violin concertos.",
            "application_url": "Application link unavailable"
        }
    ]

    resume_text = "AI Engineer skilled in Python, PyTorch, Transformers, LangChain, and FastAPI."
    ranked = match_and_rank_jobs(resume_text, synthetic_jobs, top_k=2)

    assert len(ranked) == 2
    # AI Engineer job should rank higher than Musician
    assert ranked[0]["job_id"] == 1
    assert ranked[0]["match_percentage"] > ranked[1]["match_percentage"]
    assert ranked[0]["application_url"] == "https://careers.example.com/ai"
    assert ranked[1]["application_url"] == "Application link unavailable"

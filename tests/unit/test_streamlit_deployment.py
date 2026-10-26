"""
Tests for Streamlit Community Cloud Deployment Readiness.
Verifies configuration files, secrets template, .gitignore rules,
syntax, imports, and core service compatibility.
"""

import os
import sys
import py_compile
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def test_streamlit_files_exist():
    """Verify all required deployment files exist."""
    app_py = ROOT_DIR / "streamlit_app.py"
    config_toml = ROOT_DIR / ".streamlit" / "config.toml"
    secrets_example = ROOT_DIR / ".streamlit" / "secrets.toml.example"
    req_txt = ROOT_DIR / "requirements.txt"
    req_streamlit = ROOT_DIR / "requirements-streamlit.txt"

    assert app_py.exists(), "streamlit_app.py must exist"
    assert config_toml.exists(), ".streamlit/config.toml must exist"
    assert secrets_example.exists(), ".streamlit/secrets.toml.example must exist"
    assert req_txt.exists(), "requirements.txt must exist"
    assert req_streamlit.exists(), "requirements-streamlit.txt must exist"

def test_gitignore_protects_streamlit_secrets():
    """Verify .gitignore includes .streamlit/secrets.toml to prevent credential leaks."""
    gitignore_path = ROOT_DIR / ".gitignore"
    assert gitignore_path.exists(), ".gitignore must exist"
    content = gitignore_path.read_text(encoding="utf-8")
    assert ".streamlit/secrets.toml" in content, ".gitignore must contain .streamlit/secrets.toml"

def test_streamlit_app_syntax():
    """Verify streamlit_app.py has valid Python syntax without compilation errors."""
    app_py = str(ROOT_DIR / "streamlit_app.py")
    compiled = py_compile.compile(app_py, doraise=True)
    assert compiled is not None

def test_secrets_template_contains_only_placeholders():
    """Verify secrets template has no real secrets or tokens."""
    secrets_example = ROOT_DIR / ".streamlit" / "secrets.toml.example"
    content = secrets_example.read_text(encoding="utf-8")

    # Verify key placeholders exist
    assert "TURSO_DATABASE_URL" in content
    assert "TURSO_AUTH_TOKEN" in content
    assert "RAPIDAPI_KEY" in content
    assert "GEMINI_API_KEY" in content
    assert "JWT_SECRET_KEY" in content

    # Ensure no real api keys or tokens are in the template
    forbidden_tokens = ["AIzaSy", "gsk_", "eyJ", "qdrant.io:6333/"]
    for token in forbidden_tokens:
        assert token not in content, f"Template must not contain real secret pattern: {token}"

def test_config_toml_settings():
    """Verify .streamlit/config.toml contains recommended cloud server and theme settings."""
    config_toml = ROOT_DIR / ".streamlit" / "config.toml"
    content = config_toml.read_text(encoding="utf-8")

    assert "[theme]" in content
    assert "[server]" in content
    assert "headless = true" in content
    assert "maxUploadSize = 10" in content

def test_requirements_include_streamlit():
    """Verify streamlit is listed in both requirement files."""
    req_txt = (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8")
    req_streamlit = (ROOT_DIR / "requirements-streamlit.txt").read_text(encoding="utf-8")

    assert "streamlit" in req_txt.lower()
    assert "streamlit" in req_streamlit.lower()

def test_in_process_skill_extraction_and_matching():
    """Verify that backend services called by streamlit_app.py run correctly in-process."""
    from app.services.skill_service import extract_skills, detect_candidate_role, discover_relevant_roles
    from app.services.matching_service import match_and_rank_jobs, generate_platform_search_links

    sample_resume = (
        "Senior AI & Full Stack Developer with 4 years building Python, FastAPI, Docker, "
        "and PostgreSQL services. Experience with PyTorch, LangChain, Transformers, and Qdrant."
    )

    skills = extract_skills(sample_resume)
    assert "python" in skills
    assert "fastapi" in skills
    assert "docker" in skills

    detected_role = detect_candidate_role(skills, sample_resume)
    assert detected_role in ["AI Engineer", "Backend Developer"]

    discovered = discover_relevant_roles(skills, sample_resume, target_role=detected_role, max_roles=3)
    assert len(discovered) >= 1

    sample_jobs = [
        {
            "id": 1,
            "job_title": "AI Engineer",
            "company": "OpenAI Partner",
            "job_role": "AI Engineer",
            "required_skills": "Python, PyTorch, LangChain, Transformers, Docker",
            "application_url": "https://example.com/apply/1",
            "location": "India",
            "source_type": "live_api"
        },
        {
            "id": 2,
            "job_title": "Chef",
            "company": "Culinary House",
            "job_role": "Chef",
            "required_skills": "Baking, Pastry, French Cuisine",
            "application_url": "Application link unavailable",
            "location": "Paris",
            "source_type": "static_catalog"
        }
    ]

    ranked = match_and_rank_jobs(
        resume_text=sample_resume,
        jobs=sample_jobs,
        target_role=detected_role,
        relevant_roles=discovered,
        top_k=2
    )

    assert len(ranked) == 2
    assert ranked[0]["job_title"] == "AI Engineer"
    assert ranked[0]["match_percentage"] > ranked[1]["match_percentage"]
    assert len(ranked[0]["matched_skills"]) > 0

    links = generate_platform_search_links("AI Engineer", "Google", "Bangalore")
    assert "Google" in links
    assert "LinkedIn" in links
    assert "Indeed" in links

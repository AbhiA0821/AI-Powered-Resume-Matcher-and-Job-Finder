from app.services.skill_service import (
    extract_skills,
    normalize_skill,
    parse_job_required_skills,
    analyze_skill_gap,
    detect_candidate_role,
    discover_relevant_roles,
)
from app.services.matching_service import (
    match_and_rank_jobs,
    compute_skill_score,
    compute_role_score,
)

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

def test_multi_role_discovery_with_multi_skilled_resume():
    """
    Verifies that a resume containing multiple skill families (Python, Machine Learning,
    SQL, Data Analysis, NLP, LLMs, FastAPI) identifies relevant roles across those disciplines.
    """
    resume_skills = [
        "python", "machine learning", "sql", "data analysis",
        "natural language processing", "large language models", "fastapi"
    ]
    resume_text = (
        "AI and Data specialist with 4 years experience in Python, Machine Learning, "
        "SQL data pipelines, Data Analysis, FastAPI microservices, and NLP/LLM applications."
    )

    # 1. Without specified target role: should discover top roles matching skills & evidence
    discovered = discover_relevant_roles(resume_skills, extracted_text=resume_text, target_role=None, max_roles=3)
    assert len(discovered) <= 3
    assert len(discovered) >= 2
    # Discovered roles should be relevant to the candidate's skills
    valid_roles = {
        "AI Engineer", "Machine Learning Engineer", "Data Scientist",
        "Data Analyst", "Data Engineer", "Generative AI Engineer", "NLP Engineer",
        "Python Developer", "LLM Application Developer", "MLOps Engineer"
    }
    for role in discovered:
        assert role in valid_roles

    # 2. With explicit target role (e.g. AI Engineer): target role must be prioritized first
    prioritized = discover_relevant_roles(resume_skills, extracted_text=resume_text, target_role="AI Engineer", max_roles=3)
    assert len(prioritized) <= 3
    assert prioritized[0] == "AI Engineer"
    # Additional roles should still be discovered from the candidate's skills
    assert len(prioritized) >= 2
    assert any(r in [
        "Machine Learning Engineer", "Data Scientist", "Data Analyst",
        "Generative AI Engineer", "LLM Application Developer", "NLP Engineer"
    ] for r in prioritized[1:])

def test_compute_role_score_with_relevant_roles():
    """Verifies that relevant_roles boosts related jobs while maintaining target role priority."""
    target_role = "AI Engineer"
    relevant_roles = ["AI Engineer", "Machine Learning Engineer", "Data Scientist"]

    # Target role match
    score_target, label_target = compute_role_score(target_role, "AI Engineer", "Staff AI Engineer", relevant_roles)
    assert score_target == 100.0
    assert label_target == "High"

    # Discovered role match
    score_ml, label_ml = compute_role_score(target_role, "Machine Learning Engineer", "Senior ML Engineer", relevant_roles)
    assert score_ml == 85.0
    assert label_ml == "High"

    # Discovered role match via title
    score_ds, label_ds = compute_role_score(target_role, "Data Scientist", "Lead Data Scientist", relevant_roles)
    assert score_ds == 85.0
    assert label_ds == "High"

    # Unrelated role
    score_unrelated, label_unrelated = compute_role_score(target_role, "Dentist", "Dental Surgeon", relevant_roles)
    assert score_unrelated <= 40.0
    assert label_unrelated == "Fair"

def test_multi_role_matching_and_ranking():
    """
    Verifies that jobs across multiple relevant roles are scored, ranked, and
    compete fairly against unrelated positions.
    """
    resume_text = (
        "Experienced engineer skilled in Python, Machine Learning, SQL, "
        "Data Analysis, FastAPI, NLP, and Large Language Models."
    )
    relevant_roles = ["AI Engineer", "Machine Learning Engineer", "Data Scientist"]

    multi_role_jobs = [
        {
            "id": 101,
            "job_title": "AI Application Engineer",
            "company": "Cognitive AI",
            "job_role": "AI Engineer",
            "required_skills": "Python, Machine Learning, NLP, FastAPI",
            "experience_level": "Mid",
            "description": "Build generative AI and LLM workflows.",
            "application_url": "https://careers.example.com/ai-app"
        },
        {
            "id": 102,
            "job_title": "Machine Learning Engineer",
            "company": "Predictive Labs",
            "job_role": "Machine Learning Engineer",
            "required_skills": "Python, Machine Learning, SQL",
            "experience_level": "Mid",
            "description": "Develop and deploy scalable ML models.",
            "application_url": "https://careers.example.com/ml-eng"
        },
        {
            "id": 103,
            "job_title": "Data Scientist",
            "company": "Insight Analytics",
            "job_role": "Data Scientist",
            "required_skills": "Python, SQL, Data Analysis, Machine Learning",
            "experience_level": "Mid",
            "description": "Extract insights and predictive features from tabular data.",
            "application_url": "https://careers.example.com/data-sci"
        },
        {
            "id": 104,
            "job_title": "Tax Preparer",
            "company": "Finance Corp",
            "job_role": "Accounting",
            "required_skills": "Tax Law, Quickbooks, Auditing",
            "experience_level": "Senior",
            "description": "Audit corporate balance sheets.",
            "application_url": "https://careers.example.com/tax"
        }
    ]

    ranked = match_and_rank_jobs(
        resume_text=resume_text,
        jobs=multi_role_jobs,
        target_role="AI Engineer",
        relevant_roles=relevant_roles,
        top_k=4
    )

    assert len(ranked) == 4
    top_3_ids = [r["job_id"] for r in ranked[:3]]
    # AI Engineer, ML Engineer, and Data Scientist should all rank above Tax Preparer
    assert 101 in top_3_ids
    assert 102 in top_3_ids
    assert 103 in top_3_ids
    assert ranked[3]["job_id"] == 104
    # All top 3 technical jobs should have competitive match scores
    for r in ranked[:3]:
        assert r["match_percentage"] > 60.0
        assert r["application_url"].startswith("https://careers.example.com/")
    assert ranked[3]["match_percentage"] < ranked[2]["match_percentage"]


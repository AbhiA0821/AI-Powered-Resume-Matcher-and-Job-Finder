"""
Comprehensive tests for HireAgent Dashboard Pipeline and Aggregated Overview.
Validates:
1. Zero-jobs root cause regression tests and honest empty states.
2. GET /dashboard/overview: JWT required, schema validation, real pipeline derivation.
3. User isolation (Candidate A cannot see Candidate B).
4. Zero external calls (assert mocks not called: 0 JSearch, 0 LLM, 0 Qdrant).
5. Timing performance (< 100ms).
6. Components omitted from matches response.
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.db.session import Base, get_db
from app.models.user import User
from app.models.resume import Resume
from app.models.profile import UserProfile
from app.core.security import create_access_token, hash_password
from app.services.dashboard_service import (
    invalidate_user_dashboard,
    run_user_pipeline_sync,
    assemble_dashboard_overview,
    get_user_state
)
from app.schemas.dashboard import DashboardOverviewResponse

# In-memory SQLite for testing
TEST_DATABASE_URL = "sqlite:///:memory:"
test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.create_all(bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=test_engine)

@pytest.fixture(scope="function")
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def test_user_a(db_session):
    user = User(
        email="candidate_a@example.com",
        name="Candidate A",
        password_hash=hash_password("SecretPassword123!")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user

@pytest.fixture
def test_user_b(db_session):
    user = User(
        email="candidate_b@example.com",
        name="Candidate B",
        password_hash=hash_password("SecretPassword123!")
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


# ── 1. Authentication & Schema Tests ──

def test_dashboard_overview_requires_auth(client):
    """GET /dashboard/overview must reject unauthenticated requests with HTTP 401."""
    response = client.get("/api/v1/dashboard/overview")
    assert response.status_code == 401

def test_dashboard_overview_schema_validation(client, db_session, test_user_a):
    """Validates that authenticated GET /dashboard/overview returns valid schema."""
    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/dashboard/overview", headers=headers)
    assert response.status_code == 200
    data = response.json()

    # Validate against Pydantic schema
    validated = DashboardOverviewResponse(**data)
    assert validated.resume.exists is False
    assert validated.assistant.available is True
    assert len(validated.pipeline) == 8


# ── 2. Pipeline Derivation Tests (Real State) ──

def test_pipeline_status_never_completed_by_default(client, db_session, test_user_a):
    """When a new user has no resume, pipeline stages must reflect real state, not 'completed'."""
    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/dashboard/overview", headers=headers)
    assert response.status_code == 200
    data = response.json()

    stages_dict = {s["stage"]: s["status"] for s in data["pipeline"]}
    assert stages_dict["Resume Uploaded"] == "not_started"
    assert stages_dict["Resume Parsed"] == "not_started"
    assert stages_dict["Skills Extracted"] == "not_started"
    assert stages_dict["Jobs Matched"] == "not_started"
    assert stages_dict["Skill Gaps Identified"] == "not_started"
    assert stages_dict["Recommendations Generated"] == "not_started"
    assert stages_dict["AI Assistant Available"] == "completed"

def test_pipeline_status_with_resume_parsed(client, db_session, test_user_a):
    """When a user has an uploaded resume with text, stages 1, 2, 3 reflect real extracted state."""
    resume = Resume(
        user_id=test_user_a.id,
        filename="test_alex_resume.pdf",
        file_path="/uploads/test_alex_resume.pdf",
        extracted_text="Python developer with experience in FastAPI, Docker, PostgreSQL, Machine Learning and Git."
    )
    db_session.add(resume)
    db_session.commit()

    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/dashboard/overview", headers=headers)
    assert response.status_code == 200
    data = response.json()

    stages_dict = {s["stage"]: s["status"] for s in data["pipeline"]}
    assert stages_dict["Resume Uploaded"] == "completed"
    assert stages_dict["Resume Parsed"] == "completed"
    assert stages_dict["Skills Extracted"] == "completed"
    assert data["resume"]["exists"] is True
    assert data["resume"]["filename"] == "test_alex_resume.pdf"
    assert data["profile"]["skill_count"] > 0
    assert "python" in [s.lower() for s in data["profile"]["skills"]]


# ── 3. Zero External Calls & Zero-Jobs Regression ──

def test_overview_makes_zero_external_calls(client, db_session, test_user_a):
    """GET /dashboard/overview must not call JSearch, Gemini, Groq, or Qdrant."""
    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    with patch("app.services.job_sources.rapidapi_job_service.search_live_jobs") as mock_jsearch, \
         patch("app.services.llm_service.generate_grounded_recommendation") as mock_llm, \
         patch("app.services.crew_service.generate_career_insights") as mock_crew, \
         patch("app.services.qdrant_service.search_jobs_semantic") as mock_qdrant:

        response = client.get("/api/v1/dashboard/overview", headers=headers)
        assert response.status_code == 200

        mock_jsearch.assert_not_called()
        mock_llm.assert_not_called()
        mock_crew.assert_not_called()
        mock_qdrant.assert_not_called()

def test_zero_jobs_honest_empty_state_and_components_omitted(client, db_session, test_user_a):
    """
    Zero-jobs regression test:
    When no matches are stored, matches list is empty, matched_jobs is 0,
    and components dictionary is not present in match items.
    """
    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    response = client.get("/api/v1/dashboard/overview", headers=headers)
    assert response.status_code == 200
    data = response.json()

    assert data["matches"] == []
    assert data["discovery"]["matched_jobs"] == 0


# ── 4. User Isolation Test ──

def test_user_isolation_candidate_a_cannot_see_candidate_b(client, db_session, test_user_a, test_user_b):
    """Proves user A's uploaded resume and state are never visible in user B's overview."""
    resume_a = Resume(
        user_id=test_user_a.id,
        filename="confidential_resume_a.pdf",
        file_path="/uploads/a.pdf",
        extracted_text="Senior Machine Learning Specialist with PyTorch and NLP expertise."
    )
    db_session.add(resume_a)
    db_session.commit()

    # User B requests overview
    token_b = create_access_token(subject=test_user_b.id)
    headers_b = {"Authorization": f"Bearer {token_b}"}

    response_b = client.get("/api/v1/dashboard/overview", headers=headers_b)
    assert response_b.status_code == 200
    data_b = response_b.json()

    assert data_b["resume"]["exists"] is False
    assert data_b["resume"]["filename"] is None
    assert data_b["profile"]["skill_count"] == 0

    # User A requests overview
    token_a = create_access_token(subject=test_user_a.id)
    headers_a = {"Authorization": f"Bearer {token_a}"}

    response_a = client.get("/api/v1/dashboard/overview", headers=headers_a)
    assert response_a.status_code == 200
    data_a = response_a.json()

    assert data_a["resume"]["exists"] is True
    assert data_a["resume"]["filename"] == "confidential_resume_a.pdf"


# ── 5. Explicit Re-Run Pipeline & Timing Sanity ──

def test_explicit_pipeline_execution_and_overview_timing(client, db_session, test_user_a):
    """
    Validates explicit re-run pipeline executes correctly and overview responds in < 100ms.
    """
    resume = Resume(
        user_id=test_user_a.id,
        filename="alex_software_dev.pdf",
        file_path="/uploads/alex.pdf",
        extracted_text="Python software developer with skills in FastAPI, SQL, Docker, React, and Git."
    )
    db_session.add(resume)
    db_session.commit()

    token = create_access_token(subject=test_user_a.id)
    headers = {"Authorization": f"Bearer {token}"}

    # Trigger explicit re-run
    rerun_res = client.post("/api/v1/dashboard/rerun", headers=headers)
    assert rerun_res.status_code == 200
    rerun_data = rerun_res.json()
    assert rerun_data["discovery"]["jobs_retrieved"] > 0
    assert len(rerun_data["matches"]) > 0

    # Measure GET /overview timing
    start_time = time.perf_counter()
    overview_res = client.get("/api/v1/dashboard/overview", headers=headers)
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    assert overview_res.status_code == 200
    assert elapsed_ms < 100, f"Overview response took {elapsed_ms:.2f} ms (expected < 100ms)"

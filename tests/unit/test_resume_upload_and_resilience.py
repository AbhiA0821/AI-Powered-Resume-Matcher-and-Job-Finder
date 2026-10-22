import os
import io
import pytest
from unittest.mock import patch, MagicMock
from fastapi import UploadFile

from app.models.user import User
from app.models.resume import Resume
from app.services.resume import save_and_create_resume, get_user_resumes
from app.services.dashboard_service import (
    assemble_dashboard_overview,
    get_user_state,
    invalidate_user_dashboard,
    _USER_DASHBOARD_STATE,
    _OVERVIEW_CACHE,
    _ACTIVE_PIPELINE_USERS
)
from app.services.embeddings import get_embedding_model
from app.services.qdrant_service import reset_qdrant_client, delete_resume_vectors, upsert_resume_chunks
from app.services.resume_chunking import index_user_resume_vectors
from app.services.matching_service import match_and_rank_jobs
from app.services.jobs import get_jobs_from_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from app.db.session import Base

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

@pytest.fixture
def test_user(db_session):
    user = User(
        name="Resilience Test User",
        email=f"resilience_user_{os.urandom(4).hex()}@example.com",
        password_hash="hashed_pw_test"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_resume_upload_and_persistence(db_session, test_user):
    """Verifies that an uploaded resume is extracted, saved to disk, and committed to DB."""
    content = b"%PDF-1.4 Mock PDF content with FastAPI, Python, Docker, AWS, and PostgreSQL skills."
    upload_file = UploadFile(filename="resume_test.pdf", file=io.BytesIO(content))

    with patch("app.services.resume.extract_resume_text", return_value="Experienced Python engineer with FastAPI, Docker, and AWS."):
        resume = save_and_create_resume(
            db=db_session,
            user=test_user,
            file=upload_file,
            target_role="Backend Developer"
        )

    assert resume.id is not None
    assert resume.user_id == test_user.id
    assert resume.filename == "resume_test.pdf"
    assert "FastAPI" in resume.extracted_text

    # Verify persistent retrieval from database
    db_resumes = get_user_resumes(db=db_session, user=test_user)
    assert len(db_resumes) == 1
    assert db_resumes[0].id == resume.id


def test_dashboard_retrieval_after_simulated_restart(db_session, test_user):
    """Verifies that an uploaded resume remains retrievable even after in-memory state is wiped."""
    # 1. Create resume directly in DB (simulating previous session)
    resume = Resume(
        user_id=test_user.id,
        filename="persistent_resume.pdf",
        file_path="/tmp/mock_persistent.pdf",
        extracted_text="Senior Machine Learning Engineer skilled in PyTorch, Python, NLP, and LLMs."
    )
    db_session.add(resume)
    db_session.commit()

    # 2. Simulate complete application restart by wiping all in-memory caches
    _USER_DASHBOARD_STATE.clear()
    _OVERVIEW_CACHE.clear()
    _ACTIVE_PIPELINE_USERS.clear()

    # 3. Request dashboard overview
    overview = assemble_dashboard_overview(db=db_session, current_user=test_user)

    # 4. Verify resume is correctly recognized as existing
    assert overview.resume.exists is True
    assert overview.resume.filename == "persistent_resume.pdf"
    assert overview.resume.status in ["ready", "processing"]
    assert any("python" in s.lower() for s in overview.profile.skills) or "Engineer" in overview.profile.primary_role


def test_qdrant_connection_reset_resilience():
    """Verifies that [Errno 104] Connection reset by peer is handled gracefully without crashing."""
    reset_qdrant_client()

    # Mock Qdrant client throwing ConnectionResetError on upsert
    mock_client = MagicMock()
    mock_client.get_collections.side_effect = ConnectionResetError("[Errno 104] Connection reset by peer")

    with patch("app.services.qdrant_service.get_qdrant_client", return_value=mock_client):
        # Must return False rather than raising unhandled socket exception
        del_res = delete_resume_vectors(user_id=999, resume_id=888)
        assert del_res is False

        upsert_res = upsert_resume_chunks([])
        assert upsert_res is True  # Empty list returns True immediately

        from qdrant_client.http.models import PointStruct
        dummy_points = [PointStruct(id="00000000-0000-0000-0000-000000000001", vector=[0.1] * 384, payload={})]
        upsert_fail = upsert_resume_chunks(dummy_points)
        assert upsert_fail is False

    # Verify that index_user_resume_vectors does not raise an exception when Qdrant fails
    with patch("app.services.resume_chunking.upsert_resume_chunks", return_value=False):
        res = index_user_resume_vectors(
            user_id=999,
            resume_id=888,
            extracted_text="Python developer with machine learning experience."
        )
        assert res["status"] in ["skipped", "failed"]


def test_embedding_model_singleton_no_duplicate_loads():
    """Verifies that FastEmbed embedding model is loaded as a singleton and reused."""
    m1 = get_embedding_model()
    m2 = get_embedding_model()
    assert m1 is m2, "get_embedding_model() must return identical singleton instance"


def test_polling_does_not_trigger_concurrent_matching(db_session, test_user):
    """Verifies that while is_processing is True, assemble_dashboard_overview does not execute matching."""
    resume = Resume(
        user_id=test_user.id,
        filename="test_concurrent.pdf",
        file_path="/tmp/mock_concurrent.pdf",
        extracted_text="Full Stack Developer with React and Python."
    )
    db_session.add(resume)
    db_session.commit()

    # Simulate active background processing
    st = get_user_state(test_user.id)
    st["is_processing"] = True
    _ACTIVE_PIPELINE_USERS.add(test_user.id)

    with patch("app.services.dashboard_service.match_and_rank_jobs") as mock_match:
        overview = assemble_dashboard_overview(db=db_session, current_user=test_user)
        # mock_match must NOT have been called because pipeline is in progress
        mock_match.assert_not_called()
        assert overview.is_processing is True
        assert overview.resume.exists is True

    # Cleanup
    _ACTIVE_PIPELINE_USERS.discard(test_user.id)
    st["is_processing"] = False


def test_matching_with_genuine_catalog():
    """Verifies genuine semantic matching against the real job catalog."""
    cat = get_jobs_from_db(limit=10)
    jobs = cat.get("jobs", [])
    assert len(jobs) > 0, "Catalog jobs must exist"

    resume_text = "Senior Python Software Engineer with expertise in FastAPI, PostgreSQL, and Docker."
    ranked = match_and_rank_jobs(
        resume_text=resume_text,
        jobs=jobs,
        target_role="Software Engineer",
        top_k=5
    )

    assert len(ranked) > 0
    top_job = ranked[0]
    assert "match_percentage" in top_job
    assert top_job["match_percentage"] > 0
    assert "matched_skills" in top_job

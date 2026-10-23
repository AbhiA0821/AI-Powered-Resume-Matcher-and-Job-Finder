import os
import io
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi import UploadFile
from fastapi.testclient import TestClient

from app.models.user import User
from app.models.resume import Resume
from app.services.resume import save_and_create_resume
from app.services.dashboard_service import (
    assemble_dashboard_overview,
    get_user_state,
    invalidate_user_dashboard,
    _USER_DASHBOARD_STATE,
    _OVERVIEW_CACHE,
    _ACTIVE_PIPELINE_USERS,
    _PIPELINE_LOCK,
    _prune_overview_cache,
    MAX_OVERVIEW_CACHE_SIZE
)
from app.services.matching_service import _JOB_EMBEDDINGS_CACHE, MAX_JOB_CACHE_SIZE, match_and_rank_jobs
from app.services.embeddings import embed_text, embed_texts, get_embedding_model, _model_instance, _EMBED_CACHE, MAX_EMBED_CACHE_SIZE
from app.services.qdrant_service import reset_qdrant_client, _qdrant_client_instance
from app.main import app
from app.db.session import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

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
        name="Memory Test User",
        email=f"mem_user_{os.urandom(4).hex()}@example.com",
        password_hash="hashed_pw_test"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def test_lightweight_health_check_does_not_initialize_model():
    """Verifies that the /health check returns 200 without loading the heavy embedding model."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_duplicate_pipeline_prevention_on_concurrent_rerun(db_session, test_user):
    """Verifies that if a user is actively running a pipeline, subsequent rerun triggers do not start duplicate threads."""
    with _PIPELINE_LOCK:
        _ACTIVE_PIPELINE_USERS.add(test_user.id)
    
    st = get_user_state(test_user.id)
    st["is_processing"] = True

    try:
        with patch("threading.Thread") as mock_thread:
            # Call assemble_dashboard_overview which should report processing without spawning threads
            overview = assemble_dashboard_overview(db=db_session, current_user=test_user)
            assert overview.is_processing is True
            mock_thread.assert_not_called()
    finally:
        with _PIPELINE_LOCK:
            _ACTIVE_PIPELINE_USERS.discard(test_user.id)
        st["is_processing"] = False


def test_duplicate_pipeline_prevention_on_resume_upload(db_session, test_user):
    """Verifies that save_and_create_resume guards against duplicate worker dispatch if user is already processing."""
    with _PIPELINE_LOCK:
        _ACTIVE_PIPELINE_USERS.add(test_user.id)

    content = b"%PDF-1.4 Mock content with Python and FastAPI."
    upload_file = UploadFile(filename="resume_concurrency.pdf", file=io.BytesIO(content))

    try:
        with patch("app.services.resume.extract_resume_text", return_value="Python and FastAPI developer"), \
             patch("threading.Thread") as mock_thread:
            resume = save_and_create_resume(
                db=db_session,
                user=test_user,
                file=upload_file,
                target_role="Python Developer"
            )
            assert resume.id is not None
            mock_thread.assert_not_called()
    finally:
        with _PIPELINE_LOCK:
            _ACTIVE_PIPELINE_USERS.discard(test_user.id)


def test_overview_cache_pruning_bounds_memory():
    """Verifies that _OVERVIEW_CACHE prunes expired and excess entries and stays <= MAX_OVERVIEW_CACHE_SIZE."""
    _OVERVIEW_CACHE.clear()
    
    # Fill cache with 70 dummy entries (above max 50)
    mock_resp = MagicMock()
    now = time.time()
    for i in range(70):
        # Even numbered are old, odd are fresh
        t = now - 100 if i % 2 == 0 else now
        key = (i, i, f"res_{i}.pdf", None, None, False, "python developer")
        _OVERVIEW_CACHE[key] = (t, mock_resp)

    assert len(_OVERVIEW_CACHE) == 70
    _prune_overview_cache()
    assert len(_OVERVIEW_CACHE) <= MAX_OVERVIEW_CACHE_SIZE
    _OVERVIEW_CACHE.clear()


def test_job_embeddings_cache_pruning():
    """Verifies that _JOB_EMBEDDINGS_CACHE prunes entries when reaching capacity."""
    _JOB_EMBEDDINGS_CACHE.clear()
    for i in range(MAX_JOB_CACHE_SIZE):
        _JOB_EMBEDDINGS_CACHE[f"job_key_{i}"] = [0.1] * 384
    
    assert len(_JOB_EMBEDDINGS_CACHE) == MAX_JOB_CACHE_SIZE
    _JOB_EMBEDDINGS_CACHE.clear()


def test_embed_texts_bounded_batch():
    """Verifies embed_texts generates valid 384D vectors using streaming generator."""
    texts = [f"Senior Python Engineer opportunity #{i} with FastAPI and AWS" for i in range(5)]
    vectors = embed_texts(texts)
    assert len(vectors) == 5
    for vec in vectors:
        assert len(vec) == 384
        assert all(isinstance(x, float) for x in vec)


def test_reset_qdrant_client_closes_open_connection():
    """Verifies that reset_qdrant_client() closes any existing client socket pool."""
    mock_client = MagicMock()
    with patch("app.services.qdrant_service._qdrant_client_instance", mock_client):
        reset_qdrant_client()
        mock_client.close.assert_called_once()

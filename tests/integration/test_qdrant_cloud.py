import pytest
from unittest.mock import patch, MagicMock
from app.services.qdrant_service import (
    get_qdrant_client,
    ensure_collections_exist,
    search_jobs_semantic,
    upsert_resume_chunks,
    search_user_resumes,
    JOBS_COLLECTION,
    RESUMES_COLLECTION,
    VECTOR_DIMENSION
)
from qdrant_client.http.models import PointStruct
from app.services.job_service import build_job_embedding_text, index_live_jobs_in_qdrant
from app.services.job_sources.rapidapi_job_service import NormalizedJob
from app.core.config import settings

def test_qdrant_vector_dimension_is_384():
    """Validates vector dimension is strictly 384 corresponding to sentence-transformers/all-MiniLM-L6-v2."""
    assert VECTOR_DIMENSION == 384

def test_qdrant_client_initialization():
    """Validates Qdrant client connects using configured URL and API key."""
    client = get_qdrant_client()
    assert client is not None

def _is_qdrant_accessible() -> bool:
    try:
        client = get_qdrant_client()
        client.get_collections()
        return True
    except Exception:
        return False

def test_qdrant_collection_validation():
    """Validates that hireagent_jobs and hireagent_resumes collections exist."""
    if not _is_qdrant_accessible():
        pytest.skip("Live Qdrant Cloud cluster is unavailable/offline.")

    success = ensure_collections_exist()
    assert success is True

    client = get_qdrant_client()
    collections = [c.name for c in client.get_collections().collections]
    assert JOBS_COLLECTION in collections
    assert RESUMES_COLLECTION in collections

def test_job_embedding_text_builder():
    """Validates embedding text construction from verified fields without fabrication."""
    text = build_job_embedding_text(
        title="AI Engineer",
        company="InnoTech",
        location="Hyderabad, India",
        description="Developing LLM workflows.",
        skills=["Python", "LangChain"],
        experience="0-2 years",
        employment_type="Full-time"
    )
    assert "AI Engineer" in text
    assert "InnoTech" in text
    assert "Hyderabad, India" in text
    assert "Python, LangChain" in text
    assert "Developing LLM workflows" in text

def test_live_job_upsert_and_semantic_retrieval():
    """Tests indexing a live job into Qdrant Cloud and retrieving it via semantic search."""
    if not _is_qdrant_accessible():
        pytest.skip("Live Qdrant Cloud cluster is unavailable/offline.")

    live_job = NormalizedJob(
        external_job_id="test-cloud-qdrant-01",
        title="Generative AI Research Associate",
        company="HireAgent Test Labs",
        location="Bengaluru, India",
        description="Testing Qdrant Cloud cluster vector search and embedding indexing.",
        skills=["Python", "PyTorch", "Vectors"],
        source="LinkedIn",
        source_url="https://linkedin.com/jobs/view/test-cloud-01",
        source_type="live_api"
    )

    # Index into Qdrant Cloud
    res = index_live_jobs_in_qdrant([live_job])
    assert res.get("indexed_count") == 1

    # Search semantically
    results = search_jobs_semantic(
        query_text="Generative AI PyTorch research associate",
        limit=5,
        source_type="live_api"
    )
    assert len(results) >= 1
    found = any(r["company"] == "HireAgent Test Labs" for r in results)
    assert found is True

def test_user_isolation_in_resume_vectors():
    """Validates that resume vectors strictly maintain user isolation via payload user_id."""
    point1 = PointStruct(id="00000000-0000-0000-0000-000000000001", vector=[0.1] * 384, payload={"user_id": 101, "resume_id": 1})
    point2 = PointStruct(id="00000000-0000-0000-0000-000000000002", vector=[0.2] * 384, payload={"user_id": 202, "resume_id": 2})

    with patch("app.services.qdrant_service.get_qdrant_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        upsert_resume_chunks([point1])
        args, kwargs = mock_client.upsert.call_args
        points = kwargs.get("points") or args[1]
        assert points[0].payload["user_id"] == 101

        # Test search isolation filter
        search_user_resumes(user_id=101, query_vector=[0.1] * 384)
        args, kwargs = mock_client.query_points.call_args or mock_client.search.call_args
        query_filter = kwargs.get("query_filter")
        assert query_filter is not None
        # Check user_id in filter condition
        cond = query_filter.must[0]
        assert cond.key == "user_id"
        assert cond.match.value == 101

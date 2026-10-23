import os
import uuid
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from app.core.config import settings

RESUMES_COLLECTION = getattr(settings, "QDRANT_RESUMES_COLLECTION", "hireagent_resumes")
JOBS_COLLECTION = getattr(settings, "QDRANT_JOBS_COLLECTION", "hireagent_jobs")
EMBEDDING_DIM = getattr(settings, "EMBEDDING_DIMENSION", 384)
VECTOR_DIMENSION = EMBEDDING_DIM

_qdrant_client_instance: Optional[QdrantClient] = None

import time
import logging

logger = logging.getLogger("hireagent.qdrant")

def reset_qdrant_client():
    """Resets the singleton QdrantClient instance so broken or reset sockets are discarded."""
    global _qdrant_client_instance, _ENSURED_COLLECTIONS
    if _qdrant_client_instance is not None:
        try:
            _qdrant_client_instance.close()
        except Exception:
            pass
    _qdrant_client_instance = None
    _ENSURED_COLLECTIONS.clear()

def _is_connection_reset_error(exc: Exception) -> bool:
    """Detects transient socket resets or peer disconnections."""
    msg = str(exc).lower()
    return (
        isinstance(exc, (ConnectionResetError, ConnectionError)) or
        "connection reset" in msg or
        "errno 104" in msg or
        "remoteprotocolerror" in msg or
        "connection closed" in msg or
        "broken pipe" in msg
    )

def get_qdrant_client() -> QdrantClient:
    """Lazily initializes and returns the Qdrant Cloud client singleton using environment settings."""
    global _qdrant_client_instance
    if _qdrant_client_instance is None:
        url = settings.QDRANT_URL or os.getenv("QDRANT_URL")
        api_key = settings.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY")
        
        if not url or not api_key:
            raise RuntimeError("Qdrant Cloud credentials missing. Please set QDRANT_URL and QDRANT_API_KEY environment variables.")
        
        qdrant_timeout = float(os.getenv("QDRANT_TIMEOUT", "10.0"))
        _qdrant_client_instance = QdrantClient(url=url, api_key=api_key, timeout=qdrant_timeout, check_compatibility=False)
    return _qdrant_client_instance

_ENSURED_COLLECTIONS = set()

def ensure_collection(collection_name: str) -> bool:
    """Ensures a HireAgent Qdrant collection exists with 384D Cosine distance configuration.
    If the collection exists, it is reused safely without deleting or altering existing data.
    Caches verified collections in memory to eliminate redundant remote API round-trips.
    Retries once on transient socket resets.
    """
    global _ENSURED_COLLECTIONS
    if collection_name in _ENSURED_COLLECTIONS:
        return True

    for attempt in range(2):
        try:
            client = get_qdrant_client()
            collections_response = client.get_collections()
            existing_names = [c.name for c in collections_response.collections]

            if collection_name not in existing_names:
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
                )
            
            # Ensure payload indexes for filtering on RESUMES_COLLECTION
            if collection_name == RESUMES_COLLECTION:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="user_id",
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="resume_id",
                        field_schema=models.PayloadSchemaType.INTEGER
                    )
                except Exception:
                    pass

            # Ensure payload indexes for JOBS_COLLECTION
            if collection_name == JOBS_COLLECTION:
                try:
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source_type",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                    client.create_payload_index(
                        collection_name=collection_name,
                        field_name="source",
                        field_schema=models.PayloadSchemaType.KEYWORD
                    )
                except Exception:
                    pass

            # Verify existing collection vector size
            collection_info = client.get_collection(collection_name)
            vec_config = collection_info.config.params.vectors
            
            if isinstance(vec_config, VectorParams):
                actual_size = vec_config.size
            elif isinstance(vec_config, dict) and "size" in vec_config:
                actual_size = vec_config["size"]
            else:
                actual_size = EMBEDDING_DIM
                
            if actual_size != EMBEDDING_DIM:
                raise ValueError(
                    f"Collection '{collection_name}' exists but has vector size {actual_size}, "
                    f"expected {EMBEDDING_DIM} for {settings.EMBEDDING_MODEL}."
                )
            _ENSURED_COLLECTIONS.add(collection_name)
            return True
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in ensure_collection, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Failed to ensure Qdrant collection '{collection_name}': {e}")
            reset_qdrant_client()
            return False
    return False

def ensure_collections_exist() -> bool:
    """Ensures both hireagent_jobs and hireagent_resumes collections exist."""
    res1 = ensure_collection(JOBS_COLLECTION)
    res2 = ensure_collection(RESUMES_COLLECTION)
    return bool(res1 and res2)

def delete_resume_vectors(user_id: int, resume_id: int) -> bool:
    """Idempotently deletes previous points belonging strictly to user_id and resume_id in hireagent_resumes collection."""
    for attempt in range(2):
        try:
            if not ensure_collection(RESUMES_COLLECTION):
                return False
            client = get_qdrant_client()
            client.delete(
                collection_name=RESUMES_COLLECTION,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                        FieldCondition(key="resume_id", match=MatchValue(value=resume_id))
                    ]
                )
            )
            return True
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in delete_resume_vectors, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Qdrant delete_resume_vectors failed for user {user_id}, resume {resume_id}: {e}")
            reset_qdrant_client()
            return False
    return False

def upsert_resume_chunks(points: List[PointStruct]) -> bool:
    """Upserts points into the hireagent_resumes collection with connection reset recovery."""
    if not points:
        return True
    for attempt in range(2):
        try:
            if not ensure_collection(RESUMES_COLLECTION):
                return False
            client = get_qdrant_client()
            for p in points:
                if p.payload is not None and "embedding_model" not in p.payload:
                    p.payload["embedding_model"] = settings.EMBEDDING_MODEL
            client.upsert(collection_name=RESUMES_COLLECTION, points=points)
            return True
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in upsert_resume_chunks, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Qdrant upsert_resume_chunks failed: {e}")
            reset_qdrant_client()
            return False
    return False

def search_user_resumes(
    user_id: int,
    query_vector: List[float],
    limit: int = 5,
    resume_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Performs semantic similarity search restricted strictly to user_id's vectors in hireagent_resumes.
    Enforces user isolation at the database filter level.
    """
    for attempt in range(2):
        try:
            if not ensure_collection(RESUMES_COLLECTION):
                return []
            client = get_qdrant_client()
            
            must_filters = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            if resume_id is not None:
                must_filters.append(FieldCondition(key="resume_id", match=MatchValue(value=resume_id)))
                
            search_filter = Filter(must=must_filters)
            
            res = client.query_points(
                collection_name=RESUMES_COLLECTION,
                query=query_vector,
                query_filter=search_filter,
                limit=limit
            )
            
            results = []
            for hit in res.points:
                payload = hit.payload or {}
                results.append({
                    "id": str(hit.id),
                    "score": round(float(hit.score), 4) if hasattr(hit, "score") and hit.score is not None else 0.0,
                    "chunk_id": payload.get("chunk_id"),
                    "resume_id": payload.get("resume_id"),
                    "section": payload.get("section", "general"),
                    "text": payload.get("text", "")
                })
            return results
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in search_user_resumes, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Qdrant search_user_resumes failed: {e}")
            reset_qdrant_client()
            return []
    return []

def upsert_job_vectors(points: List[PointStruct]) -> bool:
    """Upserts points into the hireagent_jobs collection."""
    if not points:
        return True
    for attempt in range(2):
        try:
            if not ensure_collection(JOBS_COLLECTION):
                return False
            client = get_qdrant_client()
            for p in points:
                if p.payload is not None and "embedding_model" not in p.payload:
                    p.payload["embedding_model"] = settings.EMBEDDING_MODEL
            client.upsert(collection_name=JOBS_COLLECTION, points=points)
            return True
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in upsert_job_vectors, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Qdrant upsert_job_vectors failed: {e}")
            reset_qdrant_client()
            return False
    return False

def search_jobs_semantic(
    query_vector: Optional[List[float]] = None,
    limit: int = 5,
    source_type: Optional[str] = None,
    query_text: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Performs semantic similarity search across hireagent_jobs collection with optional filtering."""
    if query_vector is None and query_text is not None:
        from app.services.embeddings import embed_text
        query_vector = embed_text(query_text)
    elif isinstance(query_vector, str):
        from app.services.embeddings import embed_text
        query_vector = embed_text(query_vector)

    if not query_vector:
        return []

    for attempt in range(2):
        try:
            if not ensure_collection(JOBS_COLLECTION):
                return []
            client = get_qdrant_client()

            query_filter = None
            if source_type:
                query_filter = Filter(
                    must=[FieldCondition(key="source_type", match=MatchValue(value=source_type))]
                )
            
            res = client.query_points(
                collection_name=JOBS_COLLECTION,
                query=query_vector,
                query_filter=query_filter,
                limit=limit
            )
            
            results = []
            for hit in res.points:
                payload = hit.payload or {}
                results.append({
                    "score": float(hit.score),
                    "job_id": payload.get("job_id"),
                    "job_title": payload.get("job_title", ""),
                    "company": payload.get("company", ""),
                    "job_role": payload.get("job_role", ""),
                    "required_skills": payload.get("required_skills", ""),
                    "application_url": payload.get("application_url", "Application link unavailable"),
                    "source": payload.get("source", "HireAgent Test Catalog"),
                    "source_type": payload.get("source_type", "static_catalog"),
                    "location": payload.get("location", "India"),
                    "posted_at": payload.get("posted_at", ""),
                    "text": payload.get("text", "")
                })
            return results
        except Exception as e:
            if _is_connection_reset_error(e) and attempt == 0:
                logger.warning(f"Qdrant connection reset in search_jobs_semantic, resetting client and retrying: {e}")
                reset_qdrant_client()
                time.sleep(0.5)
                continue
            logger.warning(f"Qdrant search_jobs_semantic failed: {e}")
            reset_qdrant_client()
            return []
    return []


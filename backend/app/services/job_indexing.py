import uuid
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import PointStruct
from app.core.config import settings
from app.core.memory import log_memory_checkpoint
from app.services.embeddings import embed_text, embed_texts
from app.services.qdrant_service import (
    upsert_job_vectors,
    search_jobs_semantic,
    search_user_resumes
)

def generate_deterministic_job_point_id(job_id: int) -> str:
    """Generates a reproducible UUID string for a job_id.
    Ensures idempotent indexing across re-index operations.
    """
    unique_key = f"hireagent:job:{job_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

def construct_job_text_representation(job: Dict[str, Any]) -> str:
    """Constructs a deterministic text representation from actual existing job fields.
    Does NOT invent skills, companies, salaries, or descriptions.
    """
    job_title = job.get("job_title", "")
    company = job.get("company", "Company")
    job_role = job.get("job_role", "")
    required_skills = job.get("required_skills", "")
    exp = job.get("experience_level", "")
    desc = job.get("description", "")
    
    parts = []
    if job_title:
        parts.append(f"Title: {job_title}")
    if company:
        parts.append(f"Company: {company}")
    if job_role:
        parts.append(f"Role: {job_role}")
    if required_skills:
        parts.append(f"Required Skills: {required_skills}")
    if exp:
        parts.append(f"Experience: {exp}")
    if desc:
        parts.append(f"Description: {desc}")
        
    return " | ".join(parts)

def build_job_embedding_text(
    title: str = "",
    company: str = "",
    location: str = "",
    description: str = "",
    skills: Optional[List[str]] = None,
    experience: str = "",
    employment_type: str = ""
) -> str:
    """Builds textual representation for job embedding from verified fields."""
    parts = []
    if title:
        parts.append(title)
    if company:
        parts.append(f"Company: {company}")
    if location:
        parts.append(f"Location: {location}")
    if skills:
        parts.append(f"Skills: {', '.join(skills)}")
    if experience:
        parts.append(f"Experience: {experience}")
    if employment_type:
        parts.append(f"Type: {employment_type}")
    if description:
        parts.append(f"Description: {description}")
    return ". ".join(parts)

def index_all_jobs(jobs: List[Dict[str, Any]], batch_size: int = 32) -> Dict[str, Any]:
    """Indexes a list of job records into the hireagent_jobs Qdrant collection idempotently."""
    if not jobs:
        return {"indexed_count": 0, "status": "no jobs to index"}
        
    log_memory_checkpoint("job_indexing:start", f"total_jobs={len(jobs)}")
    text_representations = [construct_job_text_representation(job) for job in jobs]
    
    # Process batch embeddings in bounded chunks to keep memory usage predictable
    vectors: List[List[float]] = []
    for i in range(0, len(text_representations), batch_size):
        chunk = text_representations[i:i + batch_size]
        chunk_vecs = embed_texts(chunk)
        vectors.extend(chunk_vecs)
    
    points: List[PointStruct] = []
    for idx, job in enumerate(jobs):
        job_id = job.get("id", idx + 1)
        point_id = generate_deterministic_job_point_id(job_id)
        payload = {
            "job_id": job_id,
            "external_job_id": str(job.get("external_job_id") or job_id),
            "job_title": job.get("job_title", ""),
            "company": job.get("company", ""),
            "job_role": job.get("job_role", ""),
            "required_skills": job.get("required_skills", ""),
            "experience_level": job.get("experience_level", ""),
            "application_url": job.get("application_url", "Application link unavailable"),
            "source": job.get("source", "HireAgent Test Catalog"),
            "source_type": job.get("source_type", "static_catalog"),
            "location": job.get("location", "India"),
            "posted_at": job.get("posted_at", ""),
            "text": text_representations[idx],
            "embedding_model": settings.EMBEDDING_MODEL
        }
        points.append(PointStruct(
            id=point_id,
            vector=vectors[idx],
            payload=payload
        ))
        
    upsert_job_vectors(points)
    log_memory_checkpoint("job_indexing:complete", f"indexed_points={len(points)}")
    return {"indexed_count": len(points), "status": "success"}

def index_live_jobs_in_qdrant(jobs: List[Any]) -> Dict[str, Any]:
    """Indexes live fresh jobs from RapidAPI into Qdrant Cloud under hireagent_jobs with source_type='live_api'."""
    normalized_dicts = []
    for idx, j in enumerate(jobs):
        if hasattr(j, "model_dump"):
            from app.services.job_sources import to_hireagent_job_dict
            job_dict = to_hireagent_job_dict(j, auto_id=idx + 100000)
        elif isinstance(j, dict):
            job_dict = dict(j)
        else:
            continue
        job_dict["source_type"] = "live_api"
        normalized_dicts.append(job_dict)
    try:
        return index_all_jobs(normalized_dicts)
    except Exception as e:
        import logging
        logging.getLogger("hireagent.indexing").warning(f"Background Qdrant indexing for live jobs skipped: {e}")
        return {"indexed_count": 0, "status": "skipped", "error": str(e)}


def search_jobs_by_query_semantic(query_text: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Performs semantic similarity search against indexed jobs using a text query."""
    if not query_text or not query_text.strip():
        raise ValueError("Query string cannot be empty.")
        
    query_vector = embed_text(query_text)
    return search_jobs_semantic(query_vector=query_vector, limit=limit)

def retrieve_matched_jobs_for_resume(user_id: int, resume_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
    """Performs Resume <-> Job semantic retrieval by querying job vectors closest to the user's resume vector.
    Enforces user isolation by fetching vectors strictly belonging to user_id and resume_id.
    """
    # 1. Fetch chunks belonging strictly to user_id and resume_id from hireagent_resumes
    from app.services.qdrant_service import RESUMES_COLLECTION, get_qdrant_client
    from qdrant_client.http import models
    
    client = get_qdrant_client()
    user_points, _ = client.scroll(
        collection_name=RESUMES_COLLECTION,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id)),
                models.FieldCondition(key="resume_id", match=models.MatchValue(value=resume_id))
            ]
        ),
        limit=20,
        with_vectors=True
    )
    
    if not user_points:
        return []
        
    # 2. Average the vectors of the resume chunks to form a composite resume representation vector
    vec_dim = len(user_points[0].vector)
    composite_vec = [0.0] * vec_dim
    for p in user_points:
        for idx, val in enumerate(p.vector):
            composite_vec[idx] += val / len(user_points)
            
    # L2 Normalize the composite vector
    import math
    norm = math.sqrt(sum(v * v for v in composite_vec))
    if norm > 0:
        composite_vec = [v / norm for v in composite_vec]
        
    # 3. Search hireagent_jobs collection using composite resume vector
    return search_jobs_semantic(query_vector=composite_vec, limit=top_k)

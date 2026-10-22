import uuid
import re
from typing import List, Dict, Any, Optional
from qdrant_client.http.models import PointStruct
from app.core.config import settings
from app.core.memory import log_memory_checkpoint
from app.services.embeddings import embed_texts
from app.services.qdrant_service import (
    delete_resume_vectors,
    upsert_resume_chunks,
    search_user_resumes
)

CHUNK_SIZE = 500  # Approx 500 characters per chunk
CHUNK_OVERLAP = 100  # 100 character overlap for context preservation

def chunk_text_deterministically(text: str) -> List[Dict[str, Any]]:
    """Splits raw text into deterministic, ordered, non-empty chunks with context overlap.
    Preserves exact source text without LLM rewriting, summarization, or fact generation.
    """
    if not text or not text.strip():
        return []
        
    cleaned_text = text.strip()
    
    # Simple paragraph / double newline split or character windowing with sentence boundary fallback
    chunks: List[Dict[str, Any]] = []
    start = 0
    text_len = len(cleaned_text)
    chunk_idx = 0
    
    while start < text_len:
        end = min(start + CHUNK_SIZE, text_len)
        
        # If not at the end of the text, try to find a natural boundary (newline or period)
        if end < text_len:
            boundary = cleaned_text.rfind('\n', start, end)
            if boundary == -1 or boundary <= start:
                boundary = cleaned_text.rfind('. ', start, end)
            if boundary > start:
                end = boundary + 1
                
        chunk_content = cleaned_text[start:end].strip()
        if chunk_content:
            chunks.append({
                "chunk_id": f"chunk_{chunk_idx}",
                "section": "extracted_text",
                "text": chunk_content
            })
            chunk_idx += 1
            
        start = end - CHUNK_OVERLAP if end < text_len else text_len
        if start <= 0 or start >= text_len:
            break
            
    # Fallback if text was shorter than chunk size
    if not chunks and cleaned_text:
        chunks.append({
            "chunk_id": "chunk_0",
            "section": "extracted_text",
            "text": cleaned_text
        })
        
    return chunks

def generate_deterministic_point_id(user_id: int, resume_id: int, chunk_id: str) -> str:
    """Generates a reproducible UUID string for a specific user, resume, and chunk identifier.
    Ensures idempotent indexing across re-index operations.
    """
    unique_key = f"hireagent:user:{user_id}:resume:{resume_id}:{chunk_id}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, unique_key))

def index_user_resume_vectors(user_id: int, resume_id: int, extracted_text: str) -> Dict[str, Any]:
    """Indexes or re-indexes a user's resume into Qdrant Cloud idempotently.
    
    Flow:
    1. Idempotently removes previous points belonging to (user_id, resume_id).
    2. Deterministically chunks extracted_text.
    3. Embeds text chunks into 384D vectors using sentence-transformers/all-MiniLM-L6-v2.
    4. Upserts points with payload containing user_id, resume_id, chunk_id, section, text, and embedding_model.
    5. Releases temporary vectors and objects from memory immediately.
    """
    import logging
    logger = logging.getLogger("hireagent.chunking")

    if not extracted_text or not extracted_text.strip():
        raise ValueError("Cannot index resume with empty extracted text.")
        
    try:
        log_memory_checkpoint("resume_chunking:start", f"user={user_id} resume={resume_id}")
        # Step 1: Idempotent deletion of existing vectors for this resume
        delete_resume_vectors(user_id=user_id, resume_id=resume_id)
        
        # Step 2: Deterministic chunking
        chunks = chunk_text_deterministically(extracted_text)
        if not chunks:
            return {"resume_id": resume_id, "user_id": user_id, "indexed_chunks": 0, "status": "skipped"}
            
        chunk_texts = [c["text"] for c in chunks]
        
        # Step 3: Embed text chunks
        vectors = embed_texts(chunk_texts)
        
        # Step 4: Construct PointStruct instances with deterministic IDs
        points: List[PointStruct] = []
        for idx, chunk in enumerate(chunks):
            point_id = generate_deterministic_point_id(user_id, resume_id, chunk["chunk_id"])
            payload = {
                "user_id": user_id,
                "resume_id": resume_id,
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "text": chunk["text"],
                "embedding_model": settings.EMBEDDING_MODEL
            }
            points.append(PointStruct(
                id=point_id,
                vector=vectors[idx],
                payload=payload
            ))
            
        # Step 5: Upsert to Qdrant Cloud
        upsert_ok = upsert_resume_chunks(points)
        status_label = "success" if upsert_ok else "skipped"
        indexed_count = len(points) if upsert_ok else 0
        log_memory_checkpoint("resume_chunking:complete", f"indexed_chunks={indexed_count}")

        # Release memory buffers immediately
        del points
        del vectors
        del chunk_texts
        del chunks
        import gc
        gc.collect()

        return {
            "user_id": user_id,
            "resume_id": resume_id,
            "indexed_chunks": indexed_count,
            "status": status_label
        }
    except Exception as e:
        logger.warning(f"Vector indexing skipped for resume {resume_id}: {e}")
        import gc
        gc.collect()
        return {
            "user_id": user_id,
            "resume_id": resume_id,
            "indexed_chunks": 0,
            "status": "failed",
            "error": str(e)
        }

def search_resume_semantic(user_id: int, query: str, limit: int = 5, resume_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Performs semantic search over an authenticated user's indexed resume chunks."""
    if not query or not query.strip():
        raise ValueError("Query string cannot be empty.")
        
    from app.services.embeddings import embed_text
    query_vector = embed_text(query)
    
    return search_user_resumes(
        user_id=user_id,
        query_vector=query_vector,
        limit=limit,
        resume_id=resume_id
    )

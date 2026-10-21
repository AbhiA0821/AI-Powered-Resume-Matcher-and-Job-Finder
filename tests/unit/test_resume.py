import pytest
from app.services.resume_parser import extract_resume_text
from app.services.resume_chunking import (
    chunk_text_deterministically,
    generate_deterministic_point_id
)
from app.services.embedding_service import embed_text, embed_texts, EMBEDDING_DIMENSION

def test_resume_chunking_deterministic():
    """Verifies that chunk_text_deterministically produces stable, non-empty chunks with overlap."""
    sample_text = (
        "Jane Doe - Senior Machine Learning Engineer\n"
        "Proficient in Python, PyTorch, TensorFlow, Scikit-learn, and FastAPI.\n"
        "Experience building scalable retrieval-augmented generation (RAG) systems and MLOps pipelines."
    )
    chunks = chunk_text_deterministically(sample_text)
    assert len(chunks) >= 1
    assert "chunk_0" in chunks[0]["chunk_id"]
    assert "Machine Learning" in chunks[0]["text"]

def test_deterministic_point_id_consistency():
    """Verifies that identical user, resume, and chunk identifiers produce identical UUIDs."""
    id1 = generate_deterministic_point_id(user_id=101, resume_id=5, chunk_id="chunk_0")
    id2 = generate_deterministic_point_id(user_id=101, resume_id=5, chunk_id="chunk_0")
    id3 = generate_deterministic_point_id(user_id=102, resume_id=5, chunk_id="chunk_0")
    
    assert id1 == id2, "Identical inputs must yield identical point IDs"
    assert id1 != id3, "Different users must yield distinct point IDs (isolation)"

def test_resume_parser_empty_fallback():
    """Verifies safe handling of non-existent or invalid file paths."""
    assert extract_resume_text("non_existent_file.xyz") == ""

def test_resume_embeddings_generation():
    """Verifies that BAAI/bge-small-en-v1.5 generates 384D normalized vectors."""
    vec = embed_text("Senior Full Stack Python Developer with FastAPI and Docker")
    assert isinstance(vec, list)
    assert len(vec) == EMBEDDING_DIMENSION
    assert len(vec) == 384

def test_resume_batch_embeddings():
    """Verifies batch embedding generation preserving order."""
    texts = ["Python Developer", "Machine Learning Engineer"]
    vecs = embed_texts(texts)
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    assert len(vecs[1]) == 384
    assert vecs[0] != vecs[1]

import math
import logging
from typing import List, Optional

logger = logging.getLogger("hireagent.embeddings")

# Singleton container for lazy model initialization
_model_instance = None
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
_EMBED_CACHE: dict = {}

def get_embedding_model():
    """Lazily initializes and returns the embedding model singleton.
    Uses FastEmbed (ONNX Runtime) for ultra-low memory footprint (~67MB model, no PyTorch overhead),
    ensuring operation comfortably within Render Free's 512MB RAM constraint.
    Falls back gracefully to SentenceTransformer if FastEmbed is unavailable.
    """
    global _model_instance
    if _model_instance is None:
        try:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if hf_token and not os.environ.get("HF_TOKEN"):
                os.environ["HF_TOKEN"] = hf_token

            from fastembed import TextEmbedding
            # Initialize with 1 thread for predictable minimal memory footprint
            _model_instance = TextEmbedding(MODEL_NAME, threads=1)
            logger.info(f"Loaded FastEmbed ONNX embedding model: {MODEL_NAME}")
        except Exception as fe_err:
            logger.warning(f"FastEmbed initialization notice: {fe_err}. Falling back to SentenceTransformer.")
            try:
                from sentence_transformers import SentenceTransformer
                try:
                    _model_instance = SentenceTransformer(MODEL_NAME, local_files_only=True)
                except Exception:
                    _model_instance = SentenceTransformer(MODEL_NAME)
            except Exception as st_err:
                logger.error(f"Failed to load embedding model: {st_err}")
                raise RuntimeError(f"Embedding model unavailable: {st_err}")
    return _model_instance

def _validate_single_text(text: str, arg_name: str = "text") -> str:
    """Validates that an input argument is a non-empty, non-whitespace string."""
    if text is None or not isinstance(text, str):
        raise ValueError(f"Input '{arg_name}' must be a non-empty string.")
    cleaned = text.strip()
    if not cleaned:
        raise ValueError(f"Input '{arg_name}' cannot be empty or whitespace-only.")
    return text

def _validate_vector_output(vector: List[float]) -> List[float]:
    """Validates that generated vector matches expected dimension (384) and contains no NaN/infinite values."""
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(vector)}.")
    
    for idx, val in enumerate(vector):
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid non-finite embedding value detected at index {idx}: {val}")
    return vector

def embed_text(text: str) -> List[float]:
    """Converts a single input string into a 384-dimensional normalized vector representation.
    
    Architectural Principle:
    Embeddings are deterministic vector representations of text (Text -> Vector).
    They do NOT generate facts, summarize text, or infer missing resume skills.
    
    Normalization Note:
    'normalize_embeddings=True' is enabled so that the resulting vectors are L2-normalized (unit length).
    For normalized vectors, cosine similarity equals the dot/inner product.
    """
    valid_text = _validate_single_text(text, arg_name="text")
    if valid_text in _EMBED_CACHE:
        return _EMBED_CACHE[valid_text]

    model = get_embedding_model()
    
    # Generate normalized embedding vector via FastEmbed or fallback
    if hasattr(model, "embed"):
        raw_embedding = next(iter(model.embed([valid_text])))
        vector = [float(x) for x in raw_embedding.tolist()]
    else:
        raw_embedding = model.encode(valid_text, normalize_embeddings=True, convert_to_numpy=True)
        vector = [float(x) for x in raw_embedding.tolist()]

    validated = _validate_vector_output(vector)
    
    if len(_EMBED_CACHE) < 2000:
        _EMBED_CACHE[valid_text] = validated
    return validated

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Converts a batch list of input strings into a list of 384-dimensional normalized vector representations.
    Preserves input ordering exactly.
    """
    if texts is None or not isinstance(texts, list):
        raise ValueError("Input 'texts' must be a list of strings.")
    if len(texts) == 0:
        raise ValueError("Input 'texts' list cannot be empty.")
    
    for idx, item in enumerate(texts):
        _validate_single_text(item, arg_name=f"texts[{idx}]")
        
    model = get_embedding_model()
    
    # Generate batch embeddings preserving order
    output_vectors: List[List[float]] = []
    if hasattr(model, "embed"):
        raw_embeddings = list(model.embed(texts))
        for raw_vec in raw_embeddings:
            vector = [float(x) for x in raw_vec.tolist()]
            output_vectors.append(_validate_vector_output(vector))
    else:
        raw_embeddings = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        for raw_vec in raw_embeddings:
            vector = [float(x) for x in raw_vec.tolist()]
            output_vectors.append(_validate_vector_output(vector))
        
    return output_vectors

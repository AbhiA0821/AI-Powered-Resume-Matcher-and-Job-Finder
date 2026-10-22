import os
import math
import logging
import threading
from typing import List, Optional

from app.core.config import settings
from app.core.memory import log_memory_checkpoint

logger = logging.getLogger("hireagent.embeddings")

# Enforce minimal thread footprint across BLAS / OpenMP runtimes for ONNX
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Centralized model identifier and dimension
MODEL_NAME = settings.EMBEDDING_MODEL
EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION

# Thread-safe singleton locks
_model_init_lock = threading.Lock()
_inference_lock = threading.Lock()
_cache_lock = threading.Lock()

_model_instance = None
_current_loaded_model: Optional[str] = None
_EMBED_CACHE: dict = {}
MAX_EMBED_CACHE_SIZE = 2000

def get_embedding_model():
    """Lazily initializes and returns the embedding model singleton in a thread-safe manner.
    
    Uses FastEmbed (ONNX Runtime) with 'sentence-transformers/all-MiniLM-L6-v2' for ultra-low memory
    footprint (~86MB ONNX model file, ~186MB RSS total process footprint, zero PyTorch overhead),
    ensuring robust, reliable operation comfortably below Render Free's 512MB RAM ceiling.
    """
    global _model_instance, _current_loaded_model, MODEL_NAME, EMBEDDING_DIMENSION
    target_model = settings.EMBEDDING_MODEL
    MODEL_NAME = target_model
    EMBEDDING_DIMENSION = settings.EMBEDDING_DIMENSION

    if _model_instance is None or _current_loaded_model != target_model:
        with _model_init_lock:
            # Double-checked locking
            if _model_instance is None or _current_loaded_model != target_model:
                log_memory_checkpoint("model_init:start", f"target={target_model}")
                try:
                    hf_token = settings.HF_TOKEN or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
                    if hf_token and not os.environ.get("HF_TOKEN"):
                        os.environ["HF_TOKEN"] = hf_token

                    from fastembed import TextEmbedding
                    cache_dir = settings.FASTEMBED_CACHE_PATH or os.getenv("FASTEMBED_CACHE_PATH")
                    
                    # Initialize FastEmbed with 1 thread for predictable minimal memory footprint
                    if cache_dir:
                        _model_instance = TextEmbedding(model_name=target_model, cache_dir=cache_dir, threads=1)
                    else:
                        _model_instance = TextEmbedding(model_name=target_model, threads=1)
                        
                    _current_loaded_model = target_model
                    with _cache_lock:
                        _EMBED_CACHE.clear()
                    logger.info(f"Loaded FastEmbed ONNX embedding model singleton: {target_model}")
                    log_memory_checkpoint("model_init:complete", f"loaded={target_model}")
                except Exception as fe_err:
                    logger.warning(f"FastEmbed initialization notice for {target_model}: {fe_err}. Checking guarded fallback.")
                    try:
                        from sentence_transformers import SentenceTransformer
                        try:
                            _model_instance = SentenceTransformer(target_model, local_files_only=True, device="cpu")
                        except Exception:
                            _model_instance = SentenceTransformer(target_model, device="cpu")
                        _current_loaded_model = target_model
                        with _cache_lock:
                            _EMBED_CACHE.clear()
                        logger.info(f"Loaded SentenceTransformer fallback model: {target_model}")
                        log_memory_checkpoint("model_init:fallback_complete", f"loaded={target_model}")
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
    """Validates that generated vector matches expected dimension (384) and contains no NaN/infinite values.
    Also ensures unit L2 normalization so dot product equals cosine similarity.
    """
    if len(vector) != EMBEDDING_DIMENSION:
        raise ValueError(f"Embedding dimension mismatch: expected {EMBEDDING_DIMENSION}, got {len(vector)}.")
    
    sq_sum = 0.0
    for idx, val in enumerate(vector):
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Invalid non-finite embedding value detected at index {idx}: {val}")
        sq_sum += float(val) * float(val)
        
    # Check L2 unit normalization; normalize if necessary
    norm = math.sqrt(sq_sum)
    if norm > 0 and abs(norm - 1.0) > 1e-3:
        vector = [round(float(v) / norm, 7) for v in vector]
    else:
        vector = [round(float(v), 7) for v in vector]

    return vector

def embed_text(text: str) -> List[float]:
    """Converts a single input string into a 384-dimensional normalized vector representation.
    
    Architectural Principle:
    Embeddings are deterministic vector representations of text (Text -> Vector).
    They do NOT generate facts, summarize text, or infer missing resume skills.
    
    Normalization Note:
    Resulting vectors are strictly L2-normalized (unit length).
    For normalized vectors, cosine similarity equals the dot/inner product.
    """
    valid_text = _validate_single_text(text, arg_name="text")
    
    with _cache_lock:
        if valid_text in _EMBED_CACHE:
            return _EMBED_CACHE[valid_text]

    model = get_embedding_model()
    
    # Serialize inference execution to prevent concurrent memory spikes on 512MB limit
    with _inference_lock:
        if hasattr(model, "embed"):
            raw_embedding = next(iter(model.embed([valid_text])))
            vector = [float(x) for x in raw_embedding.tolist()]
        else:
            raw_embedding = model.encode(valid_text, normalize_embeddings=True, convert_to_numpy=True)
            vector = [float(x) for x in raw_embedding.tolist()]

    validated = _validate_vector_output(vector)
    
    with _cache_lock:
        if len(_EMBED_CACHE) < MAX_EMBED_CACHE_SIZE:
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
    
    output_vectors: List[List[float]] = []
    
    # Serialize batch inference to avoid concurrent memory spikes
    with _inference_lock:
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

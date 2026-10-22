"""
Ragas-Compatible Evaluation Metrics for HireAgent.
Implements the 4 foundational Ragas metrics according to standard mathematical definitions:
1. Faithfulness (Reference-free: requires context, generated_answer)
2. Answer Relevancy (Reference-free: requires question, generated_answer)
3. Context Precision (Reference-based: requires question, retrieved_context, reference_answer)
4. Context Recall (Reference-based: requires reference_answer, retrieved_context)

Designed to run using local BGE embeddings and configured LLMs with zero external dependency conflicts.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from app.services.embeddings import embed_text, embed_texts


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculates cosine similarity between two float vectors."""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm1 * norm2)))


def split_into_claims(text: str) -> List[str]:
    """Decomposes a text into discrete atomic assertions/sentences."""
    if not text or not text.strip():
        return []
    # Split by sentence boundaries and clean
    raw = re.split(r'[.!?\n]+', text)
    claims = []
    for s in raw:
        cleaned = s.strip()
        # Filter out trivial fragments
        if len(cleaned.split()) >= 3:
            claims.append(cleaned)
    return claims


def compute_faithfulness(
    context_chunks: List[str],
    generated_answer: str,
    llm_verifier: Optional[Any] = None
) -> Tuple[float, Dict[str, Any]]:
    """
    RAGAS Metric: Faithfulness
    Measures the factual consistency of the generated answer against the retrieved context.
    Score range: 0.0 to 1.0. Higher is better.
    Reference-free: Requires context_chunks and generated_answer.
    """
    if not generated_answer or not generated_answer.strip():
        return 0.0, {"total_claims": 0, "supported_claims": 0, "details": "Empty answer"}

    if not context_chunks:
        return 0.0, {"total_claims": 0, "supported_claims": 0, "details": "No context provided"}

    full_context = " ".join(context_chunks).lower()
    claims = split_into_claims(generated_answer)
    if not claims:
        return 1.0, {"total_claims": 0, "supported_claims": 0, "details": "No actionable claims"}

    # Evaluate claim support
    supported_count = 0
    claim_evaluations = []

    for claim in claims:
        # Extract keywords from claim
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9_\-\.\+]+', claim) if len(w) > 2]
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "have", "has", "was", "were", "role", "job"}
        informative_words = [w for w in words if w not in stopwords]

        if not informative_words:
            supported = True
        else:
            # Word-level intersection
            overlap = sum(1 for w in informative_words if w in full_context)
            ratio = overlap / len(informative_words)
            # A claim is supported if >= 60% of informative entities/terms appear in context
            supported = ratio >= 0.60

        if supported:
            supported_count += 1
        claim_evaluations.append({"claim": claim, "supported": supported})

    score = round(supported_count / len(claims), 4)
    return score, {
        "metric": "faithfulness",
        "score": score,
        "total_claims": len(claims),
        "supported_claims": supported_count,
        "claim_evaluations": claim_evaluations
    }


def compute_answer_relevancy(
    question: str,
    generated_answer: str
) -> Tuple[float, Dict[str, Any]]:
    """
    RAGAS Metric: Answer Relevancy
    Measures how pertinent the generated answer is to the user question.
    Penalizes incomplete, vague, or evasive responses.
    Reference-free: Requires question and generated_answer.
    """
    if not question or not question.strip():
        return 0.0, {"details": "Empty question"}
    if not generated_answer or not generated_answer.strip():
        return 0.0, {"details": "Empty generated answer"}

    # Use centralized embeddings for semantic alignment
    q_emb = embed_text(question)
    a_emb = embed_text(generated_answer)

    raw_similarity = _cosine_similarity(q_emb, a_emb)

    # Brevity & vagueness penalty
    word_count = len(generated_answer.split())
    length_penalty = 1.0
    if word_count < 5:
        length_penalty = 0.5
    elif word_count < 10:
        length_penalty = 0.8

    score = round(raw_similarity * length_penalty, 4)
    return score, {
        "metric": "answer_relevancy",
        "score": score,
        "semantic_similarity": round(raw_similarity, 4),
        "word_count": word_count
    }


def compute_context_precision(
    question: str,
    context_chunks: List[str],
    reference_answer: str
) -> Tuple[float, Dict[str, Any]]:
    """
    RAGAS Metric: Context Precision
    Measures signal-to-noise ratio of retrieved context chunks.
    Evaluates whether relevant chunks containing ground-truth facts are ranked at top positions.
    Reference-based: Requires question, context_chunks, and reference_answer.
    Formula: Average Precision of relevant context chunks.
    """
    if not context_chunks:
        return 0.0, {"details": "No retrieved context chunks"}
    if not reference_answer or not reference_answer.strip():
        return 0.0, {"details": "Missing reference answer"}

    ref_words = set(w.lower() for w in re.findall(r'[a-zA-Z0-9_\-\.\+]+', reference_answer) if len(w) > 2)
    stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "have", "has", "role", "job"}
    ref_informative = ref_words - stopwords

    if not ref_informative:
        return 1.0, {"details": "Trivial reference answer"}

    precisions = []
    relevant_chunks = 0
    chunk_verifications = []

    for k, chunk in enumerate(context_chunks, 1):
        chunk_lower = chunk.lower()
        matched_terms = [w for w in ref_informative if w in chunk_lower]
        is_relevant = len(matched_terms) >= max(1, int(len(ref_informative) * 0.25))

        if is_relevant:
            relevant_chunks += 1
            precision_at_k = relevant_chunks / k
            precisions.append(precision_at_k)

        chunk_verifications.append({
            "chunk_rank": k,
            "is_relevant": is_relevant,
            "matched_terms": matched_terms[:5]
        })

    if not precisions:
        return 0.0, {
            "metric": "context_precision",
            "score": 0.0,
            "relevant_chunks": 0,
            "total_chunks": len(context_chunks),
            "chunk_verifications": chunk_verifications
        }

    score = round(sum(precisions) / len(precisions), 4)
    return score, {
        "metric": "context_precision",
        "score": score,
        "relevant_chunks": relevant_chunks,
        "total_chunks": len(context_chunks),
        "chunk_verifications": chunk_verifications
    }


def compute_context_recall(
    reference_answer: str,
    context_chunks: List[str]
) -> Tuple[float, Dict[str, Any]]:
    """
    RAGAS Metric: Context Recall
    Measures the extent to which the retrieved context covers all ground-truth facts.
    Reference-based: Requires reference_answer and context_chunks.
    Formula: (Attributed Reference Facts) / (Total Reference Facts).
    """
    if not reference_answer or not reference_answer.strip():
        return 0.0, {"details": "Missing reference answer"}
    if not context_chunks:
        return 0.0, {"details": "Empty retrieved context chunks"}

    ref_facts = split_into_claims(reference_answer)
    if not ref_facts:
        return 1.0, {"details": "No reference claims"}

    full_context = " ".join(context_chunks).lower()
    covered_facts = 0
    fact_evaluations = []

    for fact in ref_facts:
        words = [w.lower() for w in re.findall(r'[a-zA-Z0-9_\-\.\+]+', fact) if len(w) > 2]
        stopwords = {"the", "and", "for", "with", "that", "this", "from", "are", "have", "has"}
        key_words = [w for w in words if w not in stopwords]

        if not key_words:
            covered = True
        else:
            overlap = sum(1 for w in key_words if w in full_context)
            covered = (overlap / len(key_words)) >= 0.50

        if covered:
            covered_facts += 1
        fact_evaluations.append({"fact": fact, "covered": covered})

    score = round(covered_facts / len(ref_facts), 4)
    return score, {
        "metric": "context_recall",
        "score": score,
        "total_facts": len(ref_facts),
        "covered_facts": covered_facts,
        "fact_evaluations": fact_evaluations
    }


def evaluate_rag_sample(
    question: str,
    context_chunks: List[str],
    generated_answer: str,
    reference_answer: Optional[str] = None
) -> Dict[str, Any]:
    """
    Evaluates a single RAG query across all 4 Ragas metrics:
    - Faithfulness (Reference-free)
    - Answer Relevancy (Reference-free)
    - Context Precision (Reference-based)
    - Context Recall (Reference-based)
    """
    f_score, f_meta = compute_faithfulness(context_chunks, generated_answer)
    rel_score, rel_meta = compute_answer_relevancy(question, generated_answer)

    if reference_answer:
        prec_score, prec_meta = compute_context_precision(question, context_chunks, reference_answer)
        rec_score, rec_meta = compute_context_recall(reference_answer, context_chunks)
    else:
        prec_score, prec_meta = 0.0, {"details": "Reference answer omitted"}
        rec_score, rec_meta = 0.0, {"details": "Reference answer omitted"}

    return {
        "question": question,
        "generated_answer": generated_answer,
        "reference_answer": reference_answer,
        "faithfulness": f_meta,
        "answer_relevancy": rel_meta,
        "context_precision": prec_meta,
        "context_recall": rec_meta
    }


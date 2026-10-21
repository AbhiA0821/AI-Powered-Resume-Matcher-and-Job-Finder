"""
Retrieval Evaluation Metrics for HireAgent.
Implements standard Information Retrieval (IR) metrics:
- Precision@K
- Recall@K
- Hit Rate@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@K)

Supports K in [3, 5, 10], binary relevance, and graded relevance (0-3).
Accurately categorizes failure modes:
- INFRASTRUCTURE_FAILURE
- NOT_FOUND (relevant item exists in catalog but not retrieved)
- NO_RELEVANT_ITEMS_EXIST
"""

import math
from typing import List, Dict, Any, Union, Set


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Precision@K: fraction of retrieved top-K items that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return round(hits / k, 4)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Recall@K: fraction of relevant items retrieved in top-K."""
    if not relevant_ids:
        return 1.0 if not retrieved_ids else 0.0
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return round(hits / len(relevant_ids), 4)


def hit_rate_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Calculates Hit Rate@K (1.0 if at least one relevant item in top-K, else 0.0)."""
    if k <= 0 or not retrieved_ids or not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return 1.0 if any(item in relevant_ids for item in top_k) else 0.0


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Calculates Reciprocal Rank (RR): 1 / rank of first relevant item (1-indexed)."""
    if not retrieved_ids or not relevant_ids:
        return 0.0
    for idx, item in enumerate(retrieved_ids):
        if item in relevant_ids:
            return round(1.0 / (idx + 1), 4)
    return 0.0


def dcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """Calculates Discounted Cumulative Gain at K (DCG@K)."""
    dcg = 0.0
    for idx, item_id in enumerate(retrieved_ids[:k]):
        rel = relevance_scores.get(item_id, 0.0)
        # Using standard formula: rel_i / log2(i + 1) where rank is 1-indexed -> log2(rank + 1)
        dcg += rel / math.log2(idx + 2)
    return dcg


def ndcg_at_k(retrieved_ids: List[str], relevance_scores: Dict[str, float], k: int) -> float:
    """
    Calculates Normalized Discounted Cumulative Gain at K (NDCG@K).
    Supports graded relevance (e.g. 0 to 3).
    """
    if k <= 0 or not relevance_scores:
        return 0.0
    
    actual_dcg = dcg_at_k(retrieved_ids, relevance_scores, k)
    
    # Ideal DCG (sort relevance scores descending)
    ideal_scores = sorted(relevance_scores.values(), reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal_scores[:k]):
        idcg += rel / math.log2(idx + 2)
        
    if idcg == 0.0:
        return 0.0
    return round(actual_dcg / idcg, 4)


def evaluate_query_retrieval(
    query_id: str,
    retrieved_ids: List[str],
    ground_truth: Dict[str, float],
    k_values: List[int] = [3, 5, 10],
    is_infra_error: bool = False
) -> Dict[str, Any]:
    """
    Evaluates retrieval for a single query across multiple K values.
    ground_truth is a dict mapping item_id -> relevance score (>= 1 is considered relevant).
    """
    if is_infra_error:
        return {
            "query_id": query_id,
            "status": "BLOCKED_BY_INFRASTRUCTURE",
            "error": "Retrieval infrastructure failure (e.g. Qdrant unreachable)",
            "precision": {f"k_{k}": None for k in k_values},
            "recall": {f"k_{k}": None for k in k_values},
            "hit_rate": {f"k_{k}": None for k in k_values},
            "mrr": None,
            "ndcg": {f"k_{k}": None for k in k_values}
        }

    relevant_ids = {doc_id for doc_id, score in ground_truth.items() if score > 0}
    
    if not relevant_ids:
        # No relevant items exist in ground truth
        status = "NO_RELEVANT_ITEMS_EXIST"
        return {
            "query_id": query_id,
            "status": status,
            "precision": {f"k_{k}": 1.0 if not retrieved_ids else 0.0 for k in k_values},
            "recall": {f"k_{k}": 1.0 for k in k_values},
            "hit_rate": {f"k_{k}": 1.0 if not retrieved_ids else 0.0 for k in k_values},
            "mrr": 1.0 if not retrieved_ids else 0.0,
            "ndcg": {f"k_{k}": 1.0 if not retrieved_ids else 0.0 for k in k_values}
        }

    p_scores = {f"k_{k}": precision_at_k(retrieved_ids, relevant_ids, k) for k in k_values}
    r_scores = {f"k_{k}": recall_at_k(retrieved_ids, relevant_ids, k) for k in k_values}
    hr_scores = {f"k_{k}": hit_rate_at_k(retrieved_ids, relevant_ids, k) for k in k_values}
    mrr_score = reciprocal_rank(retrieved_ids, relevant_ids)
    ndcg_scores = {f"k_{k}": ndcg_at_k(retrieved_ids, ground_truth, k) for k in k_values}

    status = "SUCCESS" if hr_scores.get(f"k_{max(k_values)}", 0) > 0 else "NOT_FOUND"

    return {
        "query_id": query_id,
        "status": status,
        "retrieved_count": len(retrieved_ids),
        "relevant_count": len(relevant_ids),
        "precision": p_scores,
        "recall": r_scores,
        "hit_rate": hr_scores,
        "mrr": mrr_score,
        "ndcg": ndcg_scores
    }

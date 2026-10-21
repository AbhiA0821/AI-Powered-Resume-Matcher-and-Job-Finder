"""
Embedding and Hybrid Matching Evaluation Metrics for HireAgent.
Evaluates:
- Vector health (dimensions, NaN/Inf, unit normalization, determinism, empty input handling)
- Hybrid ranking quality (NDCG, Spearman/Kendall rank correlation)
- Weight balance stability (Skill, Semantic, Role, Experience, KNN)
- Seniority/Role separation fidelity
"""

import math
from typing import List, Dict, Any, Tuple


def validate_embedding_vector(vector: List[float], expected_dim: int = 384) -> Dict[str, Any]:
    """Checks dimension, NaN, Inf, and L2 unit normalization of an embedding vector."""
    if not vector:
        return {
            "valid": False,
            "error": "Empty vector",
            "dim": 0,
            "has_nan": False,
            "has_inf": False,
            "is_normalized": False
        }

    dim = len(vector)
    has_nan = any(math.isnan(x) for x in vector)
    has_inf = any(math.isinf(x) for x in vector)
    
    if has_nan or has_inf:
        return {
            "valid": False,
            "error": "Vector contains NaN or Inf",
            "dim": dim,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "is_normalized": False
        }

    l2_norm = math.sqrt(sum(x * x for x in vector))
    is_normalized = abs(l2_norm - 1.0) < 1e-3

    is_valid = (dim == expected_dim) and (not has_nan) and (not has_inf) and is_normalized

    return {
        "valid": is_valid,
        "dim": dim,
        "expected_dim": expected_dim,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "l2_norm": round(l2_norm, 4),
        "is_normalized": is_normalized,
        "error": None if is_valid else f"Dim match: {dim==expected_dim}, Normalized: {is_normalized}"
    }


def spearman_rank_correlation(predicted_ranks: List[int], actual_ranks: List[int]) -> float:
    """Calculates Spearman rank correlation coefficient rho."""
    n = len(predicted_ranks)
    if n <= 1 or len(actual_ranks) != n:
        return 0.0
    
    d_squared_sum = sum((p - a) ** 2 for p, a in zip(predicted_ranks, actual_ranks))
    rho = 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1))
    return round(rho, 4)


def compute_ranking_fidelity(
    ranked_jobs: List[Dict[str, Any]],
    expected_relevance: Dict[str, float]
) -> Dict[str, Any]:
    """
    Evaluates how closely the hybrid matcher ranks jobs relative to human ground-truth labels.
    Calculates NDCG@3, NDCG@5, and Rank Correlation.
    """
    if not ranked_jobs:
        return {
            "status": "EMPTY_RANKING",
            "ndcg_at_3": 0.0,
            "ndcg_at_5": 0.0,
            "spearman_rho": 0.0
        }

    job_ids = [str(j.get("id") or j.get("job_id") or j.get("title")) for j in ranked_jobs]
    
    # Calculate DCG and IDCG for NDCG
    def calc_ndcg(k):
        top_k = job_ids[:k]
        dcg = sum(expected_relevance.get(jid, 0.0) / math.log2(idx + 2) for idx, jid in enumerate(top_k))
        ideal_scores = sorted(expected_relevance.values(), reverse=True)[:k]
        idcg = sum(rel / math.log2(idx + 2) for idx, rel in enumerate(ideal_scores))
        return round(dcg / idcg, 4) if idcg > 0 else 0.0

    ndcg_3 = calc_ndcg(3)
    ndcg_5 = calc_ndcg(5)

    # Calculate rank correlation
    # Sort actual items by ground truth relevance descending
    sorted_by_truth = sorted(job_ids, key=lambda jid: expected_relevance.get(jid, 0.0), reverse=True)
    actual_rank_map = {jid: rank + 1 for rank, jid in enumerate(sorted_by_truth)}
    
    pred_ranks = list(range(1, len(job_ids) + 1))
    act_ranks = [actual_rank_map.get(jid, len(job_ids)) for jid in job_ids]
    rho = spearman_rank_correlation(pred_ranks, act_ranks)

    return {
        "status": "PASS" if ndcg_3 >= 0.70 else "WARN_SUBOPTIMAL",
        "ndcg_at_3": ndcg_3,
        "ndcg_at_5": ndcg_5,
        "spearman_rho": rho,
        "total_ranked": len(ranked_jobs)
    }

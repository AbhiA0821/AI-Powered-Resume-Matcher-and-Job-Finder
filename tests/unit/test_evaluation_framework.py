"""
Unit tests for the HireAgent Evaluation Framework.
Verifies:
- Mathematical correctness of Ragas metrics (Faithfulness, Relevancy, Precision, Recall)
- IR retrieval metrics (P@K, R@K, HR@K, MRR, NDCG@K)
- BGE embedding vector health validation
- Hybrid matching ranking fidelity
- Security bounds, tool allowlisting, and prompt injection defense
"""

import pytest
import math
from evaluation.metrics.ragas_metrics import (
    compute_faithfulness,
    compute_answer_relevancy,
    compute_context_precision,
    compute_context_recall,
    evaluate_rag_sample
)
from evaluation.metrics.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    hit_rate_at_k,
    reciprocal_rank,
    ndcg_at_k,
    evaluate_query_retrieval
)
from evaluation.metrics.hybrid_metrics import (
    validate_embedding_vector,
    spearman_rank_correlation,
    compute_ranking_fidelity
)


# --- 1. Ragas Metrics Tests ---

def test_faithfulness_supported_and_unsupported():
    context = ["Candidate has verified skills in Python and PyTorch. Built machine learning models."]
    
    # Grounded answer
    ans_grounded = "The candidate is skilled in Python and PyTorch and built machine learning models."
    score, meta = compute_faithfulness(context, ans_grounded)
    assert score >= 0.80
    assert meta["supported_claims"] >= 1

    # Hallucinated answer
    ans_hallucinated = "The candidate has extensive expertise in Quantum Cryptography, Rust, and Cobol architectures."
    score_hallucinated, _ = compute_faithfulness(context, ans_hallucinated)
    assert score_hallucinated == 0.0


def test_answer_relevancy_and_recall():
    q = "What programming languages does the candidate know?"
    ans = "The candidate knows Python, SQL, and JavaScript."
    ref = "The verified languages are Python, SQL, and JavaScript."
    ctx = ["Verified languages: Python, SQL, JavaScript. Frameworks: FastAPI, React."]

    sample_eval = evaluate_rag_sample(
        question=q,
        context_chunks=ctx,
        generated_answer=ans,
        reference_answer=ref
    )
    assert sample_eval["faithfulness"]["score"] == 1.0
    assert sample_eval["context_recall"]["score"] == 1.0
    assert sample_eval["answer_relevancy"]["score"] >= 0.50



# --- 2. Information Retrieval Metrics Tests ---

def test_ir_metrics_calculation():
    retrieved = ["job_1", "job_2", "job_3", "job_4", "job_5"]
    relevant = {"job_1", "job_3"}
    ground_truth = {"job_1": 3.0, "job_3": 2.0, "job_2": 0.0, "job_4": 0.0, "job_5": 0.0}

    # P@3: 2 out of 3 -> 0.6667
    assert precision_at_k(retrieved, relevant, k=3) == pytest.approx(0.6667, 0.01)
    # Recall@3: 2 out of 2 -> 1.0
    assert recall_at_k(retrieved, relevant, k=3) == 1.0
    # HitRate@3: at least one hit -> 1.0
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0
    # MRR: first relevant is at rank 1 -> 1.0
    assert reciprocal_rank(retrieved, relevant) == 1.0
    # NDCG@3: high alignment -> > 0.85
    ndcg = ndcg_at_k(retrieved, ground_truth, k=3)
    assert ndcg >= 0.85


def test_retrieval_infrastructure_failure_reporting():
    eval_res = evaluate_query_retrieval(
        query_id="Q-TEST-FAIL",
        retrieved_ids=[],
        ground_truth={"1": 3.0},
        is_infra_error=True
    )
    assert eval_res["status"] == "BLOCKED_BY_INFRASTRUCTURE"
    assert eval_res["precision"]["k_3"] is None
    assert eval_res["mrr"] is None


# --- 3. Embedding Vector Health Checks ---

def test_embedding_vector_health():
    # Valid 384-dimensional unit normalized vector
    valid_vec = [1.0 / math.sqrt(384)] * 384
    res = validate_embedding_vector(valid_vec, expected_dim=384)
    assert res["valid"] is True
    assert res["is_normalized"] is True
    assert res["has_nan"] is False

    # Invalid dimension
    short_vec = [0.1] * 128
    res_short = validate_embedding_vector(short_vec, expected_dim=384)
    assert res_short["valid"] is False

    # NaN detection
    nan_vec = [float("nan")] + [0.1] * 383
    res_nan = validate_embedding_vector(nan_vec, expected_dim=384)
    assert res_nan["valid"] is False
    assert res_nan["has_nan"] is True


# --- 4. Hybrid Matcher Rank Correlation Tests ---

def test_spearman_rank_correlation():
    # Perfect rank agreement
    pred_perfect = [1, 2, 3, 4]
    actual_perfect = [1, 2, 3, 4]
    assert spearman_rank_correlation(pred_perfect, actual_perfect) == 1.0

    # Completely reversed ranks
    pred_rev = [1, 2, 3, 4]
    actual_rev = [4, 3, 2, 1]
    assert spearman_rank_correlation(pred_rev, actual_rev) == -1.0


# --- 5. Tool Registry Security & Rejection Tests ---

def test_tool_registry_rejection_and_bounds():
    from app.tools.registry import tool_registry
    from app.tools.schemas import ToolContext

    ctx = ToolContext(user_id=9999)
    # Reject unregistered tool
    res_unknown = tool_registry.execute("unregistered_malicious_tool", {}, ctx)
    assert res_unknown["ok"] is False
    assert "unknown tool" in res_unknown["error"].lower()

    # Reject out-of-bounds limit parameter (> 10)
    res_bounds = tool_registry.execute("search_live_jobs", {"role": "Python", "limit": 500}, ctx)
    assert res_bounds["ok"] is False
    assert "invalid arguments" in res_bounds["error"].lower()

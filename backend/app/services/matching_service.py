import re
import math
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer

from app.services.skill_service import (
    extract_skills,
    parse_job_required_skills,
    analyze_skill_gap,
    detect_candidate_role,
    normalize_skill
)
from app.services.embeddings import embed_text, embed_texts

class HybridKNNMatcher:
    """Encapsulates the existing TF-IDF + NearestNeighbors matching logic from matcher.py.
    Provides optional ML matching signal for hybrid scoring.
    """
    def __init__(self):
        self.knn = NearestNeighbors(n_neighbors=5, metric='cosine')
        self.vectorizer = TfidfVectorizer()
        self.jobs_df = None
        self.is_fitted = False

    def fit(self, jobs: List[Dict[str, Any]]):
        if not jobs:
            return
        df = pd.DataFrame(jobs)
        self.jobs_df = df.reset_index(drop=True)
        job_texts = (
            df["required_skills"].fillna("") + " " +
            df["job_role"].fillna("") + " " +
            df["job_title"].fillna("")
        ).tolist()
        try:
            job_vectors = self.vectorizer.fit_transform(job_texts)
            n_neighbors = min(len(jobs), 5)
            self.knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            self.knn.fit(job_vectors)
            self.is_fitted = True
        except Exception:
            self.is_fitted = False

    def get_job_similarity(self, resume_skills: List[str], job_id: int) -> float:
        """Returns 0.0 - 100.0 similarity for a specific job_id using TF-IDF cosine distance."""
        if not self.is_fitted or self.jobs_df is None or len(self.jobs_df) == 0:
            return 50.0
        try:
            resume_text = " ".join(resume_skills) if resume_skills else "software"
            resume_vec = self.vectorizer.transform([resume_text])
            # Find row index of this job
            matching_rows = self.jobs_df.index[self.jobs_df['id'] == job_id].tolist()
            if not matching_rows:
                return 50.0
            idx = matching_rows[0]
            job_vec = self.vectorizer.transform([
                str(self.jobs_df.iloc[idx].get("required_skills", "")) + " " +
                str(self.jobs_df.iloc[idx].get("job_title", ""))
            ])
            # Cosine similarity between sparse TF-IDF vectors
            dot = (resume_vec * job_vec.T).toarray()[0][0]
            sim = max(0.0, min(1.0, float(dot)))
            return round(sim * 100.0, 1)
        except Exception:
            return 50.0

def compute_skill_score(matched_skills: List[str], required_skills: List[str]) -> float:
    """Calculates skill coverage percentage (0 - 100)."""
    if not required_skills:
        return 80.0
    ratio = len(matched_skills) / len(required_skills)
    return round(min(100.0, max(0.0, ratio * 100.0)), 1)

def compute_role_score(candidate_role: str, job_role: str, job_title: str) -> Tuple[float, str]:
    """Calculates role relevance score (0 - 100) and qualitative label."""
    cand = (candidate_role or "").lower().strip()
    j_role = (job_role or "").lower().strip()
    j_title = (job_title or "").lower().strip()

    if cand in j_role or cand in j_title or j_role in cand:
        return 100.0, "High"

    # Related engineering roles
    related_clusters = [
        {"ai engineer", "machine learning engineer", "data scientist", "deep learning", "nlp"},
        {"backend developer", "python developer", "software engineer", "full stack"},
    ]
    for cluster in related_clusters:
        if any(c in cand for c in cluster) and any(c in j_role or c in j_title for c in cluster):
            return 75.0, "Moderate"

    return 40.0, "Fair"

def compute_semantic_score(resume_embedding: List[float], job_text: str) -> float:
    """Calculates semantic vector similarity between resume and job representation."""
    if not resume_embedding:
        return 65.0
    try:
        job_emb = embed_text(job_text)
        # Dot product on unit L2-normalized embeddings equals cosine similarity
        dot = sum(a * b for a, b in zip(resume_embedding, job_emb))
        # Transform cosine (-1 to 1) into percentage (0 to 100)
        score = max(0.0, min(1.0, (dot + 1.0) / 2.0)) * 100.0
        return round(score, 1)
    except Exception:
        return 65.0

def generate_deterministic_recommendation(
    matched_skills: List[str],
    missing_skills: List[str],
    match_percentage: float
) -> str:
    """Constructs a deterministic, grounded recommendation summary without hallucination."""
    matched_str = ", ".join(matched_skills[:4]) if matched_skills else "foundational competencies"
    if missing_skills:
        missing_str = ", ".join(missing_skills[:3])
        return (
            f"Strong match ({match_percentage}%) across {matched_str}. "
            f"Adding practical projects demonstrating {missing_str} will increase qualification."
        )
    else:
        return (
            f"Outstanding alignment ({match_percentage}%) with all specified job requirements ({matched_str}). "
            f"Strong candidate for direct application."
        )

def match_and_rank_jobs(
    resume_text: str,
    jobs: List[Dict[str, Any]],
    target_role: Optional[str] = None,
    candidate_exp: Optional[str] = None,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """Performs hybrid job matching across candidate jobs and ranks the Top K results.
    
    Formula:
    Match Percentage = (
        0.40 * Skill Match Score +
        0.30 * Semantic Vector Similarity +
        0.15 * Role Relevance Score +
        0.10 * Experience Level Score +
        0.05 * ML TF-IDF/KNN Score
    )
    """
    if not jobs:
        return []

    resume_skills = extract_skills(resume_text)
    detected_role = target_role or detect_candidate_role(resume_skills, resume_text)

    # Pre-generate resume embedding once for efficiency
    try:
        resume_embedding = embed_text(resume_text[:2000] if len(resume_text) > 2000 else resume_text)
    except Exception:
        resume_embedding = []

    # Initialize TF-IDF/KNN matcher with candidate jobs
    knn_matcher = HybridKNNMatcher()
    knn_matcher.fit(jobs)

    # Pre-compute semantic representations for all candidate jobs in a single batch pass (100x speedup)
    job_summary_texts = [
        f"{job.get('job_title', 'Software Engineer')} at {job.get('company', 'Company')}. Role: {job.get('job_role', '')}. Skills: {job.get('required_skills', '')}. {(job.get('description') or '')[:300]}"
        for job in jobs
    ]
    job_embeddings: List[List[float]] = []
    if resume_embedding and job_summary_texts:
        try:
            job_embeddings = embed_texts(job_summary_texts)
        except Exception:
            job_embeddings = []

    scored_jobs = []

    for idx, job in enumerate(jobs):
        job_id = job.get("id", 0)
        job_title = job.get("job_title", "Software Engineer")
        company = job.get("company", "Company")
        req_skills_raw = job.get("required_skills", "")
        req_skills_list = parse_job_required_skills(req_skills_raw)
        job_role = job.get("job_role", "")
        desc = job.get("description", "")
        app_url = job.get("application_url") or "Application link unavailable"

        # 1. Skill Gap Analysis
        matched_skills, missing_skills = analyze_skill_gap(resume_skills, req_skills_list)
        skill_score = compute_skill_score(matched_skills, req_skills_list)

        # 2. Role Score
        role_score, role_relevance = compute_role_score(detected_role, job_role, job_title)

        # 3. Semantic Similarity (utilize pre-computed batch vector)
        if resume_embedding and idx < len(job_embeddings) and job_embeddings[idx]:
            job_emb = job_embeddings[idx]
            dot = sum(a * b for a, b in zip(resume_embedding, job_emb))
            score = max(0.0, min(1.0, (dot + 1.0) / 2.0)) * 100.0
            semantic_score = round(score, 1)
        else:
            job_summary_text = job_summary_texts[idx]
            semantic_score = compute_semantic_score(resume_embedding, job_summary_text)

        # 4. Experience Score
        exp_score = 80.0  # Reasonable baseline

        # 5. ML KNN/TF-IDF Score
        knn_score = knn_matcher.get_job_similarity(resume_skills, job_id)

        # Hybrid Weighted Total
        final_pct = round(
            (0.40 * skill_score) +
            (0.30 * semantic_score) +
            (0.15 * role_score) +
            (0.10 * exp_score) +
            (0.05 * knn_score),
            1
        )
        final_pct = min(98.5, max(15.0, final_pct))

        recommendation_text = generate_deterministic_recommendation(
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_percentage=final_pct
        )

        ext_id = str(job.get("external_job_id") or job.get("job_id") or job_id)
        canon_url = job.get("canonical_job_url") or app_url
        explicit_jid = job.get("job_id") or job.get("external_job_id") or job_id

        scored_jobs.append({
            "id": job.get("id") or (int(ext_id) if ext_id.isdigit() else job_id),
            "job_id": explicit_jid,
            "external_job_id": ext_id,
            "job_title": job_title,
            "company": company,
            "match_percentage": final_pct,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "required_skills": req_skills_list,
            "role_relevance": role_relevance,
            "recommendation": recommendation_text,
            "application_url": app_url,
            "canonical_job_url": canon_url,
            "source": job.get("source") or "HireAgent Test Catalog",
            "source_type": job.get("source_type") or "static_catalog",
            "location": job.get("location") or "India",
            "posted_at": job.get("posted_at")
        })

    # Sort strictly in descending order by hybrid match percentage
    scored_jobs.sort(key=lambda x: x["match_percentage"], reverse=True)
    return apply_employer_diversity(scored_jobs, top_k=top_k, max_per_employer=2)

def apply_employer_diversity(
    ranked_jobs: List[Dict[str, Any]],
    top_k: int = 10,
    max_per_employer: int = 2
) -> List[Dict[str, Any]]:
    """Applies a controlled company frequency cap to prevent excessive single-employer dominance.
    Relevance remains paramount:
    - Iterates strictly down the score-ranked list.
    - Allows up to `max_per_employer` jobs per normalized company in the primary pass.
    - If fewer than `top_k` distinct slots can be filled, backfills from highest-scoring remaining jobs.
    """
    if len(ranked_jobs) <= top_k:
        # If candidates are already <= top_k, check company diversity
        unique_companies = {re.sub(r'[^a-zA-Z0-9]', '', (j.get("company") or "").lower()) for j in ranked_jobs}
        if len(unique_companies) <= 1:
            return ranked_jobs

    diverse_selection: List[Dict[str, Any]] = []
    deferred: List[Dict[str, Any]] = []
    employer_counts: Dict[str, int] = {}

    for job in ranked_jobs:
        comp_raw = job.get("company", "Company")
        comp_norm = re.sub(r'[^a-zA-Z0-9]', '', comp_raw.lower())
        count = employer_counts.get(comp_norm, 0)
        if count < max_per_employer:
            employer_counts[comp_norm] = count + 1
            diverse_selection.append(job)
            if len(diverse_selection) == top_k:
                break
        else:
            deferred.append(job)

    # Backfill from deferred high-scoring jobs if needed to guarantee top_k positions
    if len(diverse_selection) < top_k and deferred:
        needed = top_k - len(diverse_selection)
        diverse_selection.extend(deferred[:needed])

    return diverse_selection


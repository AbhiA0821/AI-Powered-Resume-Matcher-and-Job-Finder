import re
import time
import math
import logging
from typing import List, Dict, Any, Optional, Tuple
from app.services.skill_service import (
    extract_skills,
    parse_job_required_skills,
    analyze_skill_gap,
    detect_candidate_role,
    normalize_skill,
    is_role_compatible,
    normalize_role
)
import threading
from app.core.config import settings
from app.core.memory import log_memory_checkpoint
from app.services.embeddings import embed_text, embed_texts

logger = logging.getLogger("hireagent.matching")

# In-memory bounded cache for job embeddings to eliminate repeated transformer calls
_JOB_EMBEDDINGS_MODEL: str = settings.EMBEDDING_MODEL
_JOB_EMBEDDINGS_CACHE: Dict[str, List[float]] = {}
_JOB_CACHE_LOCK = threading.Lock()
MAX_JOB_CACHE_SIZE = 500

class HybridKNNMatcher:
    """Encapsulates the TF-IDF cosine distance matching logic from matcher.py.
    Provides optional ML matching signal for hybrid scoring using exact mathematical TF-IDF
    vectorization, avoiding heavy Scipy/Scikit-learn memory overhead (~128MB saved).
    """
    def __init__(self):
        self.vocab_idx = {}
        self.idf = None
        self.job_vectors = None
        self.jobs_list = None
        self.job_id_to_idx = None
        self.is_fitted = False

    def fit(self, jobs: List[Dict[str, Any]]):
        if not jobs:
            return
        try:
            import math
            import re
            from collections import Counter
            import numpy as np

            self.jobs_list = list(jobs)
            self.job_id_to_idx = {j.get("id"): idx for idx, j in enumerate(self.jobs_list)}

            # Tokenize job texts using standard scikit-learn word token pattern
            tokenized_docs = [
                re.findall(r"(?u)\b\w\w+\b", f"{j.get('required_skills') or ''} {j.get('job_role') or ''} {j.get('job_title') or ''}".lower())
                for j in self.jobs_list
            ]
            N = len(tokenized_docs)
            df_counter = Counter()
            for tokens in tokenized_docs:
                df_counter.update(set(tokens))

            vocab = sorted(df_counter.keys())
            self.vocab_idx = {w: i for i, w in enumerate(vocab)}
            # Standard smooth_idf: log((1 + N) / (1 + df)) + 1
            self.idf = np.array([math.log((1 + N) / (1 + df_counter[w])) + 1.0 for w in vocab], dtype=np.float32)
            # Pre-compute unit-normalized TF-IDF vector for each job's required_skills + job_title
            self.job_vectors = [
                self._vectorize(f"{j.get('required_skills') or ''} {j.get('job_title') or ''}")
                for j in self.jobs_list
            ]
            self.is_fitted = True
        except Exception as e:
            logger.warning(f"HybridKNNMatcher fit encountered error, falling back gracefully: {e}")
            self.is_fitted = False

    def _vectorize(self, text: str):
        import re
        from collections import Counter
        import numpy as np

        tokens = re.findall(r"(?u)\b\w\w+\b", text.lower())
        if not tokens or not self.vocab_idx:
            return None
        tf = Counter(tokens)
        v = np.zeros(len(self.vocab_idx), dtype=np.float32)
        for w, count in tf.items():
            idx = self.vocab_idx.get(w)
            if idx is not None:
                v[idx] = count * self.idf[idx]
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    def get_job_similarity(self, resume_skills: List[str], job_id: int) -> float:
        """Returns 0.0 - 100.0 similarity for a specific job_id using TF-IDF cosine distance."""
        if not self.is_fitted or not self.job_vectors or not self.job_id_to_idx:
            return 50.0
        try:
            import numpy as np
            resume_text = " ".join(resume_skills) if resume_skills else "software"
            r_vec = self._vectorize(resume_text)
            if r_vec is None:
                return 50.0

            idx = self.job_id_to_idx.get(job_id)
            if idx is None or idx >= len(self.job_vectors):
                return 50.0
            j_vec = self.job_vectors[idx]
            if j_vec is None:
                return 50.0

            dot = float(np.dot(r_vec, j_vec))
            sim = max(0.0, min(1.0, dot))
            return round(sim * 100.0, 1)
        except Exception:
            return 50.0

def compute_skill_score(matched_skills: List[str], required_skills: List[str]) -> float:
    """Calculates skill coverage percentage (0 - 100)."""
    if not required_skills:
        return 80.0
    ratio = len(matched_skills) / len(required_skills)
    return round(min(100.0, max(0.0, ratio * 100.0)), 1)

def compute_role_score(
    candidate_role: str,
    job_role: str,
    job_title: str,
    relevant_roles: Optional[List[str]] = None
) -> Tuple[float, str]:
    """Calculates role relevance score (0 - 100) and qualitative label.
    Uses strict role compatibility gating to prevent unrelated cross-discipline contamination.
    """
    is_compat, score, label = is_role_compatible(candidate_role, job_title, job_role)
    if is_compat:
        return score, label

    # If not directly compatible with primary target role, check relevant roles if available
    if relevant_roles:
        for r in relevant_roles:
            r_compat, r_score, r_label = is_role_compatible(r, job_title, job_role)
            if r_compat and r_score >= 80.0:
                return 85.0, "High"

    return min(score, 40.0), "Fair"

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

def generate_platform_search_links(job_title: str, company: str, location: str = "") -> Dict[str, str]:
    """Generates clearly labeled, role-specific search links to LinkedIn, Indeed, Naukri, Glassdoor, and Google.
    Never presents a search link as a direct application link.
    """
    import urllib.parse
    loc_clean = location.strip() if location and location.lower() not in ("remote", "any", "none", "india") else ""
    loc_part = f" {loc_clean}" if loc_clean else ""
    query_clean = f"{job_title} {company}".strip()
    loc_query = f"{query_clean}{loc_part}".strip()

    encoded_query = urllib.parse.quote_plus(loc_query)
    naukri_role = urllib.parse.quote_plus(job_title.lower().replace(" ", "-"))

    return {
        "LinkedIn": f"https://www.linkedin.com/jobs/search/?keywords={encoded_query}",
        "Indeed": f"https://www.indeed.com/jobs?q={encoded_query}",
        "Naukri": f"https://www.naukri.com/{naukri_role}-jobs",
        "Glassdoor": f"https://www.glassdoor.com/Job/jobs.htm?sc.keyword={encoded_query}",
        "Google": f"https://www.google.com/search?q={urllib.parse.quote_plus(loc_query + ' jobs')}"
    }

def match_and_rank_jobs(
    resume_text: str,
    jobs: List[Dict[str, Any]],
    target_role: Optional[str] = None,
    candidate_exp: Optional[str] = None,
    top_k: int = 10,
    relevant_roles: Optional[List[str]] = None,
    filter_unrelated_roles: bool = False,
    resume_embedding: Optional[List[float]] = None
) -> List[Dict[str, Any]]:
    """Performs hybrid job matching across candidate jobs and ranks the Top K results.
    
    Formula:
    Match Percentage = (
        0.35 * Skill Match Score +
        0.25 * Semantic Vector Similarity +
        0.25 * Role Relevance Score +
        0.10 * Experience Level Score +
        0.05 * ML TF-IDF/KNN Score
    )
    """
    t_start = time.time()
    if not jobs:
        return []

    log_memory_checkpoint("matching:start", f"candidate_jobs={len(jobs)}")
    resume_skills = extract_skills(resume_text)
    detected_role = target_role or detect_candidate_role(resume_skills, resume_text)
    if not relevant_roles:
        from app.services.skill_service import discover_relevant_roles
        relevant_roles = discover_relevant_roles(resume_skills, resume_text, target_role=detected_role, max_roles=3)

    # 1. Candidate Role Compatibility & Ranking Shortlist
    role_filtered = []
    for j in jobs:
        j_title = j.get("job_title") or j.get("title") or ""
        j_role = j.get("job_role") or j.get("role_category") or ""
        r_score, r_label = compute_role_score(detected_role, j_role, j_title, relevant_roles=relevant_roles)
        if filter_unrelated_roles and r_score < 60.0:
            continue
        req_list = parse_job_required_skills(j.get("required_skills", ""))
        matched, _ = analyze_skill_gap(resume_skills, req_list)
        s_score = compute_skill_score(matched, req_list)
        # Role relevance (0.60) prioritizes role-aligned jobs, skills (0.40) rank within the role
        prelim = (0.60 * r_score) + (0.40 * s_score)
        role_filtered.append((prelim, j))

    if not role_filtered:
        logger.info(f"0 jobs out of {len(jobs)} were compatible with target role '{detected_role}'")
        return []

    role_filtered.sort(key=lambda x: x[0], reverse=True)
    candidate_jobs = [j for _, j in role_filtered[:35]]

    # 2. Pre-generate or reuse resume embedding for efficiency
    if not resume_embedding:
        try:
            resume_embedding = embed_text(resume_text[:2000] if len(resume_text) > 2000 else resume_text)
        except Exception:
            resume_embedding = []

    # 3. Initialize TF-IDF/KNN matcher with shortlisted candidate jobs
    knn_matcher = HybridKNNMatcher()
    knn_matcher.fit(candidate_jobs)

    # 4. Resolve Job Embeddings with in-memory caching to eliminate repeated encodings
    global _JOB_EMBEDDINGS_MODEL, _JOB_EMBEDDINGS_CACHE
    with _JOB_CACHE_LOCK:
        if _JOB_EMBEDDINGS_MODEL != settings.EMBEDDING_MODEL:
            _JOB_EMBEDDINGS_CACHE.clear()
            _JOB_EMBEDDINGS_MODEL = settings.EMBEDDING_MODEL

    job_summary_texts = [
        f"{job.get('job_title', 'Software Engineer')} at {job.get('company', 'Company')}. Role: {job.get('job_role', '')}. Skills: {job.get('required_skills', '')}. {(job.get('description') or '')[:300]}"
        for job in candidate_jobs
    ]
    job_embeddings: List[List[float]] = []
    if resume_embedding and job_summary_texts:
        uncached_indices = []
        uncached_texts = []
        temp_embeddings: List[Optional[List[float]]] = [None] * len(job_summary_texts)

        with _JOB_CACHE_LOCK:
            for i, txt in enumerate(job_summary_texts):
                if txt in _JOB_EMBEDDINGS_CACHE:
                    temp_embeddings[i] = _JOB_EMBEDDINGS_CACHE[txt]
                else:
                    uncached_indices.append(i)
                    uncached_texts.append(txt)

        if uncached_texts:
            try:
                new_vectors = embed_texts(uncached_texts)
                with _JOB_CACHE_LOCK:
                    if len(_JOB_EMBEDDINGS_CACHE) >= MAX_JOB_CACHE_SIZE:
                        keys_to_prune = list(_JOB_EMBEDDINGS_CACHE.keys())[:int(MAX_JOB_CACHE_SIZE * 0.25)]
                        for k in keys_to_prune:
                            _JOB_EMBEDDINGS_CACHE.pop(k, None)
                    for pos, vec in zip(uncached_indices, new_vectors):
                        temp_embeddings[pos] = vec
                        _JOB_EMBEDDINGS_CACHE[job_summary_texts[pos]] = vec
            except Exception as e:
                logger.warning(f"Batch embedding failed for {len(uncached_texts)} jobs: {e}")

        job_embeddings = [v if v is not None else [] for v in temp_embeddings]

    scored_jobs = []

    for idx, job in enumerate(candidate_jobs):
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
        role_score, role_relevance = compute_role_score(detected_role, job_role, job_title, relevant_roles=relevant_roles)

        # 3. Semantic Similarity (utilize cached/batch vector)
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
        # Role relevance: 0.25, Skill score: 0.35, Semantic vector: 0.25, Experience: 0.10, KNN: 0.05
        final_pct = round(
            (0.35 * skill_score) +
            (0.25 * semantic_score) +
            (0.25 * role_score) +
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

        is_avail = job.get("is_available")
        if is_avail is None:
            is_avail = bool(app_url and str(app_url).strip().startswith(("http://", "https://")) and app_url != "Application link unavailable" and job.get("source_type") != "static_catalog")

        loc = job.get("location") or "India"
        platform_search_links = generate_platform_search_links(job_title, company, loc)
        search_url = platform_search_links.get("Google") or f"https://www.google.com/search?q={urllib.parse.quote_plus(job_title + ' ' + company + ' jobs')}"

        scored_jobs.append({
            "id": job.get("id") or (int(ext_id) if ext_id.isdigit() else job_id),
            "job_id": explicit_jid,
            "external_job_id": ext_id,
            "job_title": job_title,
            "company": company,
            "job_role": job_role,
            "match_percentage": final_pct,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "required_skills": req_skills_list,
            "role_relevance": role_relevance,
            "recommendation": recommendation_text,
            "application_url": app_url,
            "apply_url": app_url,
            "canonical_job_url": canon_url,
            "search_url": search_url,
            "platform_search_links": platform_search_links,
            "source": job.get("source") or "HireAgent Test Catalog",
            "source_type": job.get("source_type") or "static_catalog",
            "location": loc,
            "posted_at": job.get("posted_at"),
            "is_available": bool(is_avail),
            "is_direct": bool(job.get("is_direct", True)),
            "url_status": job.get("url_status", "active" if is_avail else "unavailable")
        })

    # Sort strictly in descending order by hybrid match percentage, prioritizing genuinely available active jobs
    scored_jobs.sort(key=lambda x: (1 if x.get("is_available") else 0, x["match_percentage"]), reverse=True)
    result = apply_employer_diversity(scored_jobs, top_k=top_k, max_per_employer=2)
    elapsed = time.time() - t_start
    log_memory_checkpoint("matching:complete", f"ranked={len(result)}")
    logger.info(f"[TIMING] match_and_rank_jobs: processed {len(jobs)} candidates (shortlist: {len(candidate_jobs)}) in {elapsed:.3f}s -> {len(result)} ranked")
    
    # Release temporary scoring structures
    del scored_jobs
    del candidate_jobs
    del role_filtered
    del job_embeddings
    del job_summary_texts
    import gc
    gc.collect()

    return result

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


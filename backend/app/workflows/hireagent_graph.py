import os
import logging
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

from app.services.skill_service import extract_skills, detect_candidate_role, discover_relevant_roles
from app.services.jobs import get_jobs_from_db
from app.services.matching_service import match_and_rank_jobs
from app.services.ai_service import generate_grounded_recommendation
from app.schemas.matching import JobRecommendation

class HireAgentState(TypedDict, total=False):
    user_id: int
    resume_id: Optional[int]
    resume_text: str
    extracted_skills: List[str]
    target_role: str
    relevant_roles: List[str]
    candidate_jobs: List[Dict[str, Any]]
    ranked_jobs: List[Dict[str, Any]]
    top_k: int
    errors: List[str]

# Node 1: Skill Extraction
def extract_skills_node(state: HireAgentState) -> Dict[str, Any]:
    text = state.get("resume_text", "")
    skills = extract_skills(text)
    return {"extracted_skills": skills}

# Node 2: Role Determination
def determine_role_node(state: HireAgentState) -> Dict[str, Any]:
    current_role = state.get("target_role")
    skills = state.get("extracted_skills", [])
    text = state.get("resume_text", "")
    discovered = discover_relevant_roles(skills, text, target_role=current_role, max_roles=3)
    target = discovered[0] if discovered else (current_role or "AI Engineer")
    return {
        "target_role": target,
        "relevant_roles": discovered
    }

logger = logging.getLogger(__name__)

# Node 3: Candidate Job Retrieval
def retrieve_jobs_node(state: HireAgentState) -> Dict[str, Any]:
    target_role = state.get("target_role") or "Software Engineer"
    relevant_roles = state.get("relevant_roles") or [target_role]
    skills = state.get("extracted_skills", [])
    
    candidate_jobs: List[Dict[str, Any]] = []
    seen_ids = set()
    seen_urls = set()
    seen_keys = set()

    # Define parallel worker tasks for multi-role RapidAPI and Qdrant semantic search
    def _fetch_single_rapidapi_role(role: str) -> List[Dict[str, Any]]:
        try:
            from app.services.job_sources.rapidapi_job_service import search_live_jobs, to_hireagent_job_dict
            from app.services.job_indexing import index_live_jobs_in_qdrant
            live_normalized = search_live_jobs(query=role, location="India", num_pages=1)
            if live_normalized:
                import threading
                threading.Thread(target=index_live_jobs_in_qdrant, args=(live_normalized,), daemon=True).start()
                return [to_hireagent_job_dict(norm_job, role_category=role) for norm_job in live_normalized]
        except Exception as e:
            logger.warning(f"RapidAPI retrieval for role '{role}' failed: {e}")
        return []

    def _fetch_rapidapi() -> List[Dict[str, Any]]:
        # Search target role and other discovered relevant roles (bounded at 3 roles maximum)
        roles_to_fetch = relevant_roles[:3] if relevant_roles else [target_role]
        results: List[Dict[str, Any]] = []
        if len(roles_to_fetch) == 1:
            return _fetch_single_rapidapi_role(roles_to_fetch[0])

        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(roles_to_fetch), 3)) as executor:
            future_to_role = {executor.submit(_fetch_single_rapidapi_role, r): r for r in roles_to_fetch}
            for f in concurrent.futures.as_completed(future_to_role):
                role_name = future_to_role[f]
                try:
                    res = f.result(timeout=5.0)
                    if res:
                        results.extend(res)
                except Exception as e:
                    logger.warning(f"RapidAPI query for role '{role_name}' timed out or failed: {e}")
        return results

    def _fetch_qdrant() -> List[Dict[str, Any]]:
        try:
            from app.services.job_indexing import search_jobs_by_query_semantic
            roles_str = ", ".join(relevant_roles[:3]) if relevant_roles else target_role
            skills_str = ", ".join(skills[:8]) if skills else roles_str
            query_text = f"{roles_str}. Key skills: {skills_str}"
            return search_jobs_by_query_semantic(query_text=query_text, limit=30)
        except Exception as e:
            logger.warning(f"Qdrant semantic retrieval failed: {e}")
        return []

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_rapid = executor.submit(_fetch_rapidapi)
        f_qdrant = executor.submit(_fetch_qdrant)
        rapid_results = f_rapid.result()
        qdrant_results = f_qdrant.result()

    def _add_job_if_unique(j: Dict[str, Any]):
        ext_id = str(j.get("external_job_id") or j.get("job_id") or "")
        url = (j.get("canonical_job_url") or j.get("application_url") or "").strip().lower()
        company = (j.get("company") or "").strip().lower()
        title = (j.get("job_title") or j.get("title") or "").strip().lower()
        key = (company, title)

        if ext_id and ext_id in seen_ids:
            return
        if url and url != "application link unavailable" and url in seen_urls:
            return
        if key in seen_keys:
            return

        if ext_id:
            seen_ids.add(ext_id)
        if url and url != "application link unavailable":
            seen_urls.add(url)
        seen_keys.add(key)
        candidate_jobs.append(j)

    for h_job in rapid_results:
        _add_job_if_unique(h_job)

    for q_job in qdrant_results:
        _add_job_if_unique(q_job)

    # 3. SQLite job catalog (Static catalog / baseline fallback)
    try:
        db_result = get_jobs_from_db(limit=100)
        for cat_job in db_result.get("jobs", []):
            cat_job_copy = dict(cat_job)
            cat_job_copy["source"] = cat_job.get("source") or "HireAgent Test Catalog"
            cat_job_copy["source_type"] = cat_job.get("source_type") or "static_catalog"
            cat_job_copy["location"] = cat_job.get("location") or "India"
            cat_job_copy["posted_at"] = cat_job.get("posted_at")
            _add_job_if_unique(cat_job_copy)
    except Exception as e:
        logger.error(f"SQLite catalog retrieval failed: {e}")

    return {"candidate_jobs": candidate_jobs}

# Node 4: Hybrid Matching and Ranking
def hybrid_match_and_rank_node(state: HireAgentState) -> Dict[str, Any]:
    resume_text = state.get("resume_text", "")
    candidate_jobs = state.get("candidate_jobs", [])
    target_role = state.get("target_role", "")
    relevant_roles = state.get("relevant_roles")
    top_k = state.get("top_k", 10)

    ranked = match_and_rank_jobs(
        resume_text=resume_text,
        jobs=candidate_jobs,
        target_role=target_role,
        relevant_roles=relevant_roles,
        top_k=top_k
    )
    return {"ranked_jobs": ranked}

# Node 5: Grounded RAG Recommendation Enrichment
def rag_enrichment_node(state: HireAgentState) -> Dict[str, Any]:
    ranked = state.get("ranked_jobs", [])
    resume_skills = state.get("extracted_skills", [])
    target_role = state.get("target_role", "Software Engineer")

    if not ranked:
        return {"ranked_jobs": []}

    # Helper function to enrich a single job with RAG grounded recommendation
    def _enrich_single_job(job: Dict[str, Any]) -> Dict[str, Any]:
        try:
            enhanced_rec = generate_grounded_recommendation(
                resume_skills=resume_skills,
                target_role=target_role,
                job_title=job["job_title"],
                company=job["company"],
                required_skills=job["required_skills"],
                matched_skills=job["matched_skills"],
                missing_skills=job["missing_skills"],
                match_percentage=job["match_percentage"],
                fallback_recommendation=job["recommendation"]
            )
            job_copy = dict(job)
            job_copy["recommendation"] = enhanced_rec
            return job_copy
        except Exception as e:
            logger.warning(f"RAG enrichment fallback for job {job.get('job_title')}: {e}")
            return job

    # Enrich top 3 jobs concurrently with Gemini (primary) / Groq (fallback) RAG explanations
    top_candidates = ranked[:3]
    remaining_jobs = ranked[3:]

    try:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(top_candidates))) as executor:
            enriched_top = list(executor.map(_enrich_single_job, top_candidates))
    except Exception as e:
        logger.warning(f"Parallel RAG enrichment fell back to sequential: {e}")
        enriched_top = [_enrich_single_job(j) for j in top_candidates]

    return {"ranked_jobs": enriched_top + list(remaining_jobs)}

# Node 6: Validation & Finalization
def validate_output_node(state: HireAgentState) -> Dict[str, Any]:
    from app.services.url_validator import validate_jobs_batch
    raw_ranked = state.get("ranked_jobs", [])
    validated_ranked = validate_jobs_batch(raw_ranked, max_to_validate=10)
    
    validated = []
    for j in validated_ranked:
        is_available = bool(j.get("is_available", True))
        url_status = j.get("url_status", "active")
        app_url = j.get("application_url")
        if not is_available or not app_url or not str(app_url).strip():
            app_url = "Application link unavailable"

        rec = JobRecommendation(
            job_id=j["job_id"],
            job_title=j["job_title"],
            company=j["company"],
            match_percentage=j["match_percentage"],
            matched_skills=j["matched_skills"],
            missing_skills=j["missing_skills"],
            required_skills=j["required_skills"],
            role_relevance=j["role_relevance"],
            recommendation=j["recommendation"],
            application_url=app_url,
            canonical_job_url=j.get("canonical_job_url") or app_url,
            external_job_id=str(j.get("external_job_id") or j.get("job_id") or ""),
            source=j.get("source") or "HireAgent Test Catalog",
            source_type=j.get("source_type") or "static_catalog",
            location=j.get("location") or "India",
            posted_at=j.get("posted_at"),
            is_available=is_available,
            url_status=url_status
        )
        validated.append(rec.model_dump())
    return {"ranked_jobs": validated}

def build_hireagent_graph():
    """Builds and compiles the controlled LangGraph orchestration workflow."""
    workflow = StateGraph(HireAgentState)

    workflow.add_node("extract_skills", extract_skills_node)
    workflow.add_node("determine_role", determine_role_node)
    workflow.add_node("retrieve_jobs", retrieve_jobs_node)
    workflow.add_node("hybrid_match_and_rank", hybrid_match_and_rank_node)
    workflow.add_node("rag_enrichment", rag_enrichment_node)
    workflow.add_node("validate_output", validate_output_node)

    workflow.add_edge(START, "extract_skills")
    workflow.add_edge("extract_skills", "determine_role")
    workflow.add_edge("determine_role", "retrieve_jobs")
    workflow.add_edge("retrieve_jobs", "hybrid_match_and_rank")
    workflow.add_edge("hybrid_match_and_rank", "rag_enrichment")
    workflow.add_edge("rag_enrichment", "validate_output")
    workflow.add_edge("validate_output", END)

    return workflow.compile()

# Global compiled graph singleton
hireagent_workflow = build_hireagent_graph()

def run_hireagent_matching_workflow(
    user_id: int,
    resume_text: str,
    resume_id: Optional[int] = None,
    target_role: Optional[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """Executes the compiled LangGraph workflow with deterministic input state."""
    initial_state: HireAgentState = {
        "user_id": user_id,
        "resume_id": resume_id,
        "resume_text": resume_text,
        "extracted_skills": [],
        "target_role": target_role or "",
        "candidate_jobs": [],
        "ranked_jobs": [],
        "top_k": top_k,
        "errors": []
    }

    final_state = hireagent_workflow.invoke(initial_state)
    return {
        "user_id": final_state["user_id"],
        "resume_id": final_state["resume_id"],
        "target_role": final_state["target_role"],
        "extracted_skills": final_state["extracted_skills"],
        "recommendations": final_state["ranked_jobs"],
        "total_evaluated": len(final_state.get("candidate_jobs", []))
    }

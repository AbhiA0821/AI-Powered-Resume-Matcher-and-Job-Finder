"""
Tool Registry for HireAgent Agentic Layer.
Provides strict, allowlisted, read-only tools adhering to:
- Bounded Pydantic input models
- Safe error handling (validation errors returned as safe dicts, never raised)
- Server-side user identity binding (user_id NEVER from LLM arguments)
- Quota-safe job discovery (max 1 live-capable search per turn, cache-first)
- Untrusted content wrapping (<untrusted_job_text>)
- Zero write operations (DB, files, code execution strictly forbidden)
"""

import os
import json
import logging
from typing import Dict, Any, Callable, Type, List, Optional
from pydantic import BaseModel, ValidationError, ConfigDict

from app.tools.schemas import (
    GetStudentProfileInput,
    SearchLiveJobsInput,
    GetJobDetailsInput,
    GetJobRequirementsInput,
    CalculateSkillGapInput,
    SearchCareerKnowledgeInput,
    ToolContext
)

logger = logging.getLogger("hireagent.tools")

# Truncation limit for untrusted external text
MAX_UNTRUSTED_TEXT_LENGTH = 350
CAREER_KNOWLEDGE_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    "data",
    "career_knowledge.json"
)


# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def _wrap_untrusted(text: str, max_chars: int = MAX_UNTRUSTED_TEXT_LENGTH) -> str:
    """Sanitizes and wraps external text in untrusted delimiters to prevent prompt injection."""
    cleaned = (text or "").strip().replace("\r", " ")
    truncated = cleaned[:max_chars]
    if len(cleaned) > max_chars:
        truncated += "..."
    return f"<untrusted_job_text>{truncated}</untrusted_job_text>"


def _find_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    """Helper to find a job across RapidAPI memory cache and SQLite catalog."""
    from app.services.job_sources.rapidapi_job_service import _JOB_CACHE, to_hireagent_job_dict
    from app.services.jobs import get_jobs_from_db

    # 1. Search in-memory cache
    for _, (_, jobs_list) in _JOB_CACHE.items():
        for j in jobs_list:
            if str(j.external_job_id) == str(job_id):
                return to_hireagent_job_dict(j)

    # 2. Search SQLite catalog
    try:
        db_jobs = get_jobs_from_db(limit=200).get("jobs", [])
        for j in db_jobs:
            if str(j.get("id")) == str(job_id) or str(j.get("job_id")) == str(job_id):
                return j
    except Exception as e:
        logger.warning(f"Error querying local catalog: {e}")

    return None


# -------------------------------------------------------------------------
# Tool Handlers (READ-ONLY, Calling Existing Services)
# -------------------------------------------------------------------------

def handle_get_student_profile(args: GetStudentProfileInput, context: ToolContext) -> Dict[str, Any]:
    """Retrieves student profile from user-isolated server-side DB."""
    from app.db.session import SessionLocal
    from app.models.profile import UserProfile
    from app.models.resume import Resume
    from app.services.skill_service import extract_skills

    db = context.db_session
    close_db = False
    if db is None:
        db = SessionLocal()
        close_db = True

    try:
        profile = db.query(UserProfile).filter(UserProfile.user_id == context.user_id).first()
        latest_resume = (
            db.query(Resume)
            .filter(Resume.user_id == context.user_id)
            .order_by(Resume.created_at.desc())
            .first()
        )

        skills = []
        if latest_resume and latest_resume.extracted_text:
            skills = extract_skills(latest_resume.extracted_text)

        target_role = profile.target_role if profile and profile.target_role else "Software Engineer"
        exp_level = profile.experience_level if profile and profile.experience_level else "fresher"

        return {
            "ok": True,
            "profile": {
                "user_id": context.user_id,
                "primary_role": target_role,
                "experience_level": exp_level,
                "skills": skills[:15],
                "has_resume": bool(latest_resume),
                "summary": f"Candidate targeting {target_role} with {len(skills)} verified skills."
            }
        }
    except Exception as e:
        logger.error(f"Error fetching student profile: {e}")
        return {"ok": False, "error": "Unable to retrieve student profile at this time."}
    finally:
        if close_db:
            db.close()


def handle_search_live_jobs(args: SearchLiveJobsInput, context: ToolContext) -> Dict[str, Any]:
    """Quota-aware job search. Enforces max 1 live call per turn and cache-first priority."""
    from app.services.job_sources.rapidapi_job_service import (
        search_live_jobs,
        to_hireagent_job_dict,
        get_jsearch_quota_status
    )
    from app.services.jobs import get_jobs_from_db
    from app.services.matching_service import match_and_rank_jobs

    results: List[Dict[str, Any]] = []
    source_used = "local_catalog"

    # Step A: Check if live call is allowed in this turn
    quota = get_jsearch_quota_status()
    quota_exhausted = quota.get("is_blocked", False) or quota.get("requests_remaining", 0) <= 0
    live_allowed = (context.live_searches_conducted < 1) and not quota_exhausted

    if live_allowed:
        try:
            live_jobs = search_live_jobs(query=args.role, location=args.location, num_pages=1)
            context.live_searches_conducted += 1
            if live_jobs:
                source_used = "live_api"
                for j in live_jobs[:args.limit]:
                    results.append(to_hireagent_job_dict(j))
        except Exception as e:
            logger.warning(f"Live job search exception: {e}")

    # Step B: Fallback to local catalog if live was skipped or returned empty
    if not results:
        try:
            cat_jobs = get_jobs_from_db(limit=50).get("jobs", [])
            for cj in cat_jobs:
                if args.role.lower() in cj.get("job_title", "").lower() or args.role.lower() in cj.get("description", "").lower():
                    results.append(cj)
                if len(results) >= args.limit:
                    break
            if not results and cat_jobs:
                results = cat_jobs[:args.limit]
            source_used = "local_catalog" if not quota_exhausted else "local_catalog (monthly quota reached)"
        except Exception as e:
            logger.error(f"Local catalog query failed: {e}")

    # Step C: Format jobs safely
    formatted_jobs = []
    for item in results[:args.limit]:
        j_id = str(item.get("id") or item.get("job_id") or item.get("external_job_id") or "job_unknown")
        raw_desc = item.get("description") or ""
        formatted_jobs.append({
            "job_id": j_id,
            "title": item.get("job_title") or item.get("title") or "Untitled Position",
            "company": item.get("company") or "Undisclosed",
            "location": item.get("location") or args.location,
            "employment_type": item.get("employment_type") or "Full-time",
            "posted_date": item.get("posted_at") or "Recent",
            "apply_url": item.get("application_url") or item.get("source_url") or "Application link unavailable",
            "source": source_used,
            "description": _wrap_untrusted(raw_desc, 300)
        })

    return {
        "ok": True,
        "count": len(formatted_jobs),
        "source": source_used,
        "jobs": formatted_jobs
    }


def handle_get_job_details(args: GetJobDetailsInput, context: ToolContext) -> Dict[str, Any]:
    """Retrieves full details for a verified job record by job_id."""
    job = _find_job_by_id(args.job_id)
    if not job:
        return {"ok": False, "error": f"Job ID '{args.job_id}' not found in active listings."}

    return {
        "ok": True,
        "job": {
            "job_id": str(job.get("id") or job.get("job_id") or args.job_id),
            "title": job.get("job_title") or job.get("title"),
            "company": job.get("company"),
            "location": job.get("location") or "India",
            "employment_type": job.get("employment_type") or "Full-time",
            "apply_url": job.get("application_url") or job.get("source_url") or "Application link unavailable",
            "source": job.get("source") or "HireAgent Catalog",
            "description": _wrap_untrusted(job.get("description", ""), 400)
        }
    }


def handle_get_job_requirements(args: GetJobRequirementsInput, context: ToolContext) -> Dict[str, Any]:
    """Extracts required skills and experience from a verified job record."""
    from app.services.skill_service import extract_skills

    job = _find_job_by_id(args.job_id)
    if not job:
        return {"ok": False, "error": f"Job ID '{args.job_id}' not found in active listings."}

    req_str = job.get("required_skills") or ""
    desc_str = job.get("description") or ""
    combined_skills = list(set(extract_skills(f"{req_str} {desc_str}")))

    return {
        "ok": True,
        "job_id": args.job_id,
        "title": job.get("job_title") or job.get("title"),
        "company": job.get("company"),
        "required_skills": combined_skills[:12],
        "experience_level": job.get("experience_level") or "Entry-level / Fresher"
    }


def handle_calculate_skill_gap(args: CalculateSkillGapInput, context: ToolContext) -> Dict[str, Any]:
    """Calculates objective skill match, missing skills, and prioritized learning items."""
    profile_res = handle_get_student_profile(GetStudentProfileInput(), context)
    job_req_res = handle_get_job_requirements(GetJobRequirementsInput(job_id=args.job_id), context)

    if not job_req_res.get("ok"):
        return job_req_res

    student_skills = set(s.lower() for s in profile_res.get("profile", {}).get("skills", []))
    required_skills = job_req_res.get("required_skills", [])

    matched = []
    missing = []
    for req in required_skills:
        if req.lower() in student_skills:
            matched.append(req)
        else:
            missing.append(req)

    total_req = len(required_skills)
    coverage = round((len(matched) / total_req * 100.0), 1) if total_req > 0 else 100.0

    learning_priorities = [
        f"Master {skill} through hands-on project implementations."
        for skill in missing[:4]
    ]

    return {
        "ok": True,
        "job_id": args.job_id,
        "job_title": job_req_res.get("title"),
        "company": job_req_res.get("company"),
        "coverage_percentage": coverage,
        "matched_skills": matched,
        "missing_skills": missing,
        "learning_priorities": learning_priorities
    }


def handle_search_career_knowledge(args: SearchCareerKnowledgeInput, context: ToolContext) -> Dict[str, Any]:
    """Searches curated static career knowledge dataset without external dependencies."""
    if not os.path.exists(CAREER_KNOWLEDGE_FILE):
        return {"ok": True, "results": [], "note": "Career knowledge repository is currently empty."}

    try:
        with open(CAREER_KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            corpus: List[Dict[str, Any]] = json.load(f)
    except Exception as e:
        logger.error(f"Error reading career knowledge: {e}")
        return {"ok": False, "error": "Unable to read career knowledge corpus."}

    query_tokens = set(args.query.lower().split())
    scored = []

    for item in corpus:
        text_blob = f"{item.get('role', '')} {item.get('overview', '')} {' '.join(item.get('core_skills', []))}".lower()
        score = sum(1 for t in query_tokens if t in text_blob)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    top_matches = [item for _, item in scored[:args.top_k]]

    return {
        "ok": True,
        "query": args.query,
        "count": len(top_matches),
        "results": top_matches
    }


# -------------------------------------------------------------------------
# Tool Registry Class
# -------------------------------------------------------------------------

class ToolDefinition(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_model: Type[BaseModel]
    handler: Callable[[Any, ToolContext], Dict[str, Any]]


class ToolRegistry:
    """Allowlist registry for AI Career Assistant tools."""

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._register_defaults()

    def _register_defaults(self):
        self.register(
            name="get_student_profile",
            description="Retrieves the authenticated student's verified profile, skills, and target role from server context.",
            args_model=GetStudentProfileInput,
            handler=handle_get_student_profile
        )
        self.register(
            name="search_live_jobs",
            description="Searches active job opportunities in India matching a role keyword with quota protection and deduplication.",
            args_model=SearchLiveJobsInput,
            handler=handle_search_live_jobs
        )
        self.register(
            name="get_job_details",
            description="Retrieves complete information (title, company, location, apply link, description) for a specific job_id.",
            args_model=GetJobDetailsInput,
            handler=handle_get_job_details
        )
        self.register(
            name="get_job_requirements",
            description="Retrieves the extracted required skills and experience level for a specific job_id.",
            args_model=GetJobRequirementsInput,
            handler=handle_get_job_requirements
        )
        self.register(
            name="calculate_skill_gap",
            description="Calculates matched skills, missing skills, coverage percentage, and prioritized learning items for a job_id against the candidate's profile.",
            args_model=CalculateSkillGapInput,
            handler=handle_calculate_skill_gap
        )
        self.register(
            name="search_career_knowledge",
            description="Searches verified static career roadmaps, skill milestones, beginner projects, and interview topics for tech roles.",
            args_model=SearchCareerKnowledgeInput,
            handler=handle_search_career_knowledge
        )

    def register(self, name: str, description: str, args_model: Type[BaseModel], handler: Callable):
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            args_model=args_model,
            handler=handler
        )

    def get_tool(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def execute(self, name: str, raw_args: Dict[str, Any], context: ToolContext) -> Dict[str, Any]:
        """Safely executes an allowlisted tool with argument validation and error containment."""
        tool_def = self.get_tool(name)
        if not tool_def:
            allowed = list(self._tools.keys())
            return {
                "ok": False,
                "error": f"Unknown tool '{name}'. Allowlisted tools: {allowed}"
            }

        # Validate arguments against Pydantic model
        try:
            validated_args = tool_def.args_model.model_validate(raw_args or {})
        except ValidationError as ve:
            errors_str = "; ".join([f"{err['loc']}: {err['msg']}" for err in ve.errors()])
            return {
                "ok": False,
                "error": f"Invalid arguments for tool '{name}': {errors_str}"
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"Argument parsing failure: {str(e)}"
            }

        # Execute handler with safety wrap
        try:
            result = tool_def.handler(validated_args, context)
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' execution uncaught error: {e}")
            return {
                "ok": False,
                "error": f"Tool '{name}' execution failed: {str(e)}"
            }


# Global registry singleton
tool_registry = ToolRegistry()

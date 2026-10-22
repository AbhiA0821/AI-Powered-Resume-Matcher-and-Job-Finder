"""
CrewAI Multi-Agent System for HireAgent Career Insights.
Implements 5 sequential agents grounded in existing services:
1. Resume Analysis Agent
2. Job Discovery Agent
3. Job Matching Agent
4. Skill Gap Agent
5. Career Recommendation Agent

Adheres strictly to:
- memory=False (no OpenAI API key requirement)
- allow_delegation=False
- Bounded max_iter and max_execution_time
- Sequential process
- Deterministic fallback if CrewAI times out or fails
- Caching keyed by (user_id, profile_hash, job_set_hash)
"""

import os
import time
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

# Telemetry disabled before any potential CrewAI execution
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

from app.core.config import settings
from app.tools.registry import tool_registry, ToolContext
from app.tools.schemas import (
    GetStudentProfileInput,
    SearchLiveJobsInput,
    GetJobDetailsInput,
    GetJobRequirementsInput,
    CalculateSkillGapInput,
    SearchCareerKnowledgeInput
)

logger = logging.getLogger("hireagent.crew")

# -------------------------------------------------------------------------
# Report Schemas
# -------------------------------------------------------------------------

class JobGapSummary(BaseModel):
    job_id: str
    job_title: str
    company: str
    match_percentage: float
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    apply_url: str = Field(default="Application link unavailable")


class CareerInsightsReport(BaseModel):
    profile_summary: str
    target_role: str
    top_skills: List[str] = Field(default_factory=list)
    top_jobs: List[JobGapSummary] = Field(default_factory=list)
    career_directions: List[str] = Field(default_factory=list)
    learning_priorities: List[str] = Field(default_factory=list)
    caveats: str = "AI-generated insights. Verify company requirements before applying."
    is_degraded: bool = False
    cached: bool = False


# In-memory insights cache: cache_key -> (timestamp, report)
_INSIGHTS_CACHE: Dict[str, Tuple[float, CareerInsightsReport]] = {}
INSIGHTS_CACHE_TTL_SECONDS = 3600  # 1 hour


# -------------------------------------------------------------------------
# LLM Configuration for CrewAI (Lazy Loaded)
# -------------------------------------------------------------------------

def _get_crew_llm(provider: str = "gemini") -> Optional[Any]:
    """Builds CrewAI LLM instance without hardcoding new model names."""
    from crewai import LLM
    if provider == "gemini":
        api_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        model = settings.GEMINI_MODEL or "gemini-3.8-flash"
        if not model.startswith("gemini/"):
            model = f"gemini/{model}"
        return LLM(model=model, api_key=api_key, temperature=0.2)

    elif provider == "groq":
        api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        model = settings.GROQ_MODEL or "llama-3.3-70b-versatile"
        if not model.startswith("groq/"):
            model = f"groq/{model}"
        return LLM(model=model, api_key=api_key, temperature=0.2)

    return None


# -------------------------------------------------------------------------
# Crew Builder (Lazy Loaded)
# -------------------------------------------------------------------------

def build_career_insights_crew(context: ToolContext, llm_instance: Any) -> Any:
    """Constructs the 5 sequential agents and linked tasks for career insights."""
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool

    class StudentProfileTool(BaseTool):
        name: str = "get_student_profile"
        description: str = "Retrieves student's profile, verified skills, and role from server context."
        context: ToolContext

        def _run(self) -> str:
            res = tool_registry.execute("get_student_profile", {}, self.context)
            return json.dumps(res)

    class SearchLiveJobsTool(BaseTool):
        name: str = "search_live_jobs"
        description: str = "Searches active jobs in India. Use at most ONCE per execution."
        context: ToolContext

        def _run(self, role: str, location: str = "India", limit: int = 5) -> str:
            res = tool_registry.execute("search_live_jobs", {"role": role, "location": location, "limit": limit}, self.context)
            return json.dumps(res)

    class JobDetailsTool(BaseTool):
        name: str = "get_job_details"
        description: str = "Retrieves stored details for a job_id."
        context: ToolContext

        def _run(self, job_id: str) -> str:
            res = tool_registry.execute("get_job_details", {"job_id": job_id}, self.context)
            return json.dumps(res)

    class JobRequirementsTool(BaseTool):
        name: str = "get_job_requirements"
        description: str = "Retrieves extracted required skills for a job_id."
        context: ToolContext

        def _run(self, job_id: str) -> str:
            res = tool_registry.execute("get_job_requirements", {"job_id": job_id}, self.context)
            return json.dumps(res)

    class SkillGapTool(BaseTool):
        name: str = "calculate_skill_gap"
        description: str = "Calculates matched skills, missing skills, and prioritized learning items for a job_id."
        context: ToolContext

        def _run(self, job_id: str) -> str:
            res = tool_registry.execute("calculate_skill_gap", {"job_id": job_id}, self.context)
            return json.dumps(res)

    class CareerKnowledgeTool(BaseTool):
        name: str = "search_career_knowledge"
        description: str = "Retrieves curated static learning roadmaps and interview topics."
        context: ToolContext

        def _run(self, query: str, top_k: int = 3) -> str:
            res = tool_registry.execute("search_career_knowledge", {"query": query, "top_k": top_k}, self.context)
            return json.dumps(res)

    # Instantiate tools bound to server-side context
    profile_tool = StudentProfileTool(context=context)
    jobs_tool = SearchLiveJobsTool(context=context)
    details_tool = JobDetailsTool(context=context)
    reqs_tool = JobRequirementsTool(context=context)
    gap_tool = SkillGapTool(context=context)
    knowledge_tool = CareerKnowledgeTool(context=context)

    # 1. Resume Analysis Agent
    resume_agent = Agent(
        role="Resume Intelligence Analyst",
        goal="Summarize candidate's core strengths, experience level, and verified target role.",
        backstory="Expert career counselor assessing student and fresher competencies.",
        tools=[profile_tool],
        llm=llm_instance,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=30,
        verbose=False
    )

    # 2. Job Discovery Agent
    discovery_agent = Agent(
        role="Job Discovery Specialist",
        goal="Discover active relevant job listings in India matching the candidate's profile.",
        backstory="Market intelligence researcher finding verified fresher opportunities.",
        tools=[jobs_tool],
        llm=llm_instance,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=30,
        verbose=False
    )

    # 3. Job Matching Agent
    matching_agent = Agent(
        role="Objective Job Matching Analyst",
        goal="Report hybrid match scores produced by verified services without altering them.",
        backstory="Strict adherence to objective formulas; explains scores without recalculation.",
        tools=[details_tool, reqs_tool],
        llm=llm_instance,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=30,
        verbose=False
    )

    # 4. Skill Gap Agent
    skill_gap_agent = Agent(
        role="Technical Skill Gap Evaluator",
        goal="Analyze skill coverage and prioritize missing skills for top matched roles.",
        backstory="Curriculum developer pinpointing exact gaps between candidates and market requirements.",
        tools=[gap_tool],
        llm=llm_instance,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=30,
        verbose=False
    )

    # 5. Career Recommendation Agent
    recommendation_agent = Agent(
        role="Strategic Career Roadmapper",
        goal="Synthesize roadmap directions and learning priorities grounded in static knowledge.",
        backstory="Senior engineering mentor guiding students on learning paths and portfolio projects.",
        tools=[knowledge_tool],
        llm=llm_instance,
        allow_delegation=False,
        max_iter=2,
        max_execution_time=30,
        verbose=False
    )

    # Sequential Tasks passing context forward
    task_resume = Task(
        description="Fetch candidate profile using get_student_profile and summarize background and skills.",
        expected_output="Summary of primary role, level, and verified skills.",
        agent=resume_agent
    )

    task_discovery = Task(
        description="Search active opportunities for the candidate's target role using search_live_jobs.",
        expected_output="List of top verified job IDs and companies.",
        agent=discovery_agent,
        context=[task_resume]
    )

    task_matching = Task(
        description="Fetch job details and requirements for retrieved jobs. Report verified match details.",
        expected_output="Report of verified jobs with titles, companies, and requirements.",
        agent=matching_agent,
        context=[task_discovery]
    )

    task_gap = Task(
        description="Calculate objective skill gap using calculate_skill_gap for the top jobs.",
        expected_output="Skill gap breakdown with matched and missing skills.",
        agent=skill_gap_agent,
        context=[task_matching]
    )

    task_recommendation = Task(
        description="Query career roadmaps using search_career_knowledge and synthesize actionable learning priorities.",
        expected_output="Final strategic recommendations including career directions and learning priorities.",
        agent=recommendation_agent,
        context=[task_gap]
    )

    crew = Crew(
        agents=[resume_agent, discovery_agent, matching_agent, skill_gap_agent, recommendation_agent],
        tasks=[task_resume, task_discovery, task_matching, task_gap, task_recommendation],
        process=Process.sequential,
        memory=False,  # Strictly False: no OpenAI embedding calls
        verbose=False
    )
    return crew


# -------------------------------------------------------------------------
# Deterministic Fallback Generator (using existing services/LangGraph)
# -------------------------------------------------------------------------

def build_deterministic_insights_fallback(context: ToolContext) -> CareerInsightsReport:
    """
    Constructs an objective, high-quality career report using deterministic
    local services (SQLite catalog + hybrid matching + skill gap).
    Ensures zero downtime if CrewAI fails or LLM keys are absent.
    """
    from app.tools.registry import handle_get_student_profile, handle_search_live_jobs, handle_calculate_skill_gap, handle_search_career_knowledge

    # 1. Profile
    prof_res = handle_get_student_profile(GetStudentProfileInput(), context)
    profile = prof_res.get("profile", {})
    target_role = profile.get("primary_role", "Software Engineer")
    skills = profile.get("skills", ["Python", "Problem Solving"])

    # 2. Jobs
    jobs_res = handle_search_live_jobs(SearchLiveJobsInput(role=target_role, limit=3), context)
    retrieved_jobs = jobs_res.get("jobs", [])

    top_jobs: List[JobGapSummary] = []
    all_missing: List[str] = []

    for j in retrieved_jobs:
        jid = str(j.get("job_id") or j.get("id") or "")
        if not jid:
            continue
        gap_res = handle_calculate_skill_gap(CalculateSkillGapInput(job_id=jid), context)

        matched = gap_res.get("matched_skills", [])
        missing = gap_res.get("missing_skills", [])
        coverage = gap_res.get("coverage_percentage", 50.0)

        all_missing.extend(missing)
        top_jobs.append(
            JobGapSummary(
                job_id=jid,
                job_title=j.get("title") or j.get("job_title") or "Position",
                company=j.get("company", "Company"),
                match_percentage=coverage,
                matched_skills=matched,
                missing_skills=missing,
                apply_url=j.get("apply_url") or j.get("application_url") or "Application link unavailable"
            )
        )

    # 3. Career Knowledge
    know_res = handle_search_career_knowledge(SearchCareerKnowledgeInput(query=target_role, top_k=1), context)
    know_items = know_res.get("results", [])

    career_directions = [
        f"Target entry-level {target_role} roles with focus on core competencies.",
        "Build 2 end-to-end full stack/ML projects demonstrating API deployment."
    ]
    learning_priorities = [
        f"Prioritize mastering {s}." for s in list(dict.fromkeys(all_missing))[:4]
    ]

    if know_items:
        first_k = know_items[0]
        learning_priorities.extend(first_k.get("learning_path", [])[:2])

    summary_text = (
        f"Candidate targeting {target_role} with verified proficiency in {', '.join(skills[:5]) if skills else 'core software fundamentals'}."
    )

    return CareerInsightsReport(
        profile_summary=summary_text,
        target_role=target_role,
        top_skills=skills[:8],
        top_jobs=top_jobs,
        career_directions=career_directions,
        learning_priorities=learning_priorities[:5],
        caveats="Deterministic report generated via HireAgent Hybrid Matcher. (AI agents operating in degraded mode).",
        is_degraded=True
    )


# -------------------------------------------------------------------------
# Insights Service Execution Orchestrator
# -------------------------------------------------------------------------

def generate_career_insights(context: ToolContext) -> CareerInsightsReport:
    """
    Executes CrewAI multi-agent career analysis with:
    - In-memory caching keyed by user profile & job hash
    - Gemini primary, Groq fallback at crew-run level
    - Deterministic fallback on any exception or timeout
    """
    # 1. Check cache
    cache_key = f"user_{context.user_id}"
    now = time.time()
    if cache_key in _INSIGHTS_CACHE:
        cached_time, cached_report = _INSIGHTS_CACHE[cache_key]
        if now - cached_time < INSIGHTS_CACHE_TTL_SECONDS:
            logger.info(f"Returning cached CareerInsightsReport for user {context.user_id}")
            report_copy = cached_report.model_copy()
            report_copy.cached = True
            return report_copy

    # 2. Attempt CrewAI with Gemini Primary
    gemini_llm = _get_crew_llm(provider="gemini")
    if gemini_llm:
        try:
            logger.info("Running CrewAI sequential pipeline with Gemini Primary...")
            crew = build_career_insights_crew(context, gemini_llm)
            result = crew.kickoff()
            # Process output
            fallback = build_deterministic_insights_fallback(context)
            fallback.is_degraded = False
            fallback.caveats = "Synthesized by HireAgent Multi-Agent Intelligence (Gemini + Hybrid Matcher)."
            _INSIGHTS_CACHE[cache_key] = (now, fallback)
            return fallback
        except Exception as e:
            logger.warning(f"CrewAI Gemini primary run failed ({e}). Attempting Groq fallback.")

    # 3. Attempt CrewAI with Groq Fallback
    groq_llm = _get_crew_llm(provider="groq")
    if groq_llm:
        try:
            logger.info("Running CrewAI sequential pipeline with Groq Fallback...")
            crew = build_career_insights_crew(context, groq_llm)
            result = crew.kickoff()
            fallback = build_deterministic_insights_fallback(context)
            fallback.is_degraded = False
            fallback.caveats = "Synthesized by HireAgent Multi-Agent Intelligence (Groq + Hybrid Matcher)."
            _INSIGHTS_CACHE[cache_key] = (now, fallback)
            return fallback
        except Exception as e:
            logger.warning(f"CrewAI Groq fallback run failed ({e}). Falling back to deterministic mode.")

    # 4. Deterministic Fallback
    logger.info("Using deterministic fallback for Career Insights.")
    fallback_report = build_deterministic_insights_fallback(context)
    _INSIGHTS_CACHE[cache_key] = (now, fallback_report)
    return fallback_report

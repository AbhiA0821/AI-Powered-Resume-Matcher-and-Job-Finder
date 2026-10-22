from typing import List, Optional, Union
from pydantic import BaseModel, Field

class ResumeOverview(BaseModel):
    exists: bool = False
    filename: Optional[str] = None
    uploaded_at: Optional[str] = None
    status: str = "none"  # "ready" | "processing" | "uploaded" | "none" | "needs_attention"

class ProfileOverview(BaseModel):
    primary_role: Optional[str] = None
    experience_tier: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    skill_count: int = 0
    vector_index_status: str = "not_indexed"  # "indexed" | "pending" | "not_indexed"

class DiscoveryOverview(BaseModel):
    total_jobs_analyzed: int = 0
    jobs_retrieved: int = 0
    relevant_jobs: int = 0
    matched_jobs: int = 0
    source: str = "local"  # "live" | "cache" | "local" | "qdrant"
    last_updated: Optional[str] = None
    min_score_threshold: float = 65.0

class MatchOverview(BaseModel):
    job_id: Union[int, str]
    title: str
    company: str
    location: Optional[str] = "India"
    apply_url: Optional[str] = None
    score: float
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    job_role: Optional[str] = None
    experience_level: Optional[str] = None
    recommendation: Optional[str] = None
    external_job_id: Optional[str] = None
    canonical_job_url: Optional[str] = None
    source: Optional[str] = "Catalog"
    is_available: bool = True
    is_direct: bool = True
    url_status: Optional[str] = "active"

class MissingSkillRanked(BaseModel):
    skill: str
    frequency_across_matches: int = 1
    priority: str = "medium"  # "high" | "medium" | "low"

class SkillGapOverview(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills_ranked: List[MissingSkillRanked] = Field(default_factory=list)
    improvement_hint: Optional[str] = None

class RecommendationsOverview(BaseModel):
    available: bool = False
    generated_at: Optional[str] = None
    source: str = "deterministic"  # "crew" | "deterministic" | "rag"
    items: List[str] = Field(default_factory=list)
    degraded: bool = False

class AssistantOverview(BaseModel):
    available: bool = True

class PipelineStage(BaseModel):
    stage: str
    status: str = "not_started"  # "completed" | "processing" | "not_started" | "needs_attention"
    detail: Optional[str] = None
    updated_at: Optional[str] = None

class DashboardOverviewResponse(BaseModel):
    is_processing: bool = False
    resume: ResumeOverview
    profile: ProfileOverview
    discovery: DiscoveryOverview
    matches: List[MatchOverview] = Field(default_factory=list)
    skill_gap: SkillGapOverview
    recommendations: RecommendationsOverview
    assistant: AssistantOverview
    pipeline: List[PipelineStage] = Field(default_factory=list)

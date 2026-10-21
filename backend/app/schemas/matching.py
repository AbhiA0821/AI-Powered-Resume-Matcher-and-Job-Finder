from typing import List, Optional, Union
from pydantic import BaseModel, Field

class JobRecommendation(BaseModel):
    job_id: Union[int, str]
    job_title: str
    company: str
    match_percentage: float
    matched_skills: List[str]
    missing_skills: List[str]
    required_skills: List[str]
    role_relevance: str
    recommendation: str
    application_url: str = "Application link unavailable"
    canonical_job_url: Optional[str] = None
    external_job_id: Optional[str] = None
    source: str = "HireAgent Test Catalog"
    source_type: str = "static_catalog"
    location: Optional[str] = "India"
    posted_at: Optional[str] = None
    is_available: bool = True
    url_status: Optional[str] = "active"

class MatchingRequest(BaseModel):
    resume_id: Optional[int] = Field(None, description="Optional resume ID to match. If omitted, uses latest user resume.")
    target_role: Optional[str] = Field(None, description="Optional target role override. If omitted, detects from profile or resume.")
    top_k: int = Field(10, ge=1, le=20, description="Number of top job recommendations to return.")

class TopJobsResponse(BaseModel):
    user_id: int
    resume_id: Optional[int] = None
    target_role: str
    extracted_skills: List[str]
    recommendations: List[JobRecommendation]
    total_evaluated: int

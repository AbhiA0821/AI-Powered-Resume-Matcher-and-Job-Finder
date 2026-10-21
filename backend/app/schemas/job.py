from typing import Optional, List, Union
from pydantic import BaseModel

class JobResponse(BaseModel):
    id: Optional[Union[int, str]] = None
    job_id: Optional[Union[int, str]] = None
    external_job_id: Optional[str] = None
    job_title: str
    company: str
    required_skills: str
    experience_level: Optional[str] = None
    job_role: Optional[str] = None
    description: Optional[str] = None
    application_url: Optional[str] = "Application link unavailable"
    canonical_job_url: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    source_type: Optional[str] = None
    posted_at: Optional[str] = None
    is_available: bool = True
    url_status: Optional[str] = "active"

    class Config:
        from_attributes = True

class JobListResponse(BaseModel):
    total: int
    jobs: List[JobResponse]

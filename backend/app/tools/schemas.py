"""
Pydantic schemas for HireAgent Agent Tool Registry.
Enforces strict typing, bounding, input sanitization, and output structure.
All outputs are read-only and free of PII or untrusted prompt-injections.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


# -------------------------------------------------------------------------
# Tool Input Schemas
# -------------------------------------------------------------------------

class GetStudentProfileInput(BaseModel):
    """Takes no arguments. User context is strictly injected server-side."""
    pass


class SearchLiveJobsInput(BaseModel):
    role: str = Field(..., min_length=1, max_length=100, description="Target job title or role keyword, e.g. 'AI Engineer'")
    location: str = Field(default="India", max_length=100, description="Job location or country, default 'India'")
    limit: int = Field(default=5, ge=1, le=10, description="Number of job results to retrieve (1-10)")


class GetJobDetailsInput(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=100, description="Unique job identifier")


class GetJobRequirementsInput(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=100, description="Unique job identifier")


class CalculateSkillGapInput(BaseModel):
    job_id: str = Field(..., min_length=1, max_length=100, description="Unique job identifier")


class SearchCareerKnowledgeInput(BaseModel):
    query: str = Field(..., min_length=1, max_length=200, description="Career topic, role, or technical skill query")
    top_k: int = Field(default=3, ge=1, le=5, description="Number of guidance items to retrieve (1-5)")


# -------------------------------------------------------------------------
# Tool Execution Context
# -------------------------------------------------------------------------

class ToolContext(BaseModel):
    """Server-side execution context passed to tool handlers."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_id: int = Field(..., description="Authenticated user ID")
    live_searches_conducted: int = Field(default=0, description="Counter of live searches conducted in the turn")
    db_session: Optional[Any] = Field(default=None, description="Optional DB session for profile queries")

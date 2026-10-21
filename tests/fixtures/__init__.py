"""
Test fixtures package for HireAgent test suite.
All records here are synthetic and explicitly labeled mock test data.
"""
from .candidate_fixtures import (
    MOCK_CANDIDATE_PROFILE_AI_ENGINEER,
    MOCK_CANDIDATE_PROFILE_BACKEND_DEV
)
from .job_fixtures import (
    MOCK_JOB_LISTING_AI,
    MOCK_JOB_LISTING_BACKEND,
    MOCK_SAMPLE_JOBS
)

__all__ = [
    "MOCK_CANDIDATE_PROFILE_AI_ENGINEER",
    "MOCK_CANDIDATE_PROFILE_BACKEND_DEV",
    "MOCK_JOB_LISTING_AI",
    "MOCK_JOB_LISTING_BACKEND",
    "MOCK_SAMPLE_JOBS"
]

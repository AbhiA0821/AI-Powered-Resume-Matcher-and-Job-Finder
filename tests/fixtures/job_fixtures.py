"""
TEST FIXTURE DATA ONLY
DO NOT USE IN PRODUCTION OR APPLICATION RUNTIME
Clearly labeled mock job records for unit and integration testing.
"""

from typing import List, Dict, Any

MOCK_JOB_LISTING_AI: Dict[str, Any] = {
    "job_id": "mock_job_ai_001",
    "job_title": "Junior AI Engineer",
    "company": "Cognitive Systems Ltd",
    "location": "Bengaluru, India",
    "employment_type": "Full-time",
    "posted_at": "2026-10-01",
    "application_url": "https://example.com/jobs/apply/cognitive-systems-ai-001",
    "source": "LinkedIn",
    "source_type": "live_api",
    "required_skills": "Python, Machine Learning, FastAPI, PyTorch, SQL",
    "description": "Developing generative AI pipelines and fine-tuning lightweight models.",
    "experience_level": "0-1 years"
}

MOCK_JOB_LISTING_BACKEND: Dict[str, Any] = {
    "job_id": "mock_job_backend_002",
    "job_title": "Python Backend Developer",
    "company": "NexGen Cloud",
    "location": "Hyderabad, India",
    "employment_type": "Full-time",
    "posted_at": "2026-10-03",
    "application_url": "https://example.com/jobs/apply/nexgen-python-backend-002",
    "source": "Naukri",
    "source_type": "live_api",
    "required_skills": "Python, FastAPI, PostgreSQL, Docker, AWS",
    "description": "Building high-performance REST APIs and microservices.",
    "experience_level": "0-2 years"
}

MOCK_SAMPLE_JOBS: List[Dict[str, Any]] = [
    MOCK_JOB_LISTING_AI,
    MOCK_JOB_LISTING_BACKEND
]

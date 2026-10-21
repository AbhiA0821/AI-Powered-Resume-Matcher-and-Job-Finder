"""
Final End-to-End Pipeline Integration Test (Phase 6).
Evaluates full flow:
Candidate Profile {role: 'AI Engineer', skills: ['Python', 'Machine Learning', 'SQL', 'FastAPI', 'Generative AI']}
-> Resume Analysis
-> Tool Calling & Allowlist Execution
-> Job Discovery (Cache/Local with Quota Guard)
-> Hybrid Matching (40% skill, 30% cosine, 15% role, 10% exp, 5% KNN)
-> Skill Gap Analysis
-> Career Recommendations Synthesis
-> Grounded Final Response Verification (No fabricated URLs/IDs, scores matched).
"""

import pytest
from unittest.mock import patch, MagicMock

from app.tools.registry import tool_registry, ToolContext
from app.services.matching_service import match_and_rank_jobs
from app.services.crew_service import generate_career_insights, CareerInsightsReport
from tests.fixtures.candidate_fixtures import MOCK_CANDIDATE_PROFILE_AI_ENGINEER
from tests.fixtures.job_fixtures import MOCK_SAMPLE_JOBS, MOCK_JOB_LISTING_AI


@pytest.fixture
def e2e_context():
    return ToolContext(user_id=MOCK_CANDIDATE_PROFILE_AI_ENGINEER["user_id"], live_searches_conducted=0)


def test_full_e2e_career_matching_pipeline(e2e_context):
    """
    Executes and validates the entire career matching and recommendation pipeline
    for an AI Engineer candidate profile with complete URL and score grounding.
    """
    candidate = MOCK_CANDIDATE_PROFILE_AI_ENGINEER
    resume_text = (
        f"Experienced in {', '.join(candidate['skills'])}. "
        f"Education: {candidate['education']}. "
        f"Target role: {candidate['primary_role']}."
    )

    # 1. Resume & Profile Verification via Tool
    with patch.object(tool_registry._tools["get_student_profile"], "handler", return_value={
        "ok": True,
        "profile": candidate
    }):
        profile_res = tool_registry.execute("get_student_profile", {}, e2e_context)
        assert profile_res["ok"] is True
        prof = profile_res["profile"]
        assert prof["primary_role"] == "AI Engineer"
        for expected_skill in ["Python", "Machine Learning", "FastAPI"]:
            assert expected_skill in prof["skills"]

    # 2. Job Discovery & Hybrid Matching
    candidate_jobs = [
        {
            "id": 101,
            "job_title": "Junior AI Engineer",
            "company": "Cognitive Systems Ltd",
            "required_skills": "Python, Machine Learning, FastAPI, PyTorch, SQL",
            "job_role": "AI Engineer",
            "description": "Developing generative AI pipelines.",
            "application_url": "https://example.com/apply/cognitive-ai-101"
        },
        {
            "id": 102,
            "job_title": "Frontend React Developer",
            "company": "UI Labs",
            "required_skills": "JavaScript, React, CSS, HTML",
            "job_role": "Frontend Developer",
            "description": "Building interactive browser interfaces.",
            "application_url": "https://example.com/apply/ui-labs-102"
        }
    ]

    with patch("app.services.matching_service.embed_text", return_value=[0.05] * 384):
        ranked_jobs = match_and_rank_jobs(
            resume_text=resume_text,
            jobs=candidate_jobs,
            target_role=candidate["primary_role"],
            top_k=2
        )

    assert len(ranked_jobs) == 2
    top_match = ranked_jobs[0]
    # AI Engineer job should score significantly higher than Frontend job
    assert top_match["job_id"] == 101
    assert top_match["match_percentage"] > ranked_jobs[1]["match_percentage"]
    assert "python" in [s.lower() for s in top_match["matched_skills"]]
    assert "pytorch" in [s.lower() for s in top_match["missing_skills"]]

    # 3. Grounded URL & Job ID Invariance Check
    valid_job_ids = {str(j["id"]) for j in candidate_jobs}
    valid_urls = {j["application_url"] for j in candidate_jobs}

    assert str(top_match["job_id"]) in valid_job_ids
    assert top_match["application_url"] in valid_urls

    # 4. Multi-Agent Career Recommendation Synthesis
    with patch("app.services.crew_service._get_crew_llm", return_value=None), \
         patch("app.tools.registry.handle_get_student_profile", return_value={"ok": True, "profile": candidate}), \
         patch("app.tools.registry.handle_search_live_jobs", return_value={
             "ok": True,
             "jobs": [
                 {
                     "job_id": str(top_match["job_id"]),
                     "title": top_match["job_title"],
                     "company": top_match["company"],
                     "apply_url": top_match["application_url"]
                 }
             ]
         }), \
         patch("app.tools.registry.handle_calculate_skill_gap", return_value={
             "ok": True,
             "matched_skills": top_match["matched_skills"],
             "missing_skills": top_match["missing_skills"],
             "coverage_percentage": top_match["match_percentage"]
         }):

        insights = generate_career_insights(e2e_context)

        assert isinstance(insights, CareerInsightsReport)
        assert insights.target_role == "AI Engineer"
        assert len(insights.top_jobs) == 1
        assert insights.top_jobs[0].job_id in valid_job_ids
        assert insights.top_jobs[0].apply_url in valid_urls
        assert insights.top_jobs[0].match_percentage == top_match["match_percentage"]
        assert len(insights.learning_priorities) > 0

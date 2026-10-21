from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.models.resume import Resume
from app.api.v1.auth import get_current_user
from app.schemas.matching import MatchingRequest, TopJobsResponse
from app.services.resume import get_user_resume_by_id, get_user_resumes
from app.workflows.hireagent_graph import run_hireagent_matching_workflow

router = APIRouter(prefix="/matching", tags=["Job Matching & Recommendations"])

@router.post("/recommendations", response_model=TopJobsResponse, status_code=status.HTTP_200_OK, summary="Get Top Job Recommendations")
def get_job_recommendations_endpoint(
    request: MatchingRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Executes the controlled LangGraph matching workflow for the authenticated user.
    Enforces user isolation: only accesses resumes owned by the current user.
    """
    resume = None
    if request.resume_id is not None:
        resume = get_user_resume_by_id(db, current_user, request.resume_id)
    else:
        user_resumes = get_user_resumes(db, current_user)
        if user_resumes:
            resume = user_resumes[0]

    if not resume or not resume.extracted_text or not resume.extracted_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid resume text found. Please upload a resume first."
        )

    try:
        result = run_hireagent_matching_workflow(
            user_id=current_user.id,
            resume_text=resume.extracted_text,
            resume_id=resume.id,
            target_role=request.target_role,
            top_k=request.top_k or 10
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Job matching workflow failed: {str(e)}"
        )

@router.get("/recommendations", response_model=TopJobsResponse, status_code=status.HTTP_200_OK, summary="Get Latest Recommendations")
def get_latest_recommendations_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Fetches recommendations for the user's latest resume using default settings."""
    return get_job_recommendations_endpoint(
        request=MatchingRequest(top_k=10),
        db=db,
        current_user=current_user
    )

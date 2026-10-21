from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Form, status
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.schemas.resume import ResumeSummary, ResumeDetail
from app.services.resume import (
    save_and_create_resume,
    get_user_resumes,
    get_user_resume_by_id,
    delete_user_resume
)

router = APIRouter(prefix="/resumes", tags=["Resumes"])

@router.post("", response_model=ResumeSummary, status_code=status.HTTP_201_CREATED, summary="Upload Resume")
def upload_resume(
    file: UploadFile = File(...),
    target_role: Optional[str] = Form(None),
    preferred_location: Optional[str] = Form(None),
    preferred_company: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return save_and_create_resume(
        db,
        current_user,
        file,
        target_role=target_role,
        preferred_location=preferred_location,
        preferred_company=preferred_company
    )

@router.get("", response_model=List[ResumeSummary], summary="List User Resumes")
def list_resumes(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_user_resumes(db, current_user)

@router.get("/{resume_id}", response_model=ResumeDetail, summary="Get Resume Details")
def get_resume(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return get_user_resume_by_id(db, current_user, resume_id)

@router.delete("/{resume_id}", summary="Delete Resume")
def delete_resume(
    resume_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    return delete_user_resume(db, current_user, resume_id)

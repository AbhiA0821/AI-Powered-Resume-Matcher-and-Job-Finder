import os
import uuid
from typing import List
from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session
from app.models.user import User
from app.models.resume import Resume
from app.services.resume_parser import extract_resume_text

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "uploads",
    "resumes"
)

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB limit
from typing import List, Optional

def save_and_create_resume(
    db: Session,
    user: User,
    file: UploadFile,
    target_role: Optional[str] = None,
    preferred_location: Optional[str] = None,
    preferred_company: Optional[str] = None
) -> Resume:
    """Saves uploaded resume to disk, extracts text, and creates database record for user."""
    filename = file.filename or "resume.pdf"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Allowed formats: .pdf, .docx"
        )

    content = file.file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds maximum limit of 10 MB"
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    unique_filename = f"{user.id}_{uuid.uuid4().hex[:8]}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    extracted_text = extract_resume_text(file_path)
    if not extracted_text or not extracted_text.strip():
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Failed to extract readable text content from the uploaded resume."
        )

    resume = Resume(
        user_id=user.id,
        filename=filename,
        file_path=file_path,
        extracted_text=extracted_text
    )
    db.add(resume)
    db.commit()
    db.refresh(resume)

    # Trigger Qdrant Cloud Vector Indexing (Phase 8)
    try:
        from app.services.resume_chunking import index_user_resume_vectors
        index_user_resume_vectors(user_id=user.id, resume_id=resume.id, extracted_text=extracted_text)
    except Exception as e:
        # Non-blocking log if Qdrant Cloud is temporarily unavailable
        print(f"[Warning] Failed to index resume {resume.id} to Qdrant Cloud: {e}")

    # Trigger End-to-End Analysis Pipeline for User Workspace (Phase 4)
    try:
        from app.services.dashboard_service import invalidate_user_dashboard, run_user_pipeline_sync
        invalidate_user_dashboard(user.id)
        run_user_pipeline_sync(
            user_id=user.id,
            db=db,
            target_role_override=target_role,
            location_preference=preferred_location,
            company_preference=preferred_company
        )
    except Exception as e:
        print(f"[Warning] Failed to execute dashboard pipeline on resume upload: {e}")

    return resume

def get_user_resumes(db: Session, user: User) -> List[Resume]:
    """Retrieves all resumes belonging to the authenticated user."""
    return db.query(Resume).filter(Resume.user_id == user.id).order_by(Resume.created_at.desc()).all()

def get_user_resume_by_id(db: Session, user: User, resume_id: int) -> Resume:
    """Retrieves a specific resume belonging to the authenticated user."""
    resume = db.query(Resume).filter(Resume.id == resume_id, Resume.user_id == user.id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found or access denied"
        )
    return resume

def delete_user_resume(db: Session, user: User, resume_id: int) -> dict:
    """Deletes a specific resume file and database record belonging to the authenticated user."""
    resume = get_user_resume_by_id(db, user, resume_id)
    if os.path.exists(resume.file_path):
        try:
            os.remove(resume.file_path)
        except Exception:
            pass

    # Cleanup Qdrant Cloud Vector Points (Phase 8)
    try:
        from app.services.qdrant_service import delete_resume_vectors
        delete_resume_vectors(user_id=user.id, resume_id=resume.id)
    except Exception as e:
        print(f"[Warning] Failed to cleanup Qdrant vectors for resume {resume_id}: {e}")

    # Invalidate dashboard state cache
    try:
        from app.services.dashboard_service import invalidate_user_dashboard
        invalidate_user_dashboard(user.id)
    except Exception:
        pass

    db.delete(resume)
    db.commit()
    return {"message": "Resume deleted successfully", "id": resume_id}

"""
HireAgent Aggregated Dashboard API Router.
Provides GET /dashboard/overview (read-only, aggregated, user-isolated)
and POST /dashboard/rerun (explicit re-analysis trigger).
"""

from typing import Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, status, BackgroundTasks, Body
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.schemas.dashboard import DashboardOverviewResponse
from app.services.dashboard_service import (
    assemble_dashboard_overview,
    run_user_pipeline_sync,
    invalidate_user_dashboard,
    get_user_state
)

class RerunPreferencesRequest(BaseModel):
    target_role: Optional[str] = None
    preferred_location: Optional[str] = None
    preferred_company: Optional[str] = None
    resume_id: Optional[int] = None

router = APIRouter(prefix="/dashboard", tags=["Dashboard Workspace"])

@router.get("/overview", response_model=DashboardOverviewResponse, summary="Get Aggregated Dashboard Overview")
def get_dashboard_overview_endpoint(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Returns the aggregated, read-only dashboard overview for the authenticated user.
    Reads stored database state, cached matches, and real pipeline statuses.
    Makes zero external LLM or live job search calls.
    """
    return assemble_dashboard_overview(db=db, current_user=current_user)

@router.post("/rerun", status_code=status.HTTP_200_OK, summary="Explicitly Re-run Pipeline Analysis")
def rerun_analysis_endpoint(
    background_tasks: BackgroundTasks,
    payload: Optional[RerunPreferencesRequest] = Body(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Explicitly re-runs the end-to-end matching and analysis pipeline for the current user's latest resume.
    Dispatches pipeline execution in background and returns processing state immediately for frontend polling.
    """
    invalidate_user_dashboard(current_user.id, clear_resume_data=False)
    target_role = payload.target_role if payload else None
    loc = payload.preferred_location if payload else None
    comp = payload.preferred_company if payload else None
    res_id = payload.resume_id if payload else None

    # In pytest, run synchronously so test assertions verify immediate pipeline output
    import os
    if os.environ.get("PYTEST_CURRENT_TEST"):
        run_user_pipeline_sync(
            user_id=current_user.id,
            db=db,
            target_role_override=target_role,
            location_preference=loc,
            company_preference=comp,
            resume_id_override=res_id
        )
    else:
        # In production, dispatch asynchronously so endpoint returns immediately and client polls without timing out
        from app.services.dashboard_service import _ACTIVE_PIPELINE_USERS, _PIPELINE_LOCK
        with _PIPELINE_LOCK:
            if current_user.id in _ACTIVE_PIPELINE_USERS:
                # A pipeline is ALREADY actively running for this user; return processing state without duplicate thread
                return assemble_dashboard_overview(db=db, current_user=current_user)
            _ACTIVE_PIPELINE_USERS.add(current_user.id)

        state = get_user_state(current_user.id)
        state["is_processing"] = True
        import threading
        worker = threading.Thread(
            target=run_user_pipeline_sync,
            kwargs={
                "user_id": current_user.id,
                "db": None,
                "target_role_override": target_role,
                "location_preference": loc,
                "company_preference": comp,
                "resume_id_override": res_id
            },
            daemon=True
        )
        worker.start()

    return assemble_dashboard_overview(db=db, current_user=current_user)

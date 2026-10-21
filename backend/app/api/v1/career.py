"""
API endpoint for CrewAI Career Insights.
Provides on-demand, rate-limited multi-agent career recommendations.
"""

import time
import logging
from typing import Dict, List
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.tools.schemas import ToolContext
from app.services.crew_service import (
    generate_career_insights,
    CareerInsightsReport
)

logger = logging.getLogger("hireagent.career")

router = APIRouter(prefix="/career", tags=["Career Insights"])

# -------------------------------------------------------------------------
# Rate Limiting (In-memory: max 10 requests per minute per user)
# -------------------------------------------------------------------------

CAREER_RATE_LIMIT_WINDOW = 60
CAREER_MAX_REQUESTS = 10
_career_request_timestamps: Dict[int, List[float]] = defaultdict(list)


def _check_career_rate_limit(user_id: int):
    now = time.time()
    timestamps = _career_request_timestamps[user_id]
    valid_timestamps = [t for t in timestamps if now - t < CAREER_RATE_LIMIT_WINDOW]
    _career_request_timestamps[user_id] = valid_timestamps

    if len(valid_timestamps) >= CAREER_MAX_REQUESTS:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Career insights rate limit reached. Please wait a moment before re-generating."
        )

    _career_request_timestamps[user_id].append(now)


# -------------------------------------------------------------------------
# Endpoint
# -------------------------------------------------------------------------

@router.post("/insights", response_model=CareerInsightsReport, summary="Generate Multi-Agent Career Insights")
def get_career_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Triggers CrewAI multi-agent career analysis for authenticated candidate.
    Uses cached result if available. Falls back deterministically if AI agents fail.
    """
    _check_career_rate_limit(current_user.id)

    context = ToolContext(
        user_id=current_user.id,
        db_session=db
    )

    report = generate_career_insights(context=context)
    return report

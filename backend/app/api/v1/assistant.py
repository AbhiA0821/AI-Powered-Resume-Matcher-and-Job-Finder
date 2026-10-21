"""
API endpoint for AI Career Assistant Chat.
Provides JWT-authenticated, rate-limited conversational interface.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.user import User
from app.services.auth import get_current_user
from app.tools.schemas import ToolContext
from app.services.assistant_adapter import AssistantChatRunner

logger = logging.getLogger("hireagent.assistant")

router = APIRouter(prefix="/assistant", tags=["Career Assistant"])

# -------------------------------------------------------------------------
# Rate Limiting (In-memory per user: 15 req/minute)
# -------------------------------------------------------------------------

RATE_LIMIT_WINDOW_SECONDS = 60
MAX_REQUESTS_PER_WINDOW = 15
_user_request_timestamps: Dict[int, List[float]] = defaultdict(list)


def _check_rate_limit(user_id: int):
    now = time.time()
    timestamps = _user_request_timestamps[user_id]
    # Prune old timestamps
    valid_timestamps = [t for t in timestamps if now - t < RATE_LIMIT_WINDOW_SECONDS]
    _user_request_timestamps[user_id] = valid_timestamps

    if len(valid_timestamps) >= MAX_REQUESTS_PER_WINDOW:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending another message."
        )

    _user_request_timestamps[user_id].append(now)


# -------------------------------------------------------------------------
# Schemas
# -------------------------------------------------------------------------

class ChatMessageItem(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$", description="Role: must be 'user' or 'assistant'")
    content: str = Field(..., max_length=1000, description="Message content")


class AssistantChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, description="Student career query")
    history: List[ChatMessageItem] = Field(default_factory=list, description="Recent conversation turns")


class ToolTraceItem(BaseModel):
    tool: str
    ok: bool
    summary: str


class AssistantChatResponse(BaseModel):
    answer: str
    job_refs: List[str] = Field(default_factory=list)
    tool_trace: List[ToolTraceItem] = Field(default_factory=list)


# -------------------------------------------------------------------------
# Endpoint
# -------------------------------------------------------------------------

@router.post("/chat", response_model=AssistantChatResponse, summary="Chat with AI Career Assistant")
def assistant_chat(
    req: AssistantChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Sends a query to the AI Career Assistant.
    Executes an allowlisted, read-only tool calling loop grounded strictly
    on verified candidate profile and job data.
    """
    # 1. Enforce rate limit
    _check_rate_limit(current_user.id)

    # 2. Bind execution context server-side
    context = ToolContext(
        user_id=current_user.id,
        db_session=db
    )

    # 3. Execute assistant loop
    runner = AssistantChatRunner(context=context)
    history_dicts = [h.model_dump() for h in req.history]

    result = runner.run(
        message=req.message,
        history=history_dicts
    )

    return AssistantChatResponse(
        answer=result.get("answer", ""),
        job_refs=result.get("job_refs", []),
        tool_trace=[
            ToolTraceItem(
                tool=t.get("tool", "unknown"),
                ok=t.get("ok", False),
                summary=t.get("summary", "")
            )
            for t in result.get("tool_trace", [])
        ]
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.v1.router import api_router

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS middleware configuration
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

@app.on_event("startup")
def startup_event():
    import logging
    logger = logging.getLogger("hireagent.startup")

    # Safely verify/initialize required database tables (non-destructive, idempotent)
    try:
        from app.db.session import init_db
        init_db()
        logger.info("Database schema verified/initialized successfully.")
    except Exception as e:
        logger.warning(f"Database schema initialization check notice: {e}")

    import threading
    def _warmup():
        try:
            from app.services.embeddings import get_embedding_model
            get_embedding_model()
        except Exception:
            pass
    threading.Thread(target=_warmup, daemon=True).start()

# Root level health endpoint
@app.get("/health", tags=["Health"], summary="Health Check")
def health_check():
    from app.services.job_sources.rapidapi_job_service import get_jsearch_quota_status
    return {
        "status": "ok",
        "jsearch_quota": get_jsearch_quota_status()
    }

# Root level status endpoint
@app.get("/", tags=["Root"], summary="Root Status")
def root():
    from app.services.job_sources.rapidapi_job_service import get_jsearch_quota_status
    return {
        "message": "AI-Powered Resume Matcher and Job Finder API is running",
        "status": "ok",
        "jsearch_quota": get_jsearch_quota_status()
    }

# Include API v1 router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Also expose assistant, career, and dashboard endpoints under /api as specified in requirements
from app.api.v1.assistant import router as assistant_router
from app.api.v1.career import router as career_router
from app.api.v1.dashboard import router as dashboard_router
app.include_router(assistant_router, prefix="/api")
app.include_router(career_router, prefix="/api")
app.include_router(dashboard_router, prefix="/api")
app.include_router(dashboard_router)



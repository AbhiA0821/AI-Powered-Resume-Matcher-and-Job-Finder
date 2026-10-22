import os
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Calculate absolute path to project root .env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
BASE_DIR = ROOT_DIR
ENV_FILE_PATH = os.path.join(ROOT_DIR, ".env")
if not os.path.exists(ENV_FILE_PATH):
    BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ENV_FILE_PATH = os.path.join(BACKEND_DIR, ".env")

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI-Powered Resume Matcher and Job Finder API"
    API_V1_STR: str = "/api/v1"
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost",
        "http://localhost:5000",
        "http://localhost:3000",
        "http://127.0.0.1:5000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        origins = []
        if isinstance(v, str):
            v_str = v.strip()
            if v_str.startswith("[") and v_str.endswith("]"):
                import json
                try:
                    origins = json.loads(v_str)
                except Exception:
                    origins = [i.strip() for i in v_str.split(",") if i.strip()]
            else:
                origins = [i.strip() for i in v_str.split(",") if i.strip()]
        elif isinstance(v, list):
            origins = list(v)

        # Automatically append FRONTEND_URL if specified in production environment
        frontend_url = os.getenv("FRONTEND_URL")
        if frontend_url and frontend_url.strip():
            clean_fe = frontend_url.strip().rstrip("/")
            if not clean_fe.startswith("http://") and not clean_fe.startswith("https://"):
                clean_fe = f"https://{clean_fe}"
            if clean_fe not in origins:
                origins.append(clean_fe)

        return origins

    # Database configuration (MySQL / Local SQLite)
    DATABASE_URL: str = "mysql+pymysql://root:password@localhost:3306/hireagent"

    # Turso Cloud Database Configuration (libSQL)
    TURSO_DATABASE_URL: Optional[str] = None
    TURSO_AUTH_TOKEN: Optional[str] = None

    # JWT Authentication Configuration
    JWT_SECRET_KEY: str = "super_secret_jwt_key_change_in_production_12345"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Firebase Authentication Configuration
    FIREBASE_CREDENTIALS_PATH: Optional[str] = None
    FIREBASE_PROJECT_ID: Optional[str] = None

    # Qdrant Vector DB Configuration
    QDRANT_URL: Optional[str] = None
    QDRANT_API_KEY: Optional[str] = None

    # Groq LLM Configuration (Fallback)
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Gemini LLM Configuration (Primary)
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_MODEL: str = "gemini-3.8-flash"

    # RapidAPI Live Job Search Configuration
    RAPIDAPI_KEY: Optional[str] = None
    RAPIDAPI_HOST: str = "jsearch.p.rapidapi.com"
    RAPIDAPI_JOB_ENDPOINT: str = "https://jsearch.p.rapidapi.com/search"

    # Hugging Face Configuration (Model Download & Verification)
    HF_TOKEN: Optional[str] = None

    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=ENV_FILE_PATH,
        extra="ignore"
    )

settings = Settings()

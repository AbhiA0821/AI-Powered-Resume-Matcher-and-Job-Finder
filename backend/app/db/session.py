from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.core.config import settings

# Base class for declarative SQLAlchemy models
class Base(DeclarativeBase):
    pass

class LibsqlProxyConnection:
    """
    Wrapper around libsql.Connection to provide SQLite compatibility for SQLAlchemy.
    libsql implements DBAPI 2.0 but does not have create_function, which SQLAlchemy's
    SQLite dialect queries during dialect initialization.
    """
    def __init__(self, conn):
        self._conn = conn

    def create_function(self, *args, **kwargs):
        # libsql does not support registering C-level functions; no-op allows SQLAlchemy startup
        pass

    def __getattr__(self, name):
        return getattr(self._conn, name)


def is_turso_configured() -> bool:
    """Return True if Turso Cloud is configured via TURSO_DATABASE_URL or libsql:// URL."""
    return bool(
        settings.TURSO_DATABASE_URL or
        (settings.DATABASE_URL and settings.DATABASE_URL.startswith("libsql://"))
    )


# Configure SQLAlchemy engine:
# Priority 1: Turso Cloud Database (libsql://)
# Priority 2: Local SQLite (sqlite://)
# Priority 3: Standard relational database (e.g., MySQL, PostgreSQL)
if is_turso_configured():
    import libsql

    turso_url = settings.TURSO_DATABASE_URL or settings.DATABASE_URL
    turso_token = settings.TURSO_AUTH_TOKEN or ""

    def _create_turso_connection():
        raw_conn = libsql.connect(
            database=turso_url,
            auth_token=turso_token,
            _check_same_thread=False
        )
        return LibsqlProxyConnection(raw_conn)

    engine = create_engine(
        "sqlite://",
        creator=_create_turso_connection,
        pool_pre_ping=True,
        pool_recycle=300,
        echo=False
    )
elif settings.DATABASE_URL and settings.DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        echo=False
    )
else:
    # Standard MySQL / PostgreSQL engine
    engine = create_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False
    )

# Session factory for creating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# FastAPI dependency to obtain a DB session per request
def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Safely and idempotently creates missing database tables without altering or dropping
    existing tables or data. Uses checkfirst=True to guarantee non-destructive execution.
    """
    from app.db import base  # noqa: F401
    Base.metadata.create_all(bind=engine, checkfirst=True)


if __name__ == "__main__":
    init_db()
    print("Database tables verified/created successfully.")

import os
import pytest
from app.db.turso_verifier import sanitize_url, sanitize_error, verify_turso_connection
from app.db.session import engine, SessionLocal, is_turso_configured
from app.models.user import User


def test_sanitize_url():
    """Verify that sensitive credentials are never displayed in connection URLs."""
    raw_with_pass = "mysql+pymysql://root:supersecret123@db.example.com:3306/production"
    sanitized = sanitize_url(raw_with_pass)
    assert "supersecret123" not in sanitized
    assert "root@" not in sanitized
    assert "db.example.com:3306" in sanitized

    turso_url = "libsql://resumematcher-abhi0821.aws-ap-south-1.turso.io"
    assert sanitize_url(turso_url) == "libsql://resumematcher-abhi0821.aws-ap-south-1.turso.io"


def test_sanitize_error():
    """Verify that tokens and JWT strings are completely scrubbed from error logs."""
    sample_token = "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.dummy_token_payload.signature_xyz"
    exc = Exception(f"Failed to connect using token {sample_token} to remote host")
    clean = sanitize_error(exc, sample_token)
    assert sample_token not in clean
    assert "[REDACTED_AUTH_TOKEN]" in clean or "[REDACTED_TOKEN]" in clean


def test_turso_connection_verification():
    """Verify live connectivity, read-only SELECT 1 execution, and schema discovery."""
    from app.core.config import settings

    turso_url = settings.TURSO_DATABASE_URL or os.getenv("TURSO_DATABASE_URL")
    turso_token = settings.TURSO_AUTH_TOKEN or os.getenv("TURSO_AUTH_TOKEN")

    if not turso_url or not turso_token:
        pytest.skip("TURSO_DATABASE_URL or TURSO_AUTH_TOKEN not set in test environment")

    result = verify_turso_connection(database_url=turso_url, auth_token=turso_token)

    assert result["success"] is True, f"Turso connection failed: {result['error']}"
    assert result["error"] is None
    assert result["latency_ms"] is not None
    assert isinstance(result["tables"], list)

    # Verify our required application tables exist
    required_tables = {"users", "resumes", "user_profiles", "job_preferences"}
    assert required_tables.issubset(set(result["tables"])), (
        f"Missing application tables in Turso. Found: {result['tables']}"
    )

    # Ensure no secret token appears in the report URL
    assert turso_token not in result["url"]


def test_turso_orm_crud_and_transaction():
    """Verify SQLAlchemy ORM insert, read, and delete operations against the database."""
    if not is_turso_configured():
        pytest.skip("Turso is not configured as the active engine")

    session = SessionLocal()
    test_email = "test_turso_verification_runner@example.com"

    try:
        # Cleanup any existing test artifact
        existing = session.query(User).filter_by(email=test_email).first()
        if existing:
            session.delete(existing)
            session.commit()

        # 1. CREATE
        new_user = User(
            name="Turso Verification Runner",
            email=test_email,
            password_hash="verified_hash_123"
        )
        session.add(new_user)
        session.commit()
        assert new_user.id is not None

        # 2. READ
        fetched = session.query(User).filter_by(email=test_email).first()
        assert fetched is not None
        assert fetched.name == "Turso Verification Runner"

        # 3. UPDATE
        fetched.name = "Turso Verification Runner Updated"
        session.commit()

        re_fetched = session.query(User).filter_by(email=test_email).first()
        assert re_fetched.name == "Turso Verification Runner Updated"

        # 4. DELETE
        session.delete(re_fetched)
        session.commit()

        deleted = session.query(User).filter_by(email=test_email).first()
        assert deleted is None
    finally:
        session.close()

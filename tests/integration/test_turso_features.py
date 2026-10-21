import pytest
from app.db.session import SessionLocal, is_turso_configured
from app.models.user import User
from app.models.resume import Resume
from app.models.profile import UserProfile
from app.models.preferences import JobPreferences
from app.services.auth import register_user, authenticate_user
from app.schemas.auth import UserRegister, UserLogin


def test_turso_user_registration_and_authentication():
    """Verify user registration, hashing, and authentication against the live database."""
    session = SessionLocal()
    test_email = "turso_auth_test_user@example.com"
    test_password = "SecurePassword123!"

    try:
        # Pre-cleanup in case of aborted runs
        existing = session.query(User).filter_by(email=test_email).first()
        if existing:
            session.delete(existing)
            session.commit()

        # 1. Register User
        reg_payload = UserRegister(
            email=test_email,
            password=test_password,
            name="Turso Test Candidate"
        )
        created_user = register_user(session, reg_payload)

        assert created_user.id is not None
        assert created_user.email == test_email
        assert created_user.name == "Turso Test Candidate"

        # 2. Authenticate User (Login)
        login_payload = UserLogin(
            email=test_email,
            password=test_password
        )
        token = authenticate_user(session, login_payload)
        assert token is not None
        assert len(token) > 20
        assert token.count(".") == 2  # Standard JWT format

        # 3. Clean up user
        db_user = session.query(User).filter_by(email=test_email).first()
        if db_user:
            session.delete(db_user)
            session.commit()

    finally:
        session.close()


def test_turso_resume_metadata_and_profile_persistence():
    """Verify resume metadata and user profile persistence against the live database."""
    session = SessionLocal()
    test_email = "turso_resume_test_user@example.com"

    try:
        # Pre-cleanup
        existing = session.query(User).filter_by(email=test_email).first()
        if existing:
            session.delete(existing)
            session.commit()

        # 1. Create Base User
        user = User(
            name="Jane Turso",
            email=test_email,
            password_hash="fakehash999"
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # 2. Persist Resume Metadata
        resume = Resume(
            user_id=user.id,
            filename="jane_turso_resume.pdf",
            file_path="/uploads/jane_turso_resume.pdf",
            extracted_text="Experienced Full Stack Python & Machine Learning Engineer with AWS expertise."
        )
        session.add(resume)

        # 3. Persist User Profile
        profile = UserProfile(
            user_id=user.id,
            target_role="Senior AI Engineer",
            experience_level="Senior",
            preferred_locations="San Francisco, CA; Remote",
            work_mode_preference="Remote",
            employment_type="Full-Time",
            min_match_score=80.0
        )
        session.add(profile)

        # 4. Persist Job Preferences
        preferences = JobPreferences(
            user_id=user.id,
            auto_apply_enabled=False,
            daily_apply_limit=10,
            desired_job_titles="AI Engineer, Staff Engineer",
            preferred_industries="Technology, AI, Healthcare",
            min_salary=150000.0
        )
        session.add(preferences)
        session.commit()

        # 5. Verify Persistence & Relationships
        reloaded_user = session.query(User).filter_by(id=user.id).first()
        assert reloaded_user is not None
        assert len(reloaded_user.resumes) == 1
        assert reloaded_user.resumes[0].filename == "jane_turso_resume.pdf"
        assert "Machine Learning Engineer" in reloaded_user.resumes[0].extracted_text
        assert reloaded_user.profile.target_role == "Senior AI Engineer"
        assert reloaded_user.preferences.min_salary == 150000.0

        # 6. Cascade cleanup
        session.delete(reloaded_user)
        session.commit()

        # Confirm cascade deletion of child records
        assert session.query(Resume).filter_by(user_id=user.id).first() is None
        assert session.query(UserProfile).filter_by(user_id=user.id).first() is None
        assert session.query(JobPreferences).filter_by(user_id=user.id).first() is None

    finally:
        session.close()

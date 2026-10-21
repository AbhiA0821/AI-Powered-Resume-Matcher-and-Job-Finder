# Import all SQLAlchemy models here for metadata discovery
from app.db.session import Base
from app.models.user import User
from app.models.resume import Resume
from app.models.profile import UserProfile
from app.models.preferences import JobPreferences

__all__ = ["Base", "User", "Resume", "UserProfile", "JobPreferences"]

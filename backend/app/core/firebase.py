import os
from typing import Dict, Any, Optional
import firebase_admin
from firebase_admin import credentials, auth
from app.core.config import settings

def get_firebase_app():
    """Initializes Firebase Admin SDK if not already initialized."""
    if not firebase_admin._apps:
        cred_path = settings.FIREBASE_CREDENTIALS_PATH
        resolved_path = None
        if cred_path:
            if os.path.exists(cred_path):
                resolved_path = cred_path
            else:
                # Check relative to project root
                from app.core.config import ROOT_DIR
                candidate = os.path.join(ROOT_DIR, cred_path)
                if os.path.exists(candidate):
                    resolved_path = candidate

        if resolved_path:
            cred = credentials.Certificate(resolved_path)
            firebase_admin.initialize_app(cred)
        else:
            # Do NOT call credentials.ApplicationDefault() here without local ADC;
            # on non-GCP machines it hangs for 13+ seconds querying 169.254.169.254.
            project_id = getattr(settings, "FIREBASE_PROJECT_ID", None) or os.getenv("FIREBASE_PROJECT_ID")
            if project_id:
                firebase_admin.initialize_app(options={"projectId": project_id})
            else:
                firebase_admin.initialize_app()
    return firebase_admin.get_app()

def verify_firebase_id_token(id_token: str) -> Dict[str, Any]:
    """Verifies Firebase ID token.
    Uses Firebase Admin SDK when a service account certificate is configured.
    Otherwise uses Google's public x509 certs for the configured project ID.
    Returns decoded token claims dictionary or raises Exception if invalid/unconfigured."""
    app = get_firebase_app()
    
    # 1. If service account credentials were provided, use Firebase Admin's verify_id_token
    cred_path = settings.FIREBASE_CREDENTIALS_PATH
    if cred_path and getattr(app, "credential", None) and hasattr(app.credential, "signer"):
        return auth.verify_id_token(id_token, app=app)
    
    # 2. Otherwise verify using Google's official public x509 cert verification for the project
    project_id = getattr(settings, "FIREBASE_PROJECT_ID", None) or os.getenv("FIREBASE_PROJECT_ID") or getattr(app, "project_id", None)
    if not project_id:
        raise ValueError("FIREBASE_PROJECT_ID or a service account credential is required to verify Firebase ID tokens.")
        
    from google.oauth2 import id_token as google_id_token
    from google.auth.transport import requests as google_requests
    
    # Verifies token signature with Google's public keys, validates audience, expiry, and issuer
    claims = google_id_token.verify_firebase_token(
        id_token,
        google_requests.Request(),
        audience=project_id
    )
    if not claims:
        raise ValueError("Invalid or expired Firebase ID token.")
        
    # Standardize claims to ensure 'uid' is available
    if "uid" not in claims and "sub" in claims:
        claims["uid"] = claims["sub"]
        
    return claims

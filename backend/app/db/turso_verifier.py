"""
Turso Cloud Database Connectivity & Schema Verifier.

Safe verification utility that:
- Connects to the configured Turso database using credentials from environment variables.
- Executes a read-only test query (SELECT 1).
- Inspects existing application tables.
- Sanitizes outputs to ensure no tokens or credentials are leaked in logs or error messages.
- Closes connections and cursors reliably in finally blocks.
"""

import os
import sys
import time
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

# Ensure backend directory is in path when run directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings


def sanitize_url(url: Optional[str]) -> str:
    """Mask any embedded passwords, tokens, or query params from a database URL."""
    if not url:
        return "[NOT_CONFIGURED]"
    try:
        parsed = urlparse(url)
        # Strip password if present in netloc
        netloc = parsed.hostname or parsed.netloc
        if parsed.port:
            netloc = f"{netloc}:{parsed.port}"
        return f"{parsed.scheme}://{netloc}{parsed.path}"
    except Exception:
        return "[MASKED_URL]"


def sanitize_error(exc: Exception, auth_token: Optional[str] = None) -> str:
    """Strip any secret tokens from the error message."""
    msg = str(exc)
    if auth_token and len(auth_token) > 8:
        msg = msg.replace(auth_token, "[REDACTED_AUTH_TOKEN]")
    # Also strip common JWT fragments if visible
    if "eyJ" in msg:
        import re
        msg = re.sub(r'eyJ[a-zA-Z0-9_\-\.]+', '[REDACTED_TOKEN]', msg)
    return msg


def verify_turso_connection(
    database_url: Optional[str] = None,
    auth_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Connect to Turso, execute a read-only query, inspect schema tables,
    and return a structured sanitized report.
    """
    url = database_url or settings.TURSO_DATABASE_URL or settings.DATABASE_URL
    token = auth_token or settings.TURSO_AUTH_TOKEN or os.getenv("TURSO_AUTH_TOKEN", "")

    sanitized_host = sanitize_url(url)

    if not url or not (url.startswith("libsql://") or "turso.io" in url):
        return {
            "success": False,
            "error": "TURSO_DATABASE_URL is not configured with a valid libsql:// or turso.io endpoint.",
            "url": sanitized_host,
            "tables": [],
            "latency_ms": None,
        }

    try:
        import libsql
    except ImportError as e:
        return {
            "success": False,
            "error": "The 'libsql' driver is not installed in the current environment.",
            "url": sanitized_host,
            "tables": [],
            "latency_ms": None,
        }

    conn = None
    cursor = None
    start_time = time.perf_counter()

    try:
        # Establish connection with thread-safety flag
        conn = libsql.connect(
            database=url,
            auth_token=token,
            _check_same_thread=False
        )
        cursor = conn.cursor()

        # Execute simple read-only query
        cursor.execute("SELECT 1;")
        res = cursor.fetchone()
        if not res or res[0] != 1:
            raise RuntimeError(f"Unexpected query result: {res}")

        # Query existing tables in Turso SQLite master
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall() if not row[0].startswith("sqlite_")]

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        required_tables = {"users", "resumes", "user_profiles", "job_preferences"}
        missing_tables = sorted(list(required_tables - set(tables)))

        return {
            "success": True,
            "message": "Successfully connected to Turso Cloud Database and executed read-only test query.",
            "url": sanitized_host,
            "tables": tables,
            "required_tables_present": len(missing_tables) == 0,
            "missing_tables": missing_tables,
            "latency_ms": latency_ms,
            "error": None,
        }

    except Exception as exc:
        sanitized_err = sanitize_error(exc, token)
        return {
            "success": False,
            "error": sanitized_err,
            "url": sanitized_host,
            "tables": [],
            "latency_ms": None,
        }
    finally:
        if cursor:
            try:
                cursor.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass


if __name__ == "__main__":
    result = verify_turso_connection()
    print("=" * 60)
    print("AI-Powered Resume Matcher and Job Finder — Turso Connection Test")
    print("=" * 60)
    print(f"Status:     {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"Target URL: {result['url']}")
    if result["success"]:
        print(f"Latency:    {result['latency_ms']} ms")
        print(f"Tables:     {result['tables']}")
        if result.get("required_tables_present"):
            print("Schema:     All required application tables are present.")
        else:
            print(f"Schema:     Missing tables: {result.get('missing_tables')} (Can be provisioned safely with `python -m app.db.session`)")
    else:
        print(f"Error:      {result['error']}")
    print("=" * 60)

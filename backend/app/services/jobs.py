import os
import sqlite3
from typing import List, Optional, Dict, Any

# Path calculation to root workspace data/resume_matcher.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DB_PATH = os.environ.get("SQLITE_DB_PATH") or os.path.join(BASE_DIR, "data", "resume_matcher.db")

import csv

def _ensure_jobs_table_initialized() -> None:
    """Ensures the jobs table exists and is populated in the local SQLite catalog."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='jobs'")
        has_table = cursor.fetchone() is not None
        has_rows = False
        if has_table:
            cursor.execute("SELECT COUNT(*) FROM jobs")
            cnt = cursor.fetchone()[0]
            has_rows = cnt > 0

        if not has_table or not has_rows:
            csv_path = os.path.join(BASE_DIR, "data", "job_descriptions.csv")
            if os.path.exists(csv_path):
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS jobs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        job_title TEXT,
                        company TEXT,
                        required_skills TEXT,
                        job_role TEXT,
                        experience_level TEXT,
                        description TEXT,
                        application_url TEXT,
                        is_available INTEGER DEFAULT 1
                    )
                """)
                with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                    reader = csv.DictReader(f)
                    rows_to_insert = []
                    for idx, row in enumerate(reader, start=1):
                        app_url = row.get("application_url", "").strip() or "Application link unavailable"
                        rows_to_insert.append((
                            idx,
                            row.get("Job Title", "").strip(),
                            row.get("Company", "").strip(),
                            row.get("skills", "").strip(),
                            row.get("Role", "").strip(),
                            row.get("Experience", "").strip(),
                            row.get("Job Description", "").strip(),
                            app_url,
                            1 if app_url != "Application link unavailable" else 0
                        ))
                    if rows_to_insert:
                        cursor.executemany("""
                            INSERT OR REPLACE INTO jobs (
                                id, job_title, company, required_skills, job_role, experience_level, description, application_url, is_available
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, rows_to_insert)
                conn.commit()
        conn.close()
    except Exception as e:
        print(f"Error ensuring jobs table initialized: {e}")

def get_jobs_from_db(
    search: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """Retrieves jobs from SQLite database data/resume_matcher.db with search and filtering support."""
    _ensure_jobs_table_initialized()

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        base_query = "FROM jobs WHERE 1=1"
        params: List[Any] = []

        if search and search.strip():
            term = f"%{search.strip()}%"
            base_query += " AND (job_title LIKE ? OR required_skills LIKE ? OR description LIKE ?)"
            params.extend([term, term, term])

        if role and role.strip():
            roles_list = [r.strip() for r in role.split(",") if r.strip()]
            if roles_list:
                role_conditions = " OR ".join(["job_title LIKE ? OR job_role LIKE ?" for _ in roles_list])
                base_query += f" AND ({role_conditions})"
                for r in roles_list:
                    r_term = f"%{r}%"
                    params.extend([r_term, r_term])

        # Count total matching rows
        count_sql = f"SELECT COUNT(*) {base_query}"
        cursor.execute(count_sql, params)
        count_row = cursor.fetchone()
        total = count_row[0] if count_row else 0

        # Fetch paginated rows
        select_sql = f"SELECT id, job_title, company, required_skills, experience_level, job_role, description, application_url, is_available {base_query} LIMIT ? OFFSET ?"
        fetch_params = params + [limit, offset]
        cursor.execute(select_sql, fetch_params)
        rows = cursor.fetchall()
        conn.close()

        jobs = []
        for r in rows:
            d = dict(r)
            app_url = d.get("application_url")
            if not app_url or not str(app_url).strip() or "job/" in str(app_url).lower():
                d["application_url"] = "Application link unavailable"
                d["is_available"] = False
            else:
                d["is_available"] = bool(d.get("is_available", 0))
            jobs.append(d)
        return {"total": total, "jobs": jobs}
    except Exception as e:
        print(f"Error querying job database: {e}")
        return {"total": 0, "jobs": []}

def get_job_by_id_from_db(job_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single job by id from SQLite database."""
    _ensure_jobs_table_initialized()
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, job_title, company, required_skills, experience_level, job_role, description, application_url, is_available FROM jobs WHERE id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        conn.close()
        if row:
            d = dict(row)
            app_url = d.get("application_url")
            if not app_url or not str(app_url).strip() or "job/" in str(app_url).lower():
                d["application_url"] = "Application link unavailable"
                d["is_available"] = False
            else:
                d["is_available"] = bool(d.get("is_available", 0))
            return d
        return None
    except Exception as e:
        print(f"Error getting job by id {job_id}: {e}")
        return None

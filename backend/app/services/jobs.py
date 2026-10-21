import os
import sqlite3
from typing import List, Optional, Dict, Any

# Path calculation to root workspace data/resume_matcher.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DB_PATH = os.environ.get("SQLITE_DB_PATH") or os.path.join(BASE_DIR, "data", "resume_matcher.db")

def get_jobs_from_db(
    search: Optional[str] = None,
    role: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
) -> Dict[str, Any]:
    """Retrieves jobs from SQLite database data/resume_matcher.db with search and filtering support."""
    if not os.path.exists(DB_PATH):
        csv_path = os.path.join(BASE_DIR, "data", "job_descriptions.csv")
        if os.path.exists(csv_path):
            try:
                import pandas as pd
                os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
                df = pd.read_csv(csv_path, nrows=500, on_bad_lines='skip', engine='python')
                df_clean = pd.DataFrame()
                df_clean['job_title'] = df['Job Title']
                df_clean['company'] = df['Company']
                df_clean['required_skills'] = df['skills']
                df_clean['job_role'] = df['Role']
                df_clean['experience_level'] = df['Experience']
                df_clean['description'] = df['Job Description']
                if 'application_url' in df.columns:
                    df_clean['application_url'] = df['application_url'].fillna("Application link unavailable")
                else:
                    df_clean['application_url'] = "Application link unavailable"
                df_clean.insert(0, 'id', range(1, len(df_clean) + 1))
                init_conn = sqlite3.connect(DB_PATH)
                df_clean.to_sql('jobs', init_conn, if_exists='replace', index=False)
                init_conn.close()
            except Exception:
                return {"total": 0, "jobs": []}
        else:
            return {"total": 0, "jobs": []}

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

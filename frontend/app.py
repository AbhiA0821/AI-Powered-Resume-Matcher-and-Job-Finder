import os
import sys
import time
from typing import Dict, Any, List, Optional
import jwt
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from dotenv import load_dotenv

# Determine project root and load environment variables from .env
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# Ensure frontend directory is in sys.path
FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))
if FRONTEND_DIR not in sys.path:
    sys.path.append(FRONTEND_DIR)

try:
    from frontend import api_client
except ImportError:
    import api_client

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "hireagent_flask_secret_key_987654321")

def get_session_token() -> Optional[str]:
    """Retrieves session JWT, verifying it is not expired. If expired or invalid, clears session and returns None."""
    token = session.get("token")
    if not token:
        return None
    try:
        if isinstance(token, str) and token.count(".") == 2:
            payload = jwt.decode(token, options={"verify_signature": False})
            exp = payload.get("exp")
            if exp and exp < time.time():
                session.clear()
                return None
    except Exception:
        session.clear()
        return None
    return token

def is_auth_error(err: Optional[str]) -> bool:
    """Checks if an error string indicates an authentication / token expiration failure."""
    if not err:
        return False
    e = str(err).lower()
    return "invalid or expired" in e or "unauthorized" in e or "invalid token" in e

@app.context_processor
def inject_user():
    token = get_session_token()
    return {
        "current_user": session.get("user") if token else None,
        "logged_in": bool(token)
    }

@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "hireagent-frontend"}), 200

# ── 1. Landing Page ──
@app.route("/")
def index():
    if get_session_token():
        return redirect(url_for("dashboard"))
    return render_template("landing.html")

def get_firebase_web_config() -> Dict[str, str]:
    """Returns public Firebase Web SDK configuration for client-side Google Sign-In."""
    return {
        "apiKey": os.getenv("FIREBASE_API_KEY", ""),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN", ""),
        "projectId": os.getenv("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.getenv("FIREBASE_APP_ID", ""),
    }

# ── 2. Login Page ──
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET" and get_session_token():
        return redirect(url_for("dashboard"))
    fb_config = get_firebase_web_config()
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not email or not password:
            flash("Please enter both email and password.", "danger")
            return render_template("login.html", firebase_config=fb_config)

        token, err = api_client.login_user(email, password)
        if token:
            session["token"] = token
            user_data, _ = api_client.get_current_user(token)
            session["user"] = user_data or {"email": email, "name": email.split("@")[0]}
            flash("Signed in successfully!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash(f"Sign in failed: {err}", "danger")

    return render_template("login.html", firebase_config=fb_config)

@app.route("/login/firebase", methods=["POST"])
def login_firebase():
    """Receives Firebase ID token from client Google Sign-In, verifies via backend, sets Flask session."""
    data = request.get_json(silent=True) or {}
    id_token = data.get("id_token")
    if not id_token:
        return jsonify({"ok": False, "error": "Missing Firebase ID token"}), 400

    token, err = api_client.login_firebase_user(id_token)
    if token:
        session["token"] = token
        user_data, _ = api_client.get_current_user(token)
        session["user"] = user_data or {"email": "google_user", "name": "Google User"}
        flash("Signed in with Google successfully!", "success")
        return jsonify({"ok": True, "redirect": url_for("dashboard")})
    return jsonify({"ok": False, "error": err or "Google authentication failed"}), 401

# ── 3. Register Page ──
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        user_res, err = api_client.register_user(name, email, password)
        if user_res:
            token, _ = api_client.login_user(email, password)
            if token:
                session["token"] = token
                session["user"] = {"name": name, "email": email}
                flash("Account created and signed in!", "success")
                return redirect(url_for("dashboard"))
        flash(f"Registration failed: {err}", "danger")

    return render_template("register.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Signed out successfully.", "info")
    return redirect(url_for("login"))

# ── 4. Dashboard Page ──
# ── 4. Dashboard / Resume Matcher Page ──
@app.route("/matcher", methods=["GET", "POST"])
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    token = get_session_token()
    if not token:
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({"ok": False, "error": "Your session has expired. Please sign in again.", "redirect": url_for("login")}), 401
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        is_ajax = request.headers.get("X-Requested-With") == "XMLHttpRequest"
        target_role = request.form.get("target_role", "").strip() or None
        preferred_location = request.form.get("preferred_location", "").strip() or None
        preferred_company = request.form.get("preferred_company", "").strip() or None
        resume_id_str = request.form.get("resume_id", "").strip()
        resume_id = int(resume_id_str) if resume_id_str.isdigit() else None
        action = request.form.get("action", "")

        # Case 1: File Upload
        if "resume_file" in request.files and request.files["resume_file"].filename != "":
            file = request.files["resume_file"]
            file_bytes = file.read()
            res, err = api_client.upload_resume(
                token,
                file_bytes,
                file.filename,
                target_role=target_role,
                preferred_location=preferred_location,
                preferred_company=preferred_company
            )
            if res:
                if is_ajax:
                    return jsonify({"ok": True, "filename": file.filename})
                flash(f"Resume '{file.filename}' analyzed and matched against jobs!", "success")
                return redirect(url_for("dashboard"))
            else:
                if is_auth_error(err):
                    session.clear()
                    if is_ajax:
                        return jsonify({"ok": False, "error": "Your session has expired. Please sign in again.", "redirect": url_for("login")}), 401
                    flash("Your session has expired. Please sign in again.", "warning")
                    return redirect(url_for("login"))
                if is_ajax:
                    return jsonify({"ok": False, "error": err or "Upload failed"}), 400
                flash(f"Upload failed: {err}", "danger")
                return redirect(url_for("dashboard"))

        # Case 2: Match Existing Resume / Apply Preferences
        elif action in ["find_matches", "match_existing"] or target_role or preferred_location or preferred_company or resume_id:
            _, err = api_client.rerun_dashboard_pipeline(
                token,
                target_role=target_role,
                preferred_location=preferred_location,
                preferred_company=preferred_company,
                resume_id=resume_id
            )
            if err:
                if is_auth_error(err):
                    session.clear()
                    if is_ajax:
                        return jsonify({"ok": False, "error": "Your session has expired. Please sign in again.", "redirect": url_for("login")}), 401
                    flash("Your session has expired. Please sign in again.", "warning")
                    return redirect(url_for("login"))
                if is_ajax:
                    return jsonify({"ok": False, "error": err}), 400
                flash(f"Matching failed: {err}", "danger")
            else:
                if is_ajax:
                    return jsonify({"ok": True})
                flash("Jobs matched successfully against your profile and preferences!", "success")
            return redirect(url_for("dashboard"))

        else:
            if is_ajax:
                return jsonify({"ok": False, "error": "Please upload a resume or select preferences."}), 400
            flash("Please upload a resume to find matching jobs.", "warning")
            return redirect(url_for("dashboard"))

    # Fetch user's aggregated overview, resumes and profile
    resumes, err_res = api_client.get_user_resumes(token)
    profile, err_prof = api_client.get_user_profile(token)
    quota = api_client.get_jsearch_quota()
    overview, err_over = api_client.get_dashboard_overview(token)

    # If the user endpoint returned an authentication error, session is stale
    if is_auth_error(err_res):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    if not overview:
        has_resume = bool(resumes)
        overview = {
            "resume": {
                "exists": has_resume,
                "filename": resumes[0].get("filename") if (has_resume and isinstance(resumes[0], dict)) else getattr(resumes[0], "filename", None) if has_resume else None,
                "status": "ready" if has_resume else "none"
            },
            "profile": {
                "primary_role": profile.get("target_role") if isinstance(profile, dict) else getattr(profile, "target_role", None),
                "experience_tier": profile.get("experience_level") if isinstance(profile, dict) else getattr(profile, "experience_level", None),
                "skills": profile.get("skills", []) if isinstance(profile, dict) else [],
                "skill_count": len(profile.get("skills", [])) if isinstance(profile, dict) else 0
            },
            "discovery": {
                "total_jobs_analyzed": 0,
                "jobs_retrieved": 0,
                "relevant_jobs": 0,
                "matched_jobs": 0,
                "source": "local",
                "min_score_threshold": 65.0
            },
            "matches": [],
            "skill_gap": {
                "matched_skills": [],
                "missing_skills_ranked": []
            },
            "recommendations": {
                "available": False,
                "items": []
            },
            "assistant": {
                "available": True
            },
            "pipeline": [
                {"stage": "Resume Uploaded", "status": "completed" if has_resume else "not_started", "detail": "Resume uploaded" if has_resume else "No resume uploaded yet"},
                {"stage": "Resume Parsed", "status": "completed" if has_resume else "not_started", "detail": "Parsed" if has_resume else "Awaiting upload"},
                {"stage": "Skills Extracted", "status": "completed" if (profile and profile.get("skills")) else "not_started", "detail": "Skills extracted" if (profile and profile.get("skills")) else "Awaiting parsing"},
                {"stage": "Jobs Retrieved", "status": "not_started", "detail": "Awaiting matching"},
                {"stage": "Jobs Matched", "status": "not_started", "detail": "Awaiting matching"},
                {"stage": "Skill Gaps Identified", "status": "not_started", "detail": "Awaiting matches"},
                {"stage": "Recommendations Generated", "status": "not_started", "detail": "Awaiting matches"},
                {"stage": "AI Assistant Available", "status": "completed", "detail": "Online"}
            ]
        }

    return render_template(
        "dashboard.html",
        overview=overview or {},
        resumes=resumes or [],
        profile=profile or {},
        quota=quota or {}
    )

@app.route("/dashboard/poll", methods=["GET"])
def dashboard_poll():
    token = get_session_token()
    if not token:
        return jsonify({"ok": False, "error": "Session expired", "redirect": url_for("login")}), 401
    overview, err = api_client.get_dashboard_overview(token)
    if err:
        if is_auth_error(err):
            session.clear()
            return jsonify({"ok": False, "error": "Session expired", "redirect": url_for("login")}), 401
        return jsonify({"error": err}), 400
    return jsonify(overview or {})

@app.route("/dashboard/rerun", methods=["POST"])
def dashboard_rerun():
    token = get_session_token()
    if not token:
        return jsonify({"ok": False, "error": "Session expired", "redirect": url_for("login")}), 401
    payload = request.get_json(silent=True) or {}
    target_role = payload.get("target_role")
    preferred_location = payload.get("preferred_location")
    preferred_company = payload.get("preferred_company")
    resume_id = payload.get("resume_id")
    res_id_int = int(resume_id) if (resume_id and str(resume_id).isdigit()) else None
    overview, err = api_client.rerun_dashboard_pipeline(
        token,
        target_role=target_role or None,
        preferred_location=preferred_location or None,
        preferred_company=preferred_company or None,
        resume_id=res_id_int
    )
    if err:
        if is_auth_error(err):
            session.clear()
            return jsonify({"ok": False, "error": "Session expired", "redirect": url_for("login")}), 401
        return jsonify({"error": err}), 400
    return jsonify(overview or {})

# ── History Page ──
@app.route("/history")
def history_page():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    resumes, err = api_client.get_user_resumes(token)
    if is_auth_error(err):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    return render_template("history.html", resumes=resumes or [])

@app.route("/history/delete/<int:resume_id>", methods=["POST"])
def history_delete(resume_id: int):
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    ok, err = api_client.delete_resume(token, resume_id)
    if is_auth_error(err):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    if ok:
        flash("Resume removed from history.", "info")
    else:
        flash(f"Failed to delete resume: {err}", "danger")
    return redirect(url_for("history_page"))


# ── 5. Resume Analysis Page ──
@app.route("/resume", methods=["GET", "POST"])
def resume_page():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    if request.method == "POST":
        if "resume_file" not in request.files:
            flash("No file part provided.", "danger")
            return redirect(url_for("resume_page"))
        file = request.files["resume_file"]
        if file.filename == "":
            flash("No file selected.", "danger")
            return redirect(url_for("resume_page"))

        file_bytes = file.read()
        res, err = api_client.upload_resume(token, file_bytes, file.filename)
        if res:
            flash(f"Resume '{file.filename}' uploaded and parsed successfully!", "success")
            return redirect(url_for("resume_page"))
        else:
            if is_auth_error(err):
                session.clear()
                flash("Your session has expired. Please sign in again.", "warning")
                return redirect(url_for("login"))
            flash(f"Upload failed: {err}", "danger")

    resumes, err_res = api_client.get_user_resumes(token)
    profile, err_prof = api_client.get_user_profile(token)
    if is_auth_error(err_res) or is_auth_error(err_prof):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    latest_resume = resumes[0] if resumes else None

    # Retrieve verified skills
    skills = []
    if profile and profile.get("skills"):
        skills = profile.get("skills")
    elif latest_resume:
        # Fallback to recommendation skills
        rec_data, _ = api_client.get_job_recommendations(token, top_k=1)
        if rec_data:
            skills = rec_data.get("extracted_skills", [])

    return render_template(
        "resume_analysis.html",
        latest_resume=latest_resume,
        profile=profile or {},
        skills=skills
    )

# ── 6. Job Discovery Page ──
@app.route("/jobs")
def jobs():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    search_query = request.args.get("search", "").strip()
    role_filter = request.args.get("role", "").strip()

    # Quota warning check
    quota = api_client.get_jsearch_quota()
    if quota and quota.get("active_warning"):
        warn_key = f"quota_warn_{quota.get('current_month')}_{quota.get('status')}"
        if not session.get(warn_key):
            flash(quota["active_warning"], "warning")
            session[warn_key] = True

    recommendations: List[Dict[str, Any]] = []
    target_role = ""

    if search_query or role_filter:
        # User entered search/filter criteria
        jobs_data, err_jobs = api_client.get_jobs(token, search=search_query or None, role=role_filter or None, limit=20)
        if is_auth_error(err_jobs):
            session.clear()
            flash("Your session has expired. Please sign in again.", "warning")
            return redirect(url_for("login"))
        raw_jobs = jobs_data.get("jobs", []) if jobs_data else []
        for j in raw_jobs:
            req_skills = [s.strip() for s in (j.get("required_skills") or "").split(",") if s.strip()]
            app_url = j.get("application_url") or j.get("canonical_job_url") or j.get("apply_url")
            has_valid_url = bool(app_url and str(app_url).strip().startswith(("http://", "https://")) and app_url != "Application link unavailable")
            is_avail = bool(j.get("is_available", has_valid_url))
            recommendations.append({
                "job_id": str(j.get("job_id") or j.get("external_job_id") or j.get("id")),
                "job_title": j.get("job_title", "Position"),
                "company": j.get("company", "Company"),
                "location": j.get("location", "India"),
                "employment_type": j.get("employment_type", "Full-time"),
                "posted_at": j.get("posted_at"),
                "match_percentage": float(j.get("match_percentage", 50.0)),
                "matched_skills": j.get("matched_skills") or req_skills[:3],
                "missing_skills": j.get("missing_skills") or req_skills[3:6],
                "source": j.get("source") or "Job Search",
                "source_type": j.get("source_type") or "live_api",
                "application_url": app_url if is_avail else "Application link unavailable",
                "apply_url": app_url if is_avail else "Application link unavailable",
                "is_available": is_avail,
                "is_direct": bool(j.get("is_direct", True))
            })
    else:
        # Hybrid matching workflow
        rec_data, err = api_client.get_job_recommendations(token, top_k=10)
        if is_auth_error(err):
            session.clear()
            flash("Your session has expired. Please sign in again.", "warning")
            return redirect(url_for("login"))
        if rec_data and rec_data.get("recommendations"):
            recommendations = rec_data.get("recommendations", [])
            target_role = rec_data.get("target_role", "")
        else:
            # Fallback to catalog / featured jobs if no resume is uploaded yet
            jobs_data, _ = api_client.get_jobs(token, limit=10)
            raw_jobs = jobs_data.get("jobs", []) if jobs_data else []
            for j in raw_jobs:
                req_skills = [s.strip() for s in (j.get("required_skills") or "").split(",") if s.strip()]
                app_url = j.get("application_url") or j.get("canonical_job_url") or j.get("apply_url")
                has_valid_url = bool(app_url and str(app_url).strip().startswith(("http://", "https://")) and app_url != "Application link unavailable")
                is_avail = bool(j.get("is_available", has_valid_url))
                recommendations.append({
                    "job_id": str(j.get("job_id") or j.get("external_job_id") or j.get("id")),
                    "job_title": j.get("job_title", "Position"),
                    "company": j.get("company", "Company"),
                    "location": j.get("location", "India"),
                    "employment_type": j.get("employment_type", "Full-time"),
                    "posted_at": j.get("posted_at"),
                    "match_percentage": float(j.get("match_percentage", 50.0)),
                    "matched_skills": j.get("matched_skills") or req_skills[:3],
                    "missing_skills": j.get("missing_skills") or req_skills[3:6],
                    "source": j.get("source") or "Job Search",
                    "source_type": j.get("source_type") or "live_api",
                    "application_url": app_url if is_avail else "Application link unavailable",
                    "apply_url": app_url if is_avail else "Application link unavailable",
                    "is_available": is_avail,
                    "is_direct": bool(j.get("is_direct", True))
                })

    return render_template(
        "jobs.html",
        recommendations=recommendations,
        target_role=target_role,
        search_query=search_query,
        role_filter=role_filter
    )

# ── 7. Job Details Page ──
@app.route("/jobs/<job_id>")
def job_detail_page(job_id: str):
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    job, err = api_client.get_job_by_id(token, job_id)
    if is_auth_error(err):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    if not job:
        flash(f"Job not found or unavailable.", "warning")
        return redirect(url_for("jobs"))

    return render_template("job_details.html", job=job)

# ── 8. Skill Gap Analysis Page ──
@app.route("/skill-gap")
def skill_gap_page():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    rec_data, err = api_client.get_job_recommendations(token, top_k=10)
    if is_auth_error(err):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    recommendations = rec_data.get("recommendations", []) if rec_data else []
    verified_skills = rec_data.get("extracted_skills", []) if rec_data else []

    # Calculate aggregate missing skills
    missing_freq: Dict[str, int] = {}
    for j in recommendations:
        for m in j.get("missing_skills", []):
            missing_freq[m] = missing_freq.get(m, 0) + 1

    sorted_missing = sorted(missing_freq.keys(), key=lambda k: missing_freq[k], reverse=True)

    return render_template(
        "skill_gap.html",
        recommendations=recommendations,
        verified_skills=verified_skills,
        aggregate_missing=sorted_missing[:12]
    )

# ── 9. Career Recommendations Page ──
@app.route("/career-recommendations", methods=["GET", "POST"])
def career_recommendations_page():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))

    resumes, err_res = api_client.get_user_resumes(token)
    if is_auth_error(err_res):
        session.clear()
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    has_resume = bool(resumes)
    report = session.get("career_insights_report")

    if request.method == "POST":
        if not has_resume:
            flash("Please upload a resume first to generate recommendations.", "warning")
            return redirect(url_for("resume_page"))

        data, err = api_client.get_career_insights(token)
        if is_auth_error(err):
            session.clear()
            flash("Your session has expired. Please sign in again.", "warning")
            return redirect(url_for("login"))
        if data and data.get("ok"):
            report = data.get("report")
            session["career_insights_report"] = report
            flash("Career insights generated successfully!", "success")
        else:
            flash(f"Could not generate insights: {err or 'Service unavailable'}", "danger")

    return render_template(
        "career_insights.html",
        has_resume=has_resume,
        report=report
    )

# ── 10. AI Career Assistant Page & Proxy ──
@app.route("/assistant")
def assistant_page():
    token = get_session_token()
    if not token:
        flash("Your session has expired. Please sign in again.", "warning")
        return redirect(url_for("login"))
    return render_template("assistant.html")

@app.route("/api/proxy/assistant/chat", methods=["POST"])
def assistant_chat_proxy():
    token = get_session_token()
    if not token:
        return jsonify({"ok": False, "error": "Your session has expired. Please sign in again.", "redirect": url_for("login")}), 401

    payload = request.get_json(silent=True) or {}
    message = payload.get("message", "").strip()
    history = payload.get("history", [])

    if not message:
        return jsonify({"ok": False, "error": "Message is required"}), 400

    data, err = api_client.chat_with_assistant(token, message, history)
    if is_auth_error(err):
        session.clear()
        return jsonify({"ok": False, "error": "Your session has expired. Please sign in again.", "redirect": url_for("login")}), 401
    if err or not data:
        return jsonify({"ok": False, "error": err or "Failed to contact assistant service"}), 500

    return jsonify(data)

if __name__ == "__main__":
    # Use stat-based reloader instead of watchdog to prevent Windows API filesystem events
    # inside site-packages (e.g. torch, transformers) from triggering repeated restarts.
    app.run(host="127.0.0.1", port=5000, debug=True, reloader_type="stat")

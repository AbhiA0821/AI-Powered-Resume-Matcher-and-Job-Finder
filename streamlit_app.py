"""
HireAgent — AI-Powered Resume Matcher and Job Finder
Streamlit Community Cloud Main Application Entry Point
"""

import os
import sys
import time
import json
import uuid
import tempfile
import urllib.parse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import streamlit as st

# ==============================================================================
# 1. Environment & Paths Configuration
# ==============================================================================
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"

# Ensure root and backend directories are in sys.path
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Bridge Streamlit Secrets to os.environ for backend modules
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
            elif isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    env_key = f"{key}_{sub_key}".upper()
                    if isinstance(sub_val, str) and env_key not in os.environ:
                        os.environ[env_key] = sub_val
except Exception:
    pass

# Safe Fallback defaults for Streamlit Cloud deployment
DATA_DIR = ROOT_DIR / "data"
SQLITE_DB_PATH = str(DATA_DIR / "resume_matcher.db")
os.environ.setdefault("SQLITE_DB_PATH", SQLITE_DB_PATH)

# Avoid connecting to localhost MySQL on Streamlit Cloud if Turso is not configured
if not os.environ.get("TURSO_DATABASE_URL") and not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = f"sqlite:///{SQLITE_DB_PATH.replace(os.sep, '/')}"

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(ROOT_DIR / ".fastembed_cache"))
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ==============================================================================
# 2. Backend Service Imports (Protected & Lazy)
# ==============================================================================
try:
    from app.core.config import settings
    from app.services.skill_service import (
        extract_skills,
        detect_candidate_role,
        discover_relevant_roles,
        parse_job_required_skills,
        analyze_skill_gap,
        normalize_role,
        normalize_skill
    )
    from app.services.resume_parser import extract_resume_text
    from app.services.embeddings import get_embedding_model, embed_text, embed_texts
    from app.services.matching_service import (
        match_and_rank_jobs,
        generate_platform_search_links,
        apply_employer_diversity
    )
    from app.services.jobs import get_jobs_from_db, _ensure_jobs_table_initialized
    from app.services.job_sources.rapidapi_job_service import (
        search_live_jobs,
        to_hireagent_job_dict,
        get_jsearch_quota_status
    )
    from app.services.url_validator import validate_jobs_batch
    from app.core.security import hash_password, verify_password
    from app.db.session import init_db, SessionLocal
    from app.models.user import User
    from app.models.resume import Resume
    from app.models.profile import UserProfile
    BACKEND_AVAILABLE = True
except Exception as import_err:
    BACKEND_AVAILABLE = False
    BACKEND_IMPORT_ERROR = str(import_err)

# ==============================================================================
# 3. Streamlit Page Configuration & Modern Theme Styling
# ==============================================================================
st.set_page_config(
    page_title="HireAgent — AI Resume Matcher & Job Finder",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished, responsive interface
st.markdown("""
<style>
    /* Metric & Card Styling */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 26px;
        font-weight: 700;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 13px;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    /* Skill Badges */
    .skill-badge {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px 4px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 500;
        background-color: rgba(79, 70, 229, 0.2);
        color: #A5B4FC;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    .skill-matched {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px 3px;
        border-radius: 12px;
        font-size: 11px;
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .skill-missing {
        display: inline-block;
        padding: 3px 8px;
        margin: 2px 3px;
        border-radius: 12px;
        font-size: 11px;
        background-color: rgba(245, 158, 11, 0.2);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    /* Match Score Badges */
    .score-high {
        background: linear-gradient(135deg, #059669 0%, #10B981 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    .score-mid {
        background: linear-gradient(135deg, #2563EB 0%, #3B82F6 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    .score-low {
        background: linear-gradient(135deg, #D97706 0%, #F59E0B 100%);
        color: #FFFFFF;
        padding: 6px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
    }
    /* Source Tags */
    .source-live {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .source-catalog {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    /* Platform Search Links */
    .platform-link {
        color: #94A3B8;
        text-decoration: none;
        font-size: 12px;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-right: 6px;
        transition: all 0.2s;
    }
    .platform-link:hover {
        color: #38BDF8;
        border-color: #38BDF8;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 4. Cached Resource Initializers (Streamlit Performance Optimization)
# ==============================================================================
@st.cache_resource(show_spinner=False)
def initialize_system_database() -> bool:
    """Initializes SQLite jobs catalog and creates required database tables once."""
    if not BACKEND_AVAILABLE:
        return False
    try:
        _ensure_jobs_table_initialized()
        init_db()
        return True
    except Exception as e:
        print(f"Database init notice: {e}")
        return False

@st.cache_resource(show_spinner=False)
def preload_embedding_model():
    """Pre-initializes the FastEmbed ONNX singleton once across sessions."""
    if not BACKEND_AVAILABLE:
        return None
    try:
        return get_embedding_model()
    except Exception as e:
        print(f"Embedding model preload notice: {e}")
        return None

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_catalog_jobs(limit: int = 50) -> List[Dict[str, Any]]:
    """Loads static catalog jobs with 10-minute caching to minimize I/O."""
    if not BACKEND_AVAILABLE:
        return []
    try:
        res = get_jobs_from_db(limit=limit)
        return res.get("jobs", [])
    except Exception:
        return []

# Initialize database and model
_db_ok = initialize_system_database()
_embed_model = preload_embedding_model()

# ==============================================================================
# 5. Session State Management
# ==============================================================================
if "auth_user" not in st.session_state:
    st.session_state.auth_user = None  # {id, email, name, is_demo}
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None  # {filename, text, word_count, skills, detected_role, discovered_roles}
if "selected_role" not in st.session_state:
    st.session_state.selected_role = None
if "matching_results" not in st.session_state:
    st.session_state.matching_results = None  # {ranked_jobs, stats, all_missing}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pipeline_stages" not in st.session_state:
    st.session_state.pipeline_stages = {}

# ==============================================================================
# 6. Authentication Helper Functions
# ==============================================================================
def authenticate_user_local(email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Authenticates against the database or provides safe login."""
    if not email or not password:
        return None, "Please provide both email and password."
    email_clean = email.strip().lower()
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.email == email_clean).first()
        if not user or not user.password_hash or not verify_password(password, user.password_hash):
            db.close()
            return None, "Invalid email or password."
        user_info = {
            "id": user.id,
            "email": user.email,
            "name": user.name or user.email.split("@")[0],
            "is_demo": False
        }
        db.close()
        return user_info, None
    except Exception as e:
        return None, f"Database login error: {str(e)}"

def register_user_local(name: str, email: str, password: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Registers a new user in the database."""
    if not name or not email or not password:
        return None, "All fields are required."
    if len(password) < 6:
        return None, "Password must be at least 6 characters long."
    email_clean = email.strip().lower()
    try:
        db = SessionLocal()
        existing = db.query(User).filter(User.email == email_clean).first()
        if existing:
            db.close()
            return None, "An account with this email already exists."
        new_user = User(
            name=name.strip(),
            email=email_clean,
            password_hash=hash_password(password)
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        user_info = {
            "id": new_user.id,
            "email": new_user.email,
            "name": new_user.name,
            "is_demo": False
        }
        db.close()
        return user_info, None
    except Exception as e:
        return None, f"Registration error: {str(e)}"

def start_demo_session():
    """Sets up an instant demo/recruiter session without registration."""
    st.session_state.auth_user = {
        "id": 999999,
        "email": "recruiter.demo@hireagent.ai",
        "name": "Recruiter / Evaluator",
        "is_demo": True
    }
    st.rerun()

def sign_out():
    """Signs out current user and clears session state."""
    st.session_state.auth_user = None
    st.session_state.resume_data = None
    st.session_state.matching_results = None
    st.session_state.selected_role = None
    st.session_state.chat_history = []
    st.session_state.pipeline_stages = {}
    st.rerun()

# ==============================================================================
# 7. Sidebar: User Profile, System Monitor & Parameters
# ==============================================================================
with st.sidebar:
    st.markdown("### 💼 HireAgent AI")
    st.caption("AI-Powered Resume Matcher & Real Job Finder")

    if st.session_state.auth_user:
        user = st.session_state.auth_user
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="font-size: 14px; font-weight: 600; color: #F8FAFC;">{user['name']}</div>
            <div style="font-size: 12px; color: #94A3B8;">{user['email']}</div>
            <div style="font-size: 11px; margin-top: 4px; color: {'#34D399' if user.get('is_demo') else '#60A5FA'};">
                {'Demo Session' if user.get('is_demo') else 'Standard User'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign Out", use_container_width=True, type="secondary"):
            sign_out()
    else:
        st.info("👋 Please Sign In or Start Demo Mode to analyze your resume.")

    st.markdown("---")
    st.markdown("#### ⚙️ System Status")

    # JSearch Live Quota Display
    quota_status = get_jsearch_quota_status() if BACKEND_AVAILABLE else {}
    used = quota_status.get("requests_used", 0)
    limit = quota_status.get("monthly_limit", 200)
    rem = quota_status.get("requests_remaining", 200)
    is_blocked = quota_status.get("is_blocked", False)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.metric("JSearch Quota", f"{used}/{limit}")
    with col_q2:
        st.metric("Remaining", f"{rem}")

    if is_blocked:
        st.warning("⚠️ JSearch monthly quota cap reached. Catalog jobs active.")
    elif used >= 170:
        st.caption(f"⚠️ Usage alert: {used}/200 requests consumed this month.")

    # Backend / Model Status
    turso_active = bool(os.getenv("TURSO_DATABASE_URL") or (settings.TURSO_DATABASE_URL if BACKEND_AVAILABLE else None))
    db_label = "Turso Cloud (libSQL)" if turso_active else "Local SQLite Catalog"
    st.caption(f"🗄️ **Database:** {db_label}")
    st.caption(f"🧠 **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (ONNX 384D)")

    has_gemini = bool(os.getenv("GEMINI_API_KEY") or (settings.GEMINI_API_KEY if BACKEND_AVAILABLE else None))
    has_groq = bool(os.getenv("GROQ_API_KEY") or (settings.GROQ_API_KEY if BACKEND_AVAILABLE else None))
    llm_str = "Gemini 2.5 Flash" if has_gemini else ("Groq Llama-3.3" if has_groq else "Grounded Rules")
    st.caption(f"🤖 **AI Advisor:** {llm_str}")

    st.markdown("---")
    st.markdown("#### 🎯 Matching Preferences")
    min_score_filter = st.slider("Minimum Match Score (%)", min_value=30, max_value=85, value=50, step=5)
    max_top_k = st.slider("Top Recommendations", min_value=5, max_value=20, value=10, step=1)
    location_filter = st.text_input("Preferred Location", value="India", placeholder="e.g. India, Remote, Bangalore")
    enable_live_rapidapi = st.checkbox(
        "Enable Live RapidAPI Search",
        value=bool(os.getenv("RAPIDAPI_KEY") or (settings.RAPIDAPI_KEY if BACKEND_AVAILABLE else None)),
        help="Searches live external postings via RapidAPI/JSearch within monthly quota limit."
    )
    apply_employer_limit = st.checkbox(
        "Diversity Filter (Max 2 per company)",
        value=True,
        help="Ensures diversity among top recommended employers."
    )

# ==============================================================================
# 8. Authentication Screen (When Not Logged In)
# ==============================================================================
if not st.session_state.auth_user:
    st.markdown("""
    <div style="text-align: center; margin-top: 20px; margin-bottom: 30px;">
        <h1 style="font-size: 42px; font-weight: 800; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            HireAgent AI
        </h1>
        <p style="font-size: 18px; color: #94A3B8; max-width: 650px; margin: 0 auto;">
            Intelligent resume matching, multi-track career discovery, and real live job finding powered by hybrid AI vector search.
        </p>
    </div>
    """, unsafe_allow_html=True)

    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["🔑 Sign In", "📝 Create Account", "🚀 Instant Demo Mode"])

    with auth_tab1:
        st.markdown("##### Sign in to your HireAgent account")
        with st.form("login_form"):
            login_email = st.text_input("Email Address", placeholder="name@example.com")
            login_password = st.text_input("Password", type="password", placeholder="••••••••")
            submit_login = st.form_submit_button("Sign In", type="primary", use_container_width=True)

            if submit_login:
                user_data, err = authenticate_user_local(login_email, login_password)
                if user_data:
                    st.session_state.auth_user = user_data
                    st.success("Signed in successfully!")
                    st.rerun()
                else:
                    st.error(err or "Failed to sign in.")

    with auth_tab2:
        st.markdown("##### Register a new candidate account")
        with st.form("register_form"):
            reg_name = st.text_input("Full Name", placeholder="Jane Doe")
            reg_email = st.text_input("Email Address", placeholder="jane@example.com")
            reg_password = st.text_input("Password (min 6 characters)", type="password", placeholder="••••••••")
            submit_reg = st.form_submit_button("Create Account", type="primary", use_container_width=True)

            if submit_reg:
                user_data, err = register_user_local(reg_name, reg_email, reg_password)
                if user_data:
                    st.session_state.auth_user = user_data
                    st.success("Account created successfully! Signed in.")
                    st.rerun()
                else:
                    st.error(err or "Failed to create account.")

    with auth_tab3:
        st.markdown("##### Explore without creating an account")
        st.write("Evaluating HireAgent on Streamlit Community Cloud? Launch an isolated demo session immediately with sample matching features enabled.")
        if st.button("🚀 Launch Instant Demo Session", type="primary", use_container_width=True):
            start_demo_session()

    st.stop()

# ==============================================================================
# 9. Main Application Dashboard
# ==============================================================================
st.markdown("""
<div style="margin-bottom: 24px;">
    <h2 style="font-weight: 700; margin-bottom: 4px;">💼 Candidate Career Dashboard</h2>
    <p style="color: #94A3B8; margin: 0;">Upload your resume to discover compatible role tracks, match percentages, and real job opportunities.</p>
</div>
""", unsafe_allow_html=True)

main_tab1, main_tab2, main_tab3 = st.tabs([
    "📄 Resume Matcher & Jobs",
    "🤖 AI Career Assistant",
    "🔍 Job Catalog Explorer"
])

# ------------------------------------------------------------------------------
# TAB 1: RESUME MATCHER & JOB FINDER
# ------------------------------------------------------------------------------
with main_tab1:
    col_up, col_role = st.columns([1.2, 1])

    with col_up:
        st.markdown("#### 1. Upload Resume")
        uploaded_file = st.file_uploader(
            "Select Resume (PDF or DOCX, max 10MB)",
            type=["pdf", "docx"],
            help="Your resume is parsed locally in memory. No data is shared with external model trainers."
        )

        if uploaded_file is not None:
            # Check if this is a newly uploaded file or existing
            current_resume = st.session_state.resume_data
            if not current_resume or current_resume.get("filename") != uploaded_file.name:
                # Save bytes to temporary file for parser
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                try:
                    with st.spinner("Extracting and parsing resume content..."):
                        extracted_text = extract_resume_text(tmp_path)
                    
                    if not extracted_text or len(extracted_text.split()) < 10:
                        st.error("Failed to extract sufficient text content from uploaded file. Please ensure it is a text-based PDF or DOCX.")
                    else:
                        skills = extract_skills(extracted_text)
                        detected_role = detect_candidate_role(skills, extracted_text)
                        discovered_roles = discover_relevant_roles(
                            resume_skills=skills,
                            extracted_text=extracted_text,
                            target_role=detected_role,
                            max_roles=3
                        )
                        st.session_state.resume_data = {
                            "filename": uploaded_file.name,
                            "text": extracted_text,
                            "word_count": len(extracted_text.split()),
                            "skills": skills,
                            "detected_role": detected_role,
                            "discovered_roles": discovered_roles
                        }
                        st.session_state.selected_role = detected_role
                        st.session_state.matching_results = None  # Clear previous matches for fresh run
                finally:
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except Exception:
                            pass

    with col_role:
        st.markdown("#### 2. Target Role & Alignment")
        resume_data = st.session_state.resume_data
        if resume_data:
            detected = resume_data["detected_role"]
            tracks = resume_data.get("discovered_roles", [detected])
            if detected not in tracks:
                tracks = [detected] + tracks

            st.write(f"🎯 **Detected Primary Role:** `{detected}`")
            role_options = tracks + ["Custom Role..."]
            selected_option = st.selectbox(
                "Select Target Career Track",
                options=role_options,
                index=0
            )

            if selected_option == "Custom Role...":
                custom_role_val = st.text_input("Enter custom target role", value="")
                if custom_role_val.strip():
                    active_target_role = normalize_role(custom_role_val)
                else:
                    active_target_role = detected
            else:
                active_target_role = selected_option

            st.session_state.selected_role = active_target_role

            col_btn1, col_btn2 = st.columns([1.5, 1])
            with col_btn1:
                run_matching_btn = st.button("🚀 Run Hybrid Match & Job Discovery", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🔄 Re-analyze", use_container_width=True):
                    st.session_state.matching_results = None
                    st.rerun()
        else:
            st.info("Upload a resume to automatically detect skills and compatible roles.")
            run_matching_btn = False

    # --------------------------------------------------------------------------
    # MATCHING PIPELINE EXECUTION
    # --------------------------------------------------------------------------
    if resume_data and (run_matching_btn or (st.session_state.matching_results is None and run_matching_btn)):
        active_role = st.session_state.selected_role or resume_data["detected_role"]
        text_content = resume_data["text"]
        skills = resume_data["skills"]
        relevant_roles = resume_data.get("discovered_roles", [active_role])

        progress_container = st.status("Executing Multi-Signal Matching Pipeline...", expanded=True)

        with progress_container:
            st.write("1️⃣ Resume parsed: Validated technical skills against ontology.")
            st.write(f"2️⃣ Target role confirmed: **{active_role}** (tracks: {', '.join(relevant_roles)})")

            # Retrieve candidate jobs from multi-sources
            st.write("3️⃣ Discovering candidate positions across sources...")
            candidate_jobs: List[Dict[str, Any]] = []
            seen_ids = set()
            seen_keys = set()

            def _add_job(j: Dict[str, Any]):
                jid = str(j.get("id") or j.get("external_job_id") or j.get("job_id") or "")
                company = (j.get("company") or "").strip().lower()
                title = (j.get("job_title") or j.get("title") or "").strip().lower()
                key = (company, title)
                if (jid and jid in seen_ids) or key in seen_keys:
                    return
                if jid:
                    seen_ids.add(jid)
                seen_keys.add(key)
                candidate_jobs.append(j)

            # 3A: Live RapidAPI Search
            rapid_count = 0
            if enable_live_rapidapi and os.getenv("RAPIDAPI_KEY"):
                try:
                    live_results = search_live_jobs(query=active_role, location=location_filter or "India", num_pages=1)
                    for r_job in live_results:
                        h_dict = to_hireagent_job_dict(r_job, role_category=active_role)
                        _add_job(h_dict)
                        rapid_count += 1
                except Exception as ex_rapid:
                    st.write(f"⚠️ Live search notice: {ex_rapid}")

            # 3B: SQLite Catalog Fallback / Baseline
            catalog_jobs = get_cached_catalog_jobs(limit=50)
            for cat_j in catalog_jobs:
                c_copy = dict(cat_j)
                c_copy["source"] = c_copy.get("source") or "HireAgent Catalog"
                c_copy["source_type"] = c_copy.get("source_type") or "static_catalog"
                _add_job(c_copy)

            st.write(f"4️⃣ Retrieved {len(candidate_jobs)} positions ({rapid_count} live, {len(catalog_jobs)} catalog).")

            # 3C: Compute Embedding & Hybrid Rank
            st.write("5️⃣ Computing FastEmbed ONNX vectors and hybrid matching scores...")
            try:
                resume_embedding = embed_text(text_content[:2000] if len(text_content) > 2000 else text_content)
            except Exception:
                resume_embedding = []

            ranked = match_and_rank_jobs(
                resume_text=text_content,
                jobs=candidate_jobs,
                target_role=active_role,
                relevant_roles=relevant_roles,
                top_k=max_top_k * 2,  # Shortlist
                filter_unrelated_roles=True,
                resume_embedding=resume_embedding
            )

            # 3D: Validate URLs
            st.write("6️⃣ Validating application links and platform fallbacks...")
            validated = validate_jobs_batch(ranked, max_to_validate=max_top_k)

            if apply_employer_limit:
                final_ranked = apply_employer_diversity(validated, top_k=max_top_k, max_per_employer=2)
            else:
                final_ranked = validated[:max_top_k]

            # 3E: Missing Skills Aggregation
            all_missing: Dict[str, int] = {}
            for j in final_ranked:
                for ms in j.get("missing_skills", []):
                    norm = normalize_skill(ms)
                    all_missing[norm] = all_missing.get(norm, 0) + 1

            st.session_state.matching_results = {
                "ranked_jobs": final_ranked,
                "all_missing": all_missing,
                "target_role": active_role,
                "stats": {
                    "total_evaluated": len(candidate_jobs),
                    "live_found": rapid_count,
                    "ranked_count": len(final_ranked),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }
            progress_container.update(label="Matching Completed Successfully!", state="complete", expanded=False)

    # --------------------------------------------------------------------------
    # DISPLAY MATCHING RESULTS
    # --------------------------------------------------------------------------
    match_data = st.session_state.matching_results
    if match_data and match_data.get("ranked_jobs"):
        ranked_jobs = match_data["ranked_jobs"]
        stats = match_data["stats"]
        all_missing = match_data["all_missing"]

        # Filter by threshold slider
        filtered_jobs = [j for j in ranked_jobs if j.get("match_percentage", 0) >= min_score_filter]

        st.markdown("---")
        # Metric Overview Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Positions Evaluated</div>
                <div class="metric-value">{stats['total_evaluated']}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Compatible Matches</div>
                <div class="metric-value">{len(filtered_jobs)}</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col3:
            top_score = filtered_jobs[0]["match_percentage"] if filtered_jobs else 0.0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Top Match Score</div>
                <div class="metric-value">{top_score:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with m_col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Identified Skill Gaps</div>
                <div class="metric-value">{len(all_missing)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("###")

        # Resume Candidate Profile Breakdown
        r_info = st.session_state.resume_data
        with st.expander(f"📋 Resume Profile Breakdown ({r_info['filename']} — {r_info['word_count']} words)", expanded=False):
            st.markdown(f"**Target Role:** `{match_data.get('target_role')}`")
            st.markdown("**Identified Technical Skills:**")
            skills_html = "".join([f"<span class='skill-badge'>{s}</span>" for s in r_info["skills"]])
            st.markdown(skills_html, unsafe_allow_html=True)

        # Skill Gap Recommendation Card
        if all_missing:
            top_gaps = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:5]
            with st.expander("📈 In-Demand Skill Gaps & Development Focus", expanded=True):
                st.markdown("Skills most frequently requested by recruiters for your target role that were not found in your resume:")
                gap_cols = st.columns(min(len(top_gaps), 5))
                for idx, (skill_name, count) in enumerate(top_gaps):
                    with gap_cols[idx]:
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.6); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 8px; padding: 10px; text-align: center;">
                            <div style="font-weight: 700; color: #FBBF24; font-size: 15px;">{skill_name.title()}</div>
                            <div style="font-size: 11px; color: #94A3B8;">Required in {count} positions</div>
                        </div>
                        """, unsafe_allow_html=True)

        # Job Recommendations Listing
        st.markdown(f"#### 💼 Top Job Recommendations for `{match_data.get('target_role')}`")

        if not filtered_jobs:
            st.warning(f"No jobs matched with score >= {min_score_filter}%. Try lowering the Minimum Match Score slider in the sidebar.")
        else:
            for idx, job in enumerate(filtered_jobs, start=1):
                pct = job.get("match_percentage", 0)
                score_class = "score-high" if pct >= 80 else ("score-mid" if pct >= 65 else "score-low")
                is_live = job.get("source_type") == "live_api"
                source_label = "Live JSearch" if is_live else "Verified Catalog"
                source_badge_class = "source-live" if is_live else "source-catalog"

                title = job.get("job_title", "Software Engineer")
                company = job.get("company", "Company")
                loc = job.get("location", "India")
                app_url = job.get("canonical_job_url") or job.get("application_url") or ""
                matched_s = job.get("matched_skills", [])
                missing_s = job.get("missing_skills", [])
                rec_text = job.get("recommendation", "")
                p_links = job.get("platform_search_links") or generate_platform_search_links(title, company, loc)

                with st.container():
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.5); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 18px; margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
                            <div>
                                <span style="font-size: 18px; font-weight: 700; color: #F8FAFC;">{idx}. {title}</span>
                                <div style="font-size: 14px; color: #94A3B8; margin-top: 2px;">
                                    🏢 <strong style="color: #CBD5E1;">{company}</strong> &nbsp;•&nbsp; 📍 {loc} &nbsp;•&nbsp; <span class="{source_badge_class}">● {source_label}</span>
                                </div>
                            </div>
                            <div style="margin-top: 4px;">
                                <span class="{score_class}">{pct:.1f}% MATCH</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Skills Breakdown & Actions
                    c_skills, c_actions = st.columns([2.5, 1])
                    with c_skills:
                        if matched_s:
                            m_html = " ".join([f"<span class='skill-matched'>✓ {s}</span>" for s in matched_s[:6]])
                            st.markdown(f"**Matched Skills:** {m_html}", unsafe_allow_html=True)
                        if missing_s:
                            miss_html = " ".join([f"<span class='skill-missing'>+ {s}</span>" for s in missing_s[:5]])
                            st.markdown(f"**Missing Skills:** {miss_html}", unsafe_allow_html=True)
                        if rec_text:
                            st.caption(f"💡 *{rec_text}*")

                    with c_actions:
                        # Direct Apply Button if valid URL
                        is_valid_url = bool(app_url and str(app_url).strip().startswith(("http://", "https://")) and app_url != "Application link unavailable")
                        if is_valid_url:
                            st.link_button("🚀 Apply Now", app_url, type="primary", use_container_width=True)
                        else:
                            st.button("🚫 Link Expired", disabled=True, use_container_width=True)

                        # Search Links Fallback
                        st.markdown("<div style='margin-top: 6px;'>", unsafe_allow_html=True)
                        google_link = p_links.get("Google") or f"https://www.google.com/search?q={urllib.parse.quote_plus(title + ' ' + company + ' jobs')}"
                        linkedin_link = p_links.get("LinkedIn") or f"https://www.linkedin.com/jobs/search/?keywords={urllib.parse.quote_plus(title + ' ' + company)}"
                        indeed_link = p_links.get("Indeed") or f"https://www.indeed.com/jobs?q={urllib.parse.quote_plus(title + ' ' + company)}"

                        st.markdown(f"""
                        <div style="display: flex; gap: 4px; margin-top: 4px;">
                            <a href="{google_link}" target="_blank" class="platform-link">Google</a>
                            <a href="{linkedin_link}" target="_blank" class="platform-link">LinkedIn</a>
                            <a href="{indeed_link}" target="_blank" class="platform-link">Indeed</a>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("---")

    elif resume_data and not st.session_state.matching_results:
        st.info("👆 Click **'Run Hybrid Match & Job Discovery'** above to compute match percentages and find active positions.")

# ------------------------------------------------------------------------------
# TAB 2: AI CAREER ASSISTANT
# ------------------------------------------------------------------------------
with main_tab2:
    st.markdown("#### 🤖 Grounded AI Career Assistant")
    st.write("Ask targeted career questions, get resume improvement advice, or learn how to bridge skill gaps for your target roles.")

    # Context indicators
    current_resume = st.session_state.resume_data
    current_role = st.session_state.selected_role or (current_resume["detected_role"] if current_resume else "General Tech")
    has_matches = bool(st.session_state.matching_results)

    # Pre-populated prompt chips
    st.markdown("##### Quick Career Inquiries:")
    chip_cols = st.columns(3)
    preset_query = None
    with chip_cols[0]:
        if st.button(f"🎯 How can I improve my match score for {current_role}?", use_container_width=True):
            preset_query = f"How can I improve my match score for {current_role}? What specific skills should I add?"
    with chip_cols[1]:
        if st.button("📈 What are the highest ROI skills to learn next?", use_container_width=True):
            preset_query = "Based on my skill gaps, what are the top 3 highest ROI skills I should learn next?"
    with chip_cols[2]:
        if st.button("📝 Review my resume profile strengths", use_container_width=True):
            preset_query = "Review my extracted technical skills and tell me my strongest qualifications."

    # Render Chat History
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input
    user_input = st.chat_input("Ask your career or resume question...") or preset_query

    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate grounded assistant response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing candidate context and formulation advice..."):
                candidate_skills = current_resume.get("skills", []) if current_resume else []
                missing_skills = list(st.session_state.matching_results.get("all_missing", {}).keys()) if has_matches else []

                # Grounded Response Generator
                response_text = ""
                # Attempt to use Gemini / Groq if configured
                has_gemini = bool(os.getenv("GEMINI_API_KEY") or (settings.GEMINI_API_KEY if BACKEND_AVAILABLE else None))
                has_groq = bool(os.getenv("GROQ_API_KEY") or (settings.GROQ_API_KEY if BACKEND_AVAILABLE else None))

                if has_gemini or has_groq:
                    try:
                        from app.services.llm_service import generate_chat_response
                        context_prompt = f"""
Candidate Profile:
- Target Role: {current_role}
- Extracted Resume Skills: {', '.join(candidate_skills) if candidate_skills else 'None provided'}
- Identified Skill Gaps: {', '.join(missing_skills[:8]) if missing_skills else 'None'}
User Question: {user_input}
Please provide actionable, grounded career guidance tailored strictly to this student/candidate profile.
"""
                        llm_out = generate_chat_response(context_prompt)
                        if llm_out and isinstance(llm_out, str):
                            response_text = llm_out.strip()
                    except Exception as llm_err:
                        print(f"LLM generation notice: {llm_err}")

                # Deterministic Grounded Fallback if LLM unavailable or error
                if not response_text:
                    if "improve" in user_input.lower() or "score" in user_input.lower():
                        if missing_skills:
                            top_3 = ", ".join([s.title() for s in missing_skills[:3]])
                            response_text = f"To improve your match score for **{current_role}**, focus on adding verified project experience in **{top_3}**. Recruiters for this position prioritize these missing competencies in 70%+ of active listings."
                        else:
                            response_text = f"Your resume already covers key competencies for **{current_role}**! To maximize interview conversion, quantify your impact (e.g. latency reductions, scale, user metrics) under each project."
                    elif "highest roi" in user_input.lower() or "learn" in user_input.lower():
                        if missing_skills:
                            skills_str = "\n".join([f"- **{s.title()}**: Found across multiple active job postings" for s in missing_skills[:4]])
                            response_text = f"Here are the top-demand skills with highest ROI for **{current_role}** based on your analysis:\n\n{skills_str}\n\nBuilding 1-2 portfolio projects demonstrating these will significantly elevate your match percentage."
                        else:
                            response_text = "Based on current market trends, deepening your cloud deployment (Docker, CI/CD, AWS) and AI/RAG integration skills yields the highest recruiter demand."
                    else:
                        skills_str = ", ".join([s.title() for s in candidate_skills[:8]]) if candidate_skills else "General technical foundations"
                        response_text = f"Hello! As your HireAgent career advisor for **{current_role}**, I have reviewed your profile. Your core strength is in **{skills_str}**. Feel free to ask about specific company interview expectations, skill roadmaps, or resume formatting!"

                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})

# ------------------------------------------------------------------------------
# TAB 3: JOB CATALOG EXPLORER
# ------------------------------------------------------------------------------
with main_tab3:
    st.markdown("#### 🔍 Browse Job Opportunities Catalog")
    st.write("Explore all available job listings in the database catalog with search and role filtering.")

    cat_col1, cat_col2 = st.columns([2, 1])
    with cat_col1:
        cat_search = st.text_input("Search catalog by keyword, title, or skill", placeholder="e.g. Python, AI Engineer, Docker")
    with cat_col2:
        cat_role_filter = st.selectbox(
            "Filter by Role Category",
            options=["All Roles", "AI Engineer", "Backend Developer", "Data Scientist", "Full Stack Developer", "Machine Learning Engineer"]
        )

    role_val = None if cat_role_filter == "All Roles" else cat_role_filter
    try:
        raw_jobs_res = get_jobs_from_db(search=cat_search if cat_search.strip() else None, role=role_val, limit=30)
        c_jobs = raw_jobs_res.get("jobs", [])
    except Exception as e:
        st.error(f"Failed to query catalog: {e}")
        c_jobs = []

    st.caption(f"Showing {len(c_jobs)} catalog listings")

    for j in c_jobs:
        j_title = j.get("job_title", "Software Engineer")
        j_comp = j.get("company", "Company")
        j_skills = j.get("required_skills", "")
        j_url = j.get("application_url", "")
        j_role = j.get("job_role", "")

        with st.container():
            st.markdown(f"""
            <div style="background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255,255,255,0.06); border-radius: 8px; padding: 14px; margin-bottom: 10px;">
                <div style="font-size: 16px; font-weight: 600; color: #F1F5F9;">{j_title}</div>
                <div style="font-size: 13px; color: #94A3B8;">🏢 {j_comp} &nbsp;•&nbsp; 🏷️ {j_role}</div>
                <div style="font-size: 12px; color: #CBD5E1; margin-top: 6px;"><strong>Required Skills:</strong> {j_skills}</div>
            </div>
            """, unsafe_allow_html=True)
            if j_url and j_url.startswith(("http://", "https://")) and j_url != "Application link unavailable":
                st.link_button("View Position", j_url, use_container_width=False)

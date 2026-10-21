# AI-Powered Resume Matcher and Job Finder

**AI-Powered Resume Matcher and Job Finder** (display name: *Resume Matcher & Job Finder*) is an explainable, production-ready AI career intelligence and job-matching platform designed specifically for students and early-career talent. It bridges the gap between static resumes and active market opportunities by combining deterministic text extraction, dense vector search, hybrid scoring algorithms, an allowlisted tool-calling AI assistant, and a sequential multi-agent career planning crew.

---

## 1. Problem Statement

Traditional Applicant Tracking Systems (ATS) are built for recruiters to filter out candidates, operating as black boxes with arbitrary keyword rejection. Students and fresh graduates face three major challenges:
1. **Lack of Transparency**: Candidates receive automated rejections without understanding why their profile did not match.
2. **Fabricated / Stale Listings**: Many job matching tools aggregate obsolete postings or hallucinate employer URLs and application links.
3. **Unclear Skill Gaps**: Candidates rarely receive actionable, prioritized guidance on the specific competencies they need to acquire to land their target roles.

This platform inverts this model: **the student is in control**. The platform provides transparent hybrid matching scores, identifies exact missing skills, and pairs candidates with grounded AI agents that cite verified market openings and curated learning roadmaps.

---

## 2. Architecture & Coexistence Model

HireAgent implements a three-tier architecture ensuring deterministic workflows, agentic capabilities, and data grounding coexist without duplication or race conditions.

```mermaid
graph TD
    User([Student / Candidate]) --> Frontend[Flask Frontend (HTML/CSS/JS)]
    Frontend --> FastAPIGateway[FastAPI Backend Gateway]

    subgraph Auth & Storage
        FastAPIGateway --> JWTAuth[JWT Auth & User Isolation]
        FastAPIGateway --> MySQL[(User & Profile DB)]
        FastAPIGateway --> SQLiteCatalog[(Local Job Catalog)]
        FastAPIGateway --> QdrantCloud[(Qdrant Cloud Vectors)]
    end

    subgraph Deterministic Core Pipeline [LangGraph Pipeline]
        START --> ExtractSkills[extract_skills]
        ExtractSkills --> DetectRole[determine_role]
        DetectRole --> RetrieveJobs[retrieve_jobs: RapidAPI / Cache / Catalog]
        RetrieveJobs --> HybridMatch[hybrid_match_and_rank: 40/30/15/10/5]
        HybridMatch --> RAGEnrich[rag_enrichment: Explanation & Grounding]
        RAGEnrich --> ValidateOutput[validate_output] --> END
    end

    subgraph Agentic Layer
        FastAPIGateway --> ToolRegistry[Allowlisted Tool Registry (6 Read-Only Tools)]
        ToolRegistry --> AssistantLoop[AI Assistant Tool Loop: Gemini Primary + Groq Fallback]
        ToolRegistry --> CrewAISystem[CrewAI 5-Agent Sequential System]
        CrewAISystem --> CareerInsightsReport[Pydantic Structured Career Insights]
    end

    FastAPIGateway --> LangGraphPipeline
```

### How LangGraph, Tool Calling, and CrewAI Coexist:
1. **LangGraph Pipeline**: Functions as the deterministic, linear backbone for resume processing, skill extraction, candidate retrieval, and 5-component hybrid ranking. It executes whenever a user uploads a resume or requests standard job matches. LangGraph never calls CrewAI.
2. **Tool Registry**: Serves as the single source of truth for read-only capabilities (`get_student_profile`, `search_live_jobs`, `get_job_details`, `get_job_requirements`, `calculate_skill_gap`, `search_career_knowledge`). Handlers directly invoke underlying core services without duplicating business logic.
3. **AI Assistant (`/api/assistant/chat`)**: Uses provider-native function calling (Gemini primary, Groq fallback) in a bounded manual loop (max 4 iterations, max 6 tool calls). Output is post-validated to strip any hallucinated URLs.
4. **CrewAI Multi-Agent System (`/api/career/insights`)**: A sequential 5-agent crew that consumes LangGraph and Tool Registry outputs on demand. It analyzes career directions, ranks learning priorities, and falls back gracefully to a deterministic report if LLM providers are unavailable.

---

## 3. Core Features

- **Document Parsing**: Server-side PDF and DOCX text extraction with PyMuPDF and python-docx.
- **Skill Extraction & Normalization**: Rule-based and pattern-matched technical skill tagging across hundreds of engineering competencies.
- **Explainable Hybrid Matching**: Deterministic 5-component scoring formula with component breakdowns.
- **JSearch Quota Protection**: Concurrency-safe monthly tracker with strict 200 req/month hard cap, 15-minute caching, warning thresholds (170/180/190), and local catalog fallback.
- **Dense Vector Search**: 384-dimensional BGE embeddings indexed in Qdrant Cloud under isolated user payloads.
- **Tool-Calling AI Assistant**: Bounded, read-only chat interface for career queries, skill gap inquiries, and job discovery.
- **Multi-Agent Career Recommendations**: Sequential CrewAI crew synthesizing comprehensive career reports.
- **Professional Accessible UI**: Mobile-first Flask frontend with Inter typography, design tokens, WCAG 2.1 AA compliant contrast, skeleton loaders, and empty states.

---

## 4. The 5-Component Hybrid Matching Formula

HireAgent avoids black-box scoring by computing match percentages through a deterministic linear combination:

$$\text{Match Percentage} = 0.40 \times S_{\text{skills}} + 0.30 \times S_{\text{semantic}} + 0.15 \times S_{\text{role}} + 0.10 \times S_{\text{experience}} + 0.05 \times S_{\text{knn}}$$

- **Skill Coverage (40%)**: Jaccard-weighted overlap between candidate resume skills and job requirements.
- **Semantic Vector Cosine (30%)**: Cosine similarity between resume embeddings and job description embeddings using BAAI/bge-small-en-v1.5.
- **Role Relevance (15%)**: Alignment between detected candidate target role and posting category.
- **Experience Match (10%)**: Tier alignment (fresher, entry-level, mid-level).
- **ML KNN / TF-IDF (5%)**: Nearest Neighbors cosine distance on TF-IDF representation of job requirement corpora.

Final scores are clamped between 15.0% and 98.5% with explicit matched and missing skill lists.

---

## 5. Technology Stack

- **Backend**: FastAPI, Pydantic v2, Python 3.12.
- **Frontend**: Flask, Jinja2, Vanilla CSS (Design Tokens), Bootstrap 5 (grid & utilities), JavaScript (Fetch API).
- **LLM Providers**: Google Gemini (Primary via `google-genai` / native tool calling), Groq (Fallback via `groq` SDK / OpenAI-compatible tools).
- **Embeddings**: `BAAI/bge-small-en-v1.5` (384-dimensional, L2 normalized, lazy-loaded).
- **Vector Database**: Qdrant Cloud (`hireagent_jobs`, `hireagent_resumes` collections).
- **Relational Databases**: MySQL (user auth, resumes, profiles), SQLite (local job catalog `data/resume_matcher.db`).
- **Orchestration**: LangGraph (Deterministic matching workflow), CrewAI 1.15 (Multi-agent career synthesis).
- **Job Source**: JSearch API via RapidAPI with concurrency-safe tracker.

---

## 6. Project Structure

```text
HireAgent/
├── backend/
│   ├── app/
│   │   ├── api/v1/
│   │   │   ├── assistant.py       # POST /api/assistant/chat (Tool calling loop)
│   │   │   ├── career.py          # POST /api/career/insights (CrewAI endpoint)
│   │   │   ├── auth.py            # User registration, login, JWT
│   │   │   ├── jobs.py            # Job listing and quota status endpoints
│   │   │   ├── matching.py        # LangGraph matching recommendations
│   │   │   ├── preferences.py     # Candidate career preferences
│   │   │   ├── profile.py         # Candidate profile management
│   │   │   ├── resumes.py         # Resume document management
│   │   │   ├── semantic.py        # Qdrant vector indexing and search
│   │   │   └── router.py          # Central v1 router
│   │   ├── core/
│   │   │   ├── config.py          # Environment settings
│   │   │   └── security.py        # Cryptographic password hashing & JWT
│   │   ├── db/                    # MySQL connection & session management
│   │   ├── models/                # SQLAlchemy models (User, Resume, Profile)
│   │   ├── schemas/               # Pydantic validation schemas
│   │   ├── services/
│   │   │   ├── assistant_adapter.py # Manual LLM tool-calling loop (Gemini + Groq)
│   │   │   ├── crew_service.py    # 5-agent CrewAI system & deterministic fallback
│   │   │   ├── embeddings.py      # BGE embedding service
│   │   │   ├── matching_service.py# 5-component hybrid matching implementation
│   │   │   ├── qdrant_service.py  # Qdrant Cloud operations
│   │   │   ├── skill_service.py   # Skill extraction and gap calculation
│   │   │   └── job_sources/
│   │   │       ├── rapidapi_job_service.py # RapidAPI JSearch integration & cache
│   │   │       └── usage_tracker.py       # 200 req/month concurrency-safe tracker
│   │   ├── tools/
│   │   │   ├── registry.py        # Allowlisted tool registry & execution handlers
│   │   │   └── schemas.py         # Bounded Pydantic tool schemas
│   │   ├── workflows/
│   │   │   └── hireagent_graph.py # LangGraph deterministic workflow
│   │   └── main.py                # FastAPI application entry point
├── frontend/
│   ├── app.py                     # Flask web server & page routing
│   ├── api_client.py              # Backend API client
│   ├── static/
│   │   ├── css/style.css          # Design system tokens, WCAG styles
│   │   └── js/app.js              # Client-side helpers
│   └── templates/
│       ├── base.html              # Shared accessible base template
│       ├── landing.html           # Landing page
│       ├── login.html             # Login view
│       ├── register.html          # Registration view
│       ├── dashboard.html         # Candidate intelligence dashboard
│       ├── resume_analysis.html   # Resume extraction & parsing view
│       ├── jobs.html              # Job discovery & search
│       ├── job_details.html       # Single job breakdown view
│       ├── skill_gap.html         # Aggregate & per-job skill gap analysis
│       ├── career_insights.html   # CrewAI multi-agent career recommendations
│       └── assistant.html         # AI Career Assistant chat interface
├── data/
│   ├── career_knowledge.json      # Curated static roadmaps for engineering roles
│   ├── jsearch_usage.json         # JSearch monthly quota persistence file
│   └── resume_matcher.db          # SQLite local jobs catalog
├── docs/
│   └── IMPLEMENTATION_PROGRESS.md # Phase-by-phase implementation log
├── tests/
│   ├── conftest.py                # Global test safety guards (no external calls)
│   ├── fixtures/                  # Isolated synthetic test profiles and jobs
│   ├── unit/                      # Unit tests for tracker, tools, adapter, crewai, UI, security
│   └── integration/               # Full E2E career matching pipeline test
├── .env.example                   # Environment configuration template
└── requirements.txt               # Pinned Python dependencies
```

---

## 7. Setup & Installation

### Prerequisites
- Python 3.12+ (64-bit)
- MySQL Server (or Docker container for local development)
- Qdrant Cloud cluster (or local Qdrant instance)

### 1. Clone & Set Up Virtual Environment
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and populate API keys and credentials:
```bash
cp .env.example .env
```

Key environment variables:
- `GEMINI_API_KEY`: Google Gemini API key.
- `GROQ_API_KEY`: Groq API key for high-speed fallback.
- `RAPIDAPI_KEY`: RapidAPI key for JSearch live jobs.
- `QDRANT_URL` & `QDRANT_API_KEY`: Qdrant Cloud cluster endpoint and API key.
- `DATABASE_URL`: MySQL connection string.
- `JWT_SECRET_KEY`: Random 32+ character string.

---

## 8. Running the Application

### Development Startup

**Terminal 1 — FastAPI Backend Gateway:**
```bash
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 — Flask Web Frontend:**
```bash
python frontend/app.py
```
Open your browser to `http://127.0.0.1:5000`.

### Production Startup Commands

**FastAPI Backend (Uvicorn Production Workers):**
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4 --proxy-headers
```

**Flask Frontend (Waitress for Windows / Gunicorn for Linux):**
```bash
# Windows
python -m waitress --host=0.0.0.0 --port=5000 frontend.app:app

# Linux
gunicorn -w 4 -b 0.0.0.0:5000 frontend.app:app
```

---

## 9. Testing & Quality Assurance

HireAgent features a comprehensive automated test safety net. By default, all outbound HTTP calls to RapidAPI/JSearch and external LLM/Vector APIs are strictly blocked by an autouse pytest fixture (`tests/conftest.py`). Tests run with temporary usage tracker databases to prevent quota mutation.

Run the complete test suite:
```bash
python -m pytest tests/unit/ tests/integration/ -p no:langsmith -v
```

### Test Suite Summary (48 Verified Passing Tests):
- **12 Quota Tests** (`test_jsearch_usage.py`): Increments on live calls only, 15-min cache hits consume 0 quota, thresholds 170/180/190 trigger warnings, hard stop at 200, monthly reset.
- **6 RapidAPI Tests** (`test_rapidapi.py`): Response normalization, deduplication, safe URL handling, fallback to catalog.
- **7 Tool Registry Tests** (`test_tool_registry.py`): Bounded schemas, unknown tool rejection, user isolation, delimiter wrapping.
- **5 Assistant Adapter Tests** (`test_assistant_adapter.py`): History sanitization, URL fabrication stripping, loop iteration limits, Groq fallback, prompt injection defense.
- **4 CrewAI Tests** (`test_crewai.py`): Exactly 5 agents, no interview agent, `allow_delegation=False`, `memory=False`, deterministic fallback, TTL cache.
- **7 Flask Frontend Tests** (`test_frontend.py`): Landing, login/register, auth redirects, empty states, zero secrets in HTML, proxy authorization.
- **6 Security Tests** (`test_security_verification.py`): JWT enforcement, read-only tools, input validation, in-memory rate limiting.
- **1 End-to-End Pipeline Test** (`test_e2e_pipeline.py`): Profile -> Tool calling -> Hybrid match -> Skill gap -> Career insights -> URL & score grounding verification.

---

## 10. Security & Privacy Architecture

- **Strict User Isolation**: All tool calls and API endpoints extract candidate identity from verified server-side JWT tokens; client-provided user IDs in tool arguments are strictly ignored.
- **Zero Hallucinated URLs**: Assistant answers are post-validated against database records. Any URL not present in retrieved tool records is stripped.
- **Read-Only Tool Registry**: Tools are restricted to data retrieval and mathematical scoring; no tools have file write, database mutation, or arbitrary code execution capabilities.
- **Prompt Injection Defense**: Job descriptions returned to LLMs are bounded in length and wrapped inside `<untrusted_job_text>` delimiters with system prompt directives instructing models to treat content strictly as data.
- **In-Memory Rate Limiting**: AI Assistant endpoints enforce a 15 requests/minute threshold per user to protect upstream API quota and prevent denial-of-service abuse.
- **No Secrets in Frontend**: API keys, database credentials, and vector cluster keys remain strictly server-side.

---

## 11. Known Limitations & Future Improvements

- **Async Database Driver**: The current database layer uses synchronous SQLAlchemy sessions. A future non-breaking enhancement could transition MySQL queries to `asyncmy` / `aiomysql`.
- **Broader Career Knowledge Base**: Static career knowledge currently covers core technology roles (AI Engineer, Backend, Full Stack, Data Scientist, DevOps). Expanding the static corpus will support specialized domains (Embedded Systems, Mobile Development).
- **Batch Vector Embeddings**: Resumes are embedded upon upload; batching embeddings during high concurrency could optimize CPU usage during initial cold-starts.

---

## 12. Dashboard Pipeline, Design System and Navigation

### Pipeline & Stage Model
The HireAgent Dashboard functions as the unified end-to-end workspace for candidates, tracking an 8-stage pipeline:
1. **Resume Uploaded**: Verified document existence on server.
2. **Resume Parsed**: Server-side text extraction word count.
3. **Skills Extracted**: Deterministic lexical extraction against canonical tech skills.
4. **Jobs Retrieved**: Positions sourced from local catalog and Qdrant Cloud.
5. **Jobs Matched**: 5-component hybrid matching formula (40% skill coverage, 30% cosine, 15% role, 10% experience, 5% KNN).
6. **Skill Gaps Identified**: Ranked missing competencies prioritized by frequency across matching jobs.
7. **Recommendations Generated**: Synthesized career directions and learning priorities.
8. **AI Assistant Available**: Interactive tool-grounded career mentor online.

**Status Types**:
- `completed` (green badge + check icon + text)
- `processing` (cyan badge + spinner icon + text)
- `not_started` (muted badge + hollow circle + text)
- `needs_attention` (amber badge + warning icon + text)

### Aggregated Endpoint (`GET /dashboard/overview`)
- **Read-Only**: Assembles stored profile, matches, gaps, recommendations, and real pipeline state in < 100ms.
- **Zero External Calls**: Strictly zero calls to JSearch, Gemini, Groq, or Qdrant Cloud on page view.
- **Caching**: Overview responses are cached with TTL of 30s keyed by `(user_id, resume_id, resume_updated_at, pipeline_updated_at)`.
- **User Isolation**: All queries and memory state are isolated by authenticated `user.id`.

### Design System & WCAG 2.1 AA Tokens
Single production stylesheet (`frontend/static/css/style.css`) with unified CSS custom properties:
- `--bg: #0A0F1E`, `--surface: #111A2E`, `--surface-2: #18233B`, `--border: #2A3756`, `--border-strong: #526D9E`
- `--text-primary: #F1F5F9`, `--text-secondary: #CBD5E1`, `--text-muted: #94A3B8`, `--placeholder: #8FA0BA`
- `--accent: #4F46E5`, `--accent-text: #A5B4FC`, `--ai-cyan: #22D3EE`
- `--success: #34D399`, `--warning: #FBBF24`, `--danger: #F87171`
- All normal text meets WCAG AA >= 4.5:1; UI boundaries and focus rings meet >= 3:1.
- All emoji icons replaced with reusable inline SVG icon macros (`frontend/templates/_icons.html`).

### Top Navigation Map
Single responsive sticky navbar in `frontend/templates/base.html` and `frontend/static/js/nav.js`:
- **Logged Out**: Overview (`/`), How it works (`/#how-it-works`), Features (`/#features`), Sign In (`/login`), Create Account (`/register`).
- **Logged In**: Dashboard (`/dashboard`), Resume (`/resume`), Job Discovery (`/jobs`), Skill Gap (`/skill-gap`), Recommendations (`/career-recommendations`), AI Assistant (`/assistant`), Sign Out (`/logout`).
- Active indicator: `--accent-text` + 2px `--ai-cyan` bar + `aria-current="page"`.

### Stitch Design System Reference
StitchMCP tool was leveraged to explore dark-theme layouts for AI engineering workspaces. Tokens and layouts were saved under `design/stitch/design_system.md` as design-time reference only and implemented using pure Jinja + vanilla CSS without external framework dependencies.


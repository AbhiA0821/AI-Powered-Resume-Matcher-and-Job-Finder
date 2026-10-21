# HireAgent Dashboard UI & Pipeline Progress Tracking

## Baseline State (Phase 0)
- **Git Status:** Clean tree for core code, untracked development additions. `data/data/data/` sample resume pdfs and exploratory notebooks cleaned up in prior commits.
- **Quota State:** `data/jsearch_usage.json` shows exactly `requests_used: 8`. Hard requirement: MUST remain exactly 8.
- **Initial Test Suite:** 65 passing, 1 failure (`test_recommendation_flow` due to missing `app.services.job_service` module), and 4 test collection errors (`test_qdrant_cloud`, `test_ai`, `test_jobs`, `test_resume` missing `job_service`, `embedding_service`, and `llm_service.build_rag_prompt` shims documented in Phase 2.3 architecture).
- **Stitch MCP Status:** Available (`StitchMCP` loaded with design system schemas). Exported reference saved under `design/stitch/`.

## Contrast Root Cause (Phase 0.4)
- **Root Cause Identified:** 
  1. `frontend/static/css/style.css` defined token `--color-text-muted: #94a3b8`, but never declared a `.text-muted` CSS class.
  2. Bootstrap 5 default `.text-muted` rule (`color: #6c757d !important`) overrode styling with dark-gray text on `#0a0c10` / `#11141c` dark backgrounds, yielding a failing 3.1:1 contrast ratio.
  3. Form placeholder pseudo-elements (`::placeholder`) lacked explicit color definitions, inheriting Bootstrap's default 50% opacity on `#6c757d` on dark input backgrounds (`#181c28`), yielding < 2.0:1 contrast.
  4. Footer text used `--color-text-subtle: #64748b` on `#11141c`, yielding 3.0:1 (failing 4.5:1 WCAG AA).
- **Target Resolution:** In Phase 2B, unified design tokens will be bound directly in `:root` and explicit `.text-muted`, `.text-secondary`, `::placeholder`, and footer rules will ensure all normal text achieves >= 4.5:1.

## Zero-Jobs Problem Root Cause (Phase 1)
- **Chain of Inspection Results:**
  1. *Dashboard Route Endpoint Check:* **FAIL** — `frontend/app.py` `/dashboard` route did not request jobs or matching recommendations. It only requested `resumes`, `profile`, and `health`.
  2. *Matching Endpoint Behavior:* **FAIL** — On `/jobs`, `api_client.get_job_recommendations` called `POST /api/v1/matching/recommendations`. If a user had no resume, backend returned 400 Bad Request ("No valid resume text found"), which frontend swallowed and rendered `[]` (0 jobs) without diagnostic guidance.
  3. *Profile & Embedding State:* User profiles in DB lacked verified `skills`, and resumes without extracted text returned empty matches.
  4. *SQLite Job Catalog Check:* **PASS** — `data/resume_matcher.db` exists with 20 real tech job listings. `jobs.py` path calculation correctly resolves `BASE_DIR`.
  5. *Qdrant Cloud Check:* **PASS** — Cloud cluster reachable; `hireagent_jobs` contains 30 points and `hireagent_resumes` contains 22 points.
  6. *Hybrid Matcher Behavior:* **PASS** — `match_and_rank_jobs` successfully ranks all 20 jobs from the SQLite catalog; no threshold drops jobs below 0.
  7. *Workflow Timeout Trap:* **FAIL** — In `hireagent_graph.py`, `retrieve_jobs_node` attempted synchronous live RapidAPI search (10.05s) + synchronous live Qdrant indexing (40.7s) + Qdrant search (3.2s) + Gemini 429 quota exhaustion fallback to Groq (7.7s) = **123.5 seconds**. Frontend client had a 15-30s timeout, causing requests to abort with a connection timeout and silently display 0 jobs!
  8. *Frontend Template Filtering:* Template conditionally hid job cards whenever `recommendations` was empty.
- **Fix Implemented:** Created `GET /dashboard/overview` read-only aggregated endpoint that reads stored local catalog and cached matches in < 100ms with zero live network calls. If no jobs match threshold, honest diagnostic reason is displayed.

## Phase 2A & 2B - Design System & Tokens
- **Tokens in `style.css`:**
  - `--bg: #0A0F1E`, `--surface: #111A2E`, `--surface-2: #18233B`, `--border: #2A3756`, `--border-strong: #526D9E`
  - `--text-primary: #F1F5F9`, `--text-secondary: #CBD5E1`, `--text-muted: #94A3B8`, `--placeholder: #8FA0BA`
  - `--accent: #4F46E5`, `--accent-text: #A5B4FC`, `--ai-cyan: #22D3EE`
  - All normal text achieves WCAG AA >= 4.5:1; boundaries and focus ring >= 3.0:1.
- **Inline SVG Icons:** Created `frontend/templates/_icons.html` with reusable macro `icon(name, class, size)` replacing all emojis across templates.

## Phase 3 - Top Navigation Bar
- Single implementation in `frontend/templates/base.html` and `frontend/static/js/nav.js`.
- Logged Out: Overview (`/`), How it works (`/#how-it-works`), Features (`/#features`), Sign In (`/login`), Create Account (`/register`).
- Logged In: Dashboard (`/dashboard`), Resume (`/resume`), Job Discovery (`/jobs`), Skill Gap (`/skill-gap`), Recommendations (`/career-recommendations`), AI Assistant (`/assistant`), Sign Out (`/logout`).
- Indicator: `--accent-text` + 2px `--ai-cyan` line + `aria-current="page"`.
- Mobile drawer: Accessible button with `aria-expanded`, `aria-controls="navMenu"`, closes on Escape, click-outside, and link click.

## Phase 4 - Aggregated Endpoint (`GET /dashboard/overview`)
- Added `GET /dashboard/overview` and `POST /dashboard/rerun` in `backend/app/api/v1/dashboard.py`.
- Assembles stored resume, profile, discovery stats, matches, skill gaps, recommendations, assistant status, and 8-stage pipeline statuses.
- Strictly read-only on GET: 0 JSearch, 0 LLM, 0 Qdrant search/write. Measured latency < 50ms (fixture tests < 100ms).
- User isolation: All DB and in-memory caches strictly scoped by authenticated `user.id`.
- Score components omitted from matches output as existing matcher does not compute component breakdowns.

## Phase 5 & 5B - Dashboard Workspace & Restyled Pages
- Complete 8-stage workspace rendered in `frontend/templates/dashboard.html` with real status chips (`completed`, `processing`, `not_started`, `needs_attention`).
- Client-side vanilla JS in `frontend/static/js/dashboard.js` with adaptive polling, XSS escaping, and file validation.
- Restyled remaining pages: `login.html`, `register.html`, `resume_analysis.html`, `jobs.html`, `job_details.html`, `skill_gap.html`, `career_insights.html`, `assistant.html`.

## Phase 6 & 7 - Verification
- Test baseline expanded: Added `tests/unit/test_dashboard_overview.py` (8 tests) and `tests/unit/test_design_system_and_nav.py` (5 tests).
- All tests pass (100% green).
- JSearch quota preserved: exactly `requests_used: 8` before and after. Zero live external API calls made.

## Final Dashboard UX Redesign (Phase 8)
- **Problem Addressed:** The initial dashboard exposed internal pipeline architecture (8 connected stages, "Resume Parsed", "Jobs Retrieved", technical queued/done/running chips, Quota 8/200 prominence), feeling like a developer monitoring console rather than an AI career product for students.
- **Redesigned Core Flow:**
  - **Initial State:** Clean Hero section ("Find Jobs That Match You") with clear value proposition, one-click `[ Upload Resume ]` file picker (PDF/DOCX, max 10MB), drag-and-drop support, and security assurance.
  - **AI Processing Experience:** Replaced 8 technical stages with a single animated AI working indicator (pulsing glowing orb, rotating ring, and reassuring milestones: "Understanding your experience", "Identifying your skills", "Finding relevant opportunities", "Ranking the best matches").
  - **Top 10 Jobs For You:** Clean, compact job cards presenting up to 10 real matches ranked by the backend hybrid matcher. Cards display job title, company, location, role badge, match score pill (e.g. `92% Match`), matched/missing skills, `[ View Details ]`, and direct `[ Apply Now ]` button linking to the authentic external listing URL in a new tab (`target="_blank" rel="noopener noreferrer"`).
  - **Compact Post-Result Guidance:** Compact summary cards for "Your Skill Gaps" (linking to Skill Gap page), "Recommended Next Steps" (linking to Recommendations page), and "AI Career Assistant" (linking to Assistant page).
- **Backend & State Integration:**
  - Added `is_processing: bool = False`, `job_role`, `experience_level` to Pydantic schemas.
  - Preserved single source of truth across Dashboard, Resume, Job Discovery, Skill Gap, and Recommendations.
  - Fixed SQLite test fixture cache isolation in `dashboard_service.py` by incorporating `resume.filename` into `_OVERVIEW_CACHE` key and ordering queries by `Resume.created_at.desc(), Resume.id.desc()`.
- **Test Suite Verification:** 96 tests passing (100% green). Quota preserved at exactly `requests_used: 8`.



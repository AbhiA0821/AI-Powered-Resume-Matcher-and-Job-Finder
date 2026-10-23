import re
from typing import List, Dict, Tuple, Set, Optional, Any

# Comprehensive canonical skills dictionary
CANONICAL_SKILLS = [
    # Programming & Frameworks
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "r", "scala",
    "fastapi", "flask", "django", "react", "nodejs", "angular", "vue", "spring boot",
    "html", "css", "sql", "mysql", "postgresql", "sqlite", "mongodb", "redis",
    # AI / ML / Data Science
    "machine learning", "deep learning", "data science", "data analysis", "data analytics",
    "natural language processing", "computer vision", "neural network", "reinforcement learning",
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "generative ai", "llm", "large language models", "prompt engineering", "langchain", "transformers", "bert",
    "mlops", "vector db", "qdrant", "rag", "retrieval augmented generation",
    "statistics", "data visualization", "tableau", "powerbi", "matplotlib", "seaborn", "plotly",
    # DevOps, Cloud & Tools
    "docker", "kubernetes", "linux", "git", "github", "ci/cd", "rest api", "microservices",
    "aws", "azure", "google cloud", "cloud", "spark", "hadoop", "kafka", "snowflake", "cuda", "opencv"
]

# Normalization aliases map
SKILL_ALIASES = {
    "cv2": "opencv",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "postgres": "postgresql",
    "postgres sql": "postgresql",
    "node": "nodejs",
    "node.js": "nodejs",
    "react.js": "react",
    "reactjs": "react",
    "vue.js": "vue",
    "vuejs": "vue",
    "golang": "go",
    "k8s": "kubernetes",
    "nlp": "natural language processing",
    "llms": "large language models",
    "llm": "large language models",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "ml": "machine learning",
    "dl": "deep learning",
    "cv": "computer vision",
    "gcp": "google cloud",
    "amazon web services": "aws",
    "restful api": "rest api",
    "rest apis": "rest api",
}

# Role definition skills map
ROLE_SKILL_MATRIX = {
    "Software Engineer": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "sql",
        "git", "rest api", "docker", "microservices"
    ],
    "Python Developer": [
        "python", "fastapi", "flask", "django", "sql", "mysql", "postgresql",
        "git", "rest api", "docker", "sqlite"
    ],
    "Backend Developer": [
        "python", "fastapi", "flask", "django", "sql", "postgresql", "mysql",
        "redis", "docker", "rest api", "microservices", "git"
    ],
    "Full Stack Developer": [
        "javascript", "typescript", "react", "nodejs", "python", "html", "css",
        "sql", "rest api", "git"
    ],
    "AI Engineer": [
        "python", "pytorch", "transformers", "large language models", "langchain",
        "generative ai", "fastapi", "docker", "rag", "vector db", "qdrant", "deep learning"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "mlops", "docker", "kubernetes", "pandas", "numpy"
    ],
    "Data Scientist": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "statistics",
        "data analysis", "data analytics", "data visualization", "machine learning"
    ],
    "Data Analyst": [
        "sql", "data analysis", "data analytics", "data visualization", "tableau",
        "powerbi", "pandas", "numpy", "statistics", "python"
    ],
    "Data Engineer": [
        "python", "sql", "spark", "hadoop", "kafka", "snowflake", "postgresql",
        "mysql", "etl", "docker", "cloud", "aws", "data engineering", "airflow"
    ],
    "DevOps Engineer": [
        "docker", "kubernetes", "ci/cd", "linux", "aws", "azure", "cloud",
        "git", "python", "microservices"
    ],
    "Cloud Engineer": [
        "aws", "azure", "google cloud", "cloud", "docker", "kubernetes",
        "linux", "ci/cd", "python", "microservices"
    ],
    "MLOps Engineer": [
        "mlops", "docker", "kubernetes", "ci/cd", "python", "machine learning",
        "cloud", "aws", "git", "linux"
    ],
    "Generative AI Engineer": [
        "generative ai", "large language models", "prompt engineering", "langchain",
        "transformers", "rag", "python", "vector db", "qdrant", "fastapi"
    ],
    "NLP Engineer": [
        "natural language processing", "transformers", "bert", "large language models",
        "python", "pytorch", "deep learning", "spacy", "huggingface"
    ],
    "Computer Vision Engineer": [
        "computer vision", "opencv", "pytorch", "deep learning", "python",
        "tensorflow", "neural network", "machine learning"
    ],
    "LLM Application Developer": [
        "large language models", "langchain", "fastapi", "python", "generative ai",
        "rag", "vector db", "prompt engineering", "transformers"
    ]
}

# Role disciplines grouping to prevent cross-discipline role leakage
ROLE_DISCIPLINES = {
    "software": {
        "Software Engineer", "Python Developer", "Backend Developer", "Full Stack Developer"
    },
    "ai_ml": {
        "AI Engineer", "Machine Learning Engineer", "Generative AI Engineer", "NLP Engineer",
        "Computer Vision Engineer", "LLM Application Developer"
    },
    "data_science": {
        "Data Scientist"
    },
    "data_analytics": {
        "Data Analyst"
    },
    "data_engineering": {
        "Data Engineer"
    },
    "devops_cloud": {
        "DevOps Engineer", "Cloud Engineer"
    },
    "mlops": {
        "MLOps Engineer"
    }
}

# Role aliases map for title/role normalization and semantic matching
ROLE_ALIASES = {
    "Software Engineer": [
        "software engineer", "software developer", "sde", "swe", "software development engineer",
        "systems engineer", "programmer", "software programmer", "application developer", "core engineer"
    ],
    "Python Developer": [
        "python developer", "python engineer", "python backend", "python programmer", "python software engineer"
    ],
    "Backend Developer": [
        "backend developer", "backend engineer", "back end developer", "server engineer", "api developer"
    ],
    "Full Stack Developer": [
        "full stack developer", "fullstack developer", "full-stack developer", "full stack engineer",
        "fullstack engineer", "web developer", "mern developer", "mean developer"
    ],
    "AI Engineer": [
        "ai engineer", "artificial intelligence engineer", "ai developer", "applied ai",
        "ai specialist", "ai researcher", "genai engineer", "generative ai engineer"
    ],
    "Machine Learning Engineer": [
        "machine learning engineer", "ml engineer", "deep learning engineer", "ml developer",
        "machine learning specialist", "applied machine learning"
    ],
    "Data Scientist": [
        "data scientist", "data science", "decision scientist", "applied scientist", "statistical modeler"
    ],
    "Data Analyst": [
        "data analyst", "business intelligence analyst", "bi analyst", "analytics consultant",
        "data analytics specialist", "reporting analyst"
    ],
    "Data Engineer": [
        "data engineer", "big data engineer", "data platform engineer", "etl developer",
        "data warehouse engineer", "data infrastructure engineer"
    ],
    "DevOps Engineer": [
        "devops engineer", "devops", "site reliability engineer", "sre", "platform engineer",
        "infrastructure engineer", "build and release engineer"
    ],
    "Cloud Engineer": [
        "cloud engineer", "cloud architect", "aws engineer", "azure engineer",
        "cloud infrastructure engineer", "cloud solutions engineer"
    ],
    "MLOps Engineer": [
        "mlops engineer", "machine learning operations", "ml platform engineer", "ai ops"
    ],
    "Generative AI Engineer": [
        "generative ai engineer", "genai engineer", "llm engineer", "prompt engineer",
        "llm application developer", "generative ai specialist"
    ],
    "NLP Engineer": [
        "nlp engineer", "natural language processing engineer", "computational linguist",
        "nlp specialist", "text analytics engineer"
    ],
    "Computer Vision Engineer": [
        "computer vision engineer", "cv engineer", "image processing engineer", "vision engineer"
    ],
    "LLM Application Developer": [
        "llm application developer", "llm developer", "prompt engineer", "genai developer"
    ]
}

def clean_text(text: str) -> str:
    """Cleans input text by removing excessive whitespace, URLs, and unusual control characters."""
    if not text:
        return ""
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[\r\t\f\v]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_skill(skill: str) -> str:
    """Normalizes a skill string using lowercasing, stripping, and alias resolution."""
    cleaned = skill.strip().lower()
    return SKILL_ALIASES.get(cleaned, cleaned)

def extract_skills(text: str) -> List[str]:
    """Deterministically extracts identified skills from input text using word-boundary matching.
    Preserves unique normalized skills list.
    """
    if not text:
        return []

    cleaned = clean_text(text).lower()
    found_skills: Set[str] = set()

    # Match aliases first
    for alias, canonical in SKILL_ALIASES.items():
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(alias) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, cleaned):
            found_skills.add(canonical)

    # Match canonical skills
    for skill in CANONICAL_SKILLS:
        pattern = r'(?<![a-zA-Z0-9])' + re.escape(skill) + r'(?![a-zA-Z0-9])'
        if re.search(pattern, cleaned):
            found_skills.add(skill)

    return sorted(list(found_skills))

def parse_job_required_skills(skills_str: str) -> List[str]:
    """Parses comma- or pipe-delimited required skills string into normalized list."""
    if not skills_str or not skills_str.strip():
        return []
    raw_tokens = re.split(r'[,|;]', skills_str)
    result = []
    for token in raw_tokens:
        tok = token.strip()
        if tok:
            normalized = normalize_skill(tok)
            result.append(normalized)
    return list(dict.fromkeys(result))

def analyze_skill_gap(resume_skills: List[str], job_skills: List[str]) -> Tuple[List[str], List[str]]:
    """Deterministically compares resume skills against job required skills.
    Returns (matched_skills, missing_skills).
    """
    resume_set = {normalize_skill(s) for s in resume_skills}
    matched: List[str] = []
    missing: List[str] = []

    for req_skill in job_skills:
        normalized_req = normalize_skill(req_skill)
        if normalized_req in resume_set or any(normalized_req in s or s in normalized_req for s in resume_set):
            matched.append(req_skill.strip())
        else:
            missing.append(req_skill.strip())

    return matched, missing

def normalize_role(role: Optional[str]) -> str:
    """Normalizes role name against known supported IT roles."""
    if not role or not isinstance(role, str):
        return "Software Engineer"
    r_clean = role.strip().lower()
    for canonical in ROLE_SKILL_MATRIX:
        if canonical.lower() == r_clean:
            return canonical
    for canonical, aliases in ROLE_ALIASES.items():
        if any(a == r_clean or a in r_clean for a in aliases):
            return canonical
    return role.strip()

def is_role_compatible(target_role: Optional[str], job_title: str, job_role: str = "") -> Tuple[bool, float, str]:
    """
    Evaluates whether a candidate job is genuinely relevant and compatible with target_role.
    Returns (is_compatible: bool, role_score: float, role_label: str).
    Ensures role relevance strongly gates matching and prevents unrelated cross-discipline contamination.
    """
    if not target_role:
        return True, 75.0, "Moderate"

    canonical_target = normalize_role(target_role)
    t_norm = canonical_target.lower()
    title_norm = (job_title or "").strip().lower()
    role_norm = (job_role or "").strip().lower()
    combined_job = f"{title_norm} {role_norm}".strip()

    target_aliases = ROLE_ALIASES.get(canonical_target, [t_norm])

    # 1. Exact or Alias Match on Target Role (100% score)
    if any(alias in combined_job for alias in target_aliases):
        return True, 100.0, "High"

    # 2. Check Discipline / Cluster Compatibility
    target_disciplines = [
        disc for disc, roles in ROLE_DISCIPLINES.items()
        if canonical_target in roles
    ]
    if not target_disciplines:
        target_disciplines = ["software"]

    # 3. Check for direct negative exclusions based on target discipline
    if "software" in target_disciplines:
        unrelated_titles = [
            "ai engineer", "artificial intelligence", "machine learning", "ml engineer",
            "deep learning", "nlp", "computer vision", "generative ai", "genai", "prompt engineer",
            "llm engineer", "data scientist", "data analyst", "bi analyst", "business intelligence",
            "clinical", "nurse", "sales", "marketing", "recruiter", "accountant",
            "graphic designer", "content writer"
        ]
        if any(unrelated in title_norm for unrelated in unrelated_titles):
            if not any(alias in title_norm for alias in target_aliases):
                return False, 25.0, "Low"

    if "ai_ml" in target_disciplines:
        unrelated_titles = [
            "data scientist", "data analyst", "bi analyst", "software engineer",
            "backend developer", "frontend developer", "full stack",
            "web developer", "mobile developer", "ios developer", "android developer",
            "devops engineer", "cloud engineer", "sre", "systems engineer", "firmware",
            "sales", "marketing", "accountant", "recruiter", "graphic designer"
        ]
        if any(unrelated in title_norm for unrelated in unrelated_titles):
            if not any(alias in title_norm for alias in target_aliases):
                return False, 25.0, "Low"

    if "data_analytics" in target_disciplines or canonical_target == "Data Analyst":
        unrelated_titles = [
            "full stack", "frontend", "devops", "cloud engineer", "firmware", "embedded"
        ]
        if any(unrelated in title_norm for unrelated in unrelated_titles):
            return False, 25.0, "Low"

    if "devops_cloud" in target_disciplines:
        unrelated_titles = [
            "data scientist", "data analyst", "frontend developer", "sales", "marketing",
            "nlp engineer", "computer vision"
        ]
        if any(unrelated in title_norm for unrelated in unrelated_titles):
            return False, 25.0, "Low"

    # 4. Check if job matches any other role within the SAME discipline (85% score)
    for disc in target_disciplines:
        discipline_roles = ROLE_DISCIPLINES.get(disc, set())
        for adj_role in discipline_roles:
            if adj_role == canonical_target:
                continue
            adj_aliases = ROLE_ALIASES.get(adj_role, [adj_role.lower()])
            if any(alias in combined_job for alias in adj_aliases):
                return True, 85.0, "High"

    # 5. General engineering title fallback ONLY within same technical scope
    # Ensure job doesn't belong to a different technical discipline
    job_disciplines = set()
    for disc, roles in ROLE_DISCIPLINES.items():
        for r in roles:
            r_aliases = ROLE_ALIASES.get(r, [r.lower()])
            if any(alias in combined_job for alias in r_aliases):
                job_disciplines.add(disc)

    if job_disciplines and not any(d in target_disciplines for d in job_disciplines):
        # Job belongs to an identified opposing discipline, do not allow generic keyword overlap
        return False, 25.0, "Low"

    general_eng_terms = ["engineer", "developer", "programmer", "architect", "lead"]
    if any(t in title_norm for t in general_eng_terms) and any(t in t_norm for t in general_eng_terms):
        return True, 70.0, "Moderate"

    return False, 30.0, "Low"

def detect_candidate_role(resume_skills: List[str], extracted_text: str = "") -> str:
    """Detects the most appropriate target role from extracted resume skills and text."""
    normalized_skills = {normalize_skill(s) for s in resume_skills}

    best_role = "Software Engineer"
    max_overlap = -1

    for role, role_skills in ROLE_SKILL_MATRIX.items():
        overlap = sum(1 for s in role_skills if s in normalized_skills)
        if overlap > max_overlap:
            max_overlap = overlap
            best_role = role

    return best_role

def discover_relevant_roles(
    resume_skills: List[str],
    extracted_text: str = "",
    target_role: Optional[str] = None,
    max_roles: int = 3
) -> List[str]:
    """
    Identifies a small, relevant set of job roles grounded in the candidate's actual skills and resume text.
    - If target_role is provided (e.g. user selected 'Software Engineer'), it is prioritized as the first role query,
      and any additional roles are strictly restricted to the same technical discipline.
    - Additional relevant roles are discovered from candidate skills within that discipline.
    - Returns a bounded, deduplicated list of roles (at most max_roles).
    """
    normalized_skills = {normalize_skill(s) for s in resume_skills}
    text_lower = (extracted_text or "").lower()

    scored_roles = []
    canonical_target = normalize_role(target_role) if target_role and target_role.strip() else None

    # Identify allowed discipline roles if target_role is specified
    allowed_roles: Optional[Set[str]] = None
    if canonical_target:
        allowed_roles = set()
        allowed_roles.add(canonical_target)
        for disc, roles in ROLE_DISCIPLINES.items():
            if canonical_target in roles:
                allowed_roles.update(roles)

    for role, role_skills in ROLE_SKILL_MATRIX.items():
        if allowed_roles is not None and role not in allowed_roles:
            continue

        overlap = sum(1 for s in role_skills if s in normalized_skills)
        if not role_skills:
            continue
        coverage = overlap / len(role_skills)

        role_lower = role.lower()
        text_boost = 0.0
        if role_lower in text_lower:
            text_boost += 1.5
        elif any(part in text_lower for part in role_lower.split() if len(part) > 3):
            text_boost += 0.5

        if overlap >= 2 or (overlap >= 1 and text_boost > 0):
            score = (overlap * 2.0) + (coverage * 5.0) + text_boost
            scored_roles.append((role, score, overlap))

    # Sort descending by score, then overlap
    scored_roles.sort(key=lambda x: (x[1], x[2]), reverse=True)

    discovered: List[str] = []

    # If target_role is specified, place it first
    if canonical_target:
        discovered.append(canonical_target)
    elif scored_roles:
        discovered.append(scored_roles[0][0])
    else:
        discovered.append("Software Engineer")

    # Add other top qualifying roles within the allowed discipline up to max_roles
    for role, _, _ in scored_roles:
        if role not in discovered:
            discovered.append(role)
        if len(discovered) >= max_roles:
            break

    return discovered

def recommend_top_roles(
    resume_skills: List[str],
    extracted_text: str = "",
    top_n: int = 5
) -> List[Dict[str, Any]]:
    """
    Analyzes candidate's extracted skills, projects, and resume text to recommend top 3-5 suitable IT roles
    (e.g. Software Engineer, Python Developer, Backend Developer, AI Engineer, Machine Learning Engineer,
    Data Scientist, Data Engineer, Generative AI Engineer).
    Produces meaningful, calibrated fit scores (0-100%) and grounded explanations based on actual matched skills.
    Never hallucinates or fabricates skills.
    """
    resume_skill_set = {normalize_skill(s) for s in resume_skills}
    text_lower = (extracted_text or "").lower()

    scored_roles = []
    for role, role_skills in ROLE_SKILL_MATRIX.items():
        matched = [s for s in role_skills if s in resume_skill_set]
        coverage = len(matched) / len(role_skills) if role_skills else 0.0

        role_lower = role.lower()
        text_match = 1.0 if role_lower in text_lower else (
            0.5 if any(p in text_lower for p in role_lower.split() if len(p) > 3) else 0.0
        )

        if matched:
            raw_fit = 42.0 + (coverage * 42.0) + (text_match * 10.0) + min(6.0, len(matched) * 1.0)
            skills_preview = ", ".join(s.title() for s in matched[:4])
            explanation = f"Strong alignment with {skills_preview} identified in your resume."
        else:
            raw_fit = 25.0 + (text_match * 15.0)
            explanation = f"Foundational IT competencies relevant to {role} roles."

        fit_score = round(min(97.0, max(25.0, raw_fit)), 1)
        scored_roles.append({
            "role": role,
            "fit_score": fit_score,
            "explanation": explanation,
            "matched_skills": [s.title() for s in matched[:4]]
        })

    # Sort primarily by fit score, secondarily by number of matched skills
    scored_roles.sort(key=lambda x: (x["fit_score"], len(x["matched_skills"])), reverse=True)
    return scored_roles[:top_n]

import re
from typing import List, Dict, Tuple, Set

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
    "aws", "azure", "google cloud", "cloud", "spark", "hadoop", "kafka", "snowflake", "cuda"
]

# Normalization aliases map
SKILL_ALIASES = {
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
    "AI Engineer": [
        "python", "pytorch", "transformers", "large language models", "langchain",
        "generative ai", "fastapi", "docker", "rag", "vector db", "qdrant"
    ],
    "Machine Learning Engineer": [
        "python", "machine learning", "deep learning", "tensorflow", "pytorch",
        "scikit-learn", "mlops", "docker", "kubernetes", "pandas", "numpy"
    ],
    "Data Scientist": [
        "python", "r", "sql", "pandas", "numpy", "scikit-learn", "statistics",
        "data analysis", "data analytics", "data visualization", "machine learning"
    ],
    "Backend Developer": [
        "python", "fastapi", "flask", "django", "sql", "postgresql", "mysql",
        "redis", "docker", "rest api", "microservices", "git"
    ],
    "Python Developer": [
        "python", "flask", "django", "sql", "mysql", "git", "rest api", "html", "css"
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

def detect_candidate_role(resume_skills: List[str], extracted_text: str = "") -> str:
    """Detects the most appropriate target role from extracted resume skills and text."""
    normalized_skills = {normalize_skill(s) for s in resume_skills}

    best_role = "AI Engineer"
    max_overlap = -1

    for role, role_skills in ROLE_SKILL_MATRIX.items():
        overlap = sum(1 for s in role_skills if s in normalized_skills)
        if overlap > max_overlap:
            max_overlap = overlap
            best_role = role

    return best_role

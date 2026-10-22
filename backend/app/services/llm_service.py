import os
import time
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger("hireagent.llm")

class CareerExplanationSchema(BaseModel):
    """Pydantic schema enforcing structured, hallucination-resistant LLM outputs."""
    summary: str = Field(..., description="Concise grounded summary of candidate alignment.")
    why_match: List[str] = Field(default_factory=list, description="Key match justifications derived strictly from context.")
    matched_skills: List[str] = Field(default_factory=list, description="Verified matched skills present in candidate profile.")
    missing_skills: List[str] = Field(default_factory=list, description="Verified missing skills required by the job.")
    skill_gap_priority: List[str] = Field(default_factory=list, description="Top skills to bridge first for career progression.")
    recommendation: str = Field(..., description="Actionable career advice to bridge gaps.")

def build_grounded_context_prompt(
    resume_skills: List[str],
    target_role: str,
    job_title: str,
    company: str,
    required_skills: List[str],
    matched_skills: List[str],
    missing_skills: List[str],
    match_percentage: float
) -> str:
    """Constructs a strict, injection-resistant context prompt for grounded recommendations."""
    return f"""You are HireAgent, a professional AI Career Assistant for students and freshers.
Your task is to provide an objective, grounded job match explanation based ONLY on the verified data below.

STRICT RULES:
1. ONLY use the verified facts provided below.
2. Do NOT invent, assume, or hallucinate any skills, experiences, company details, salaries, or URLs.
3. Do NOT recalculate or modify the match percentage.
4. Output MUST be valid JSON conforming to the requested schema.

--- VERIFIED CONTEXT ---
Target Role: {target_role}
Candidate Skills: {', '.join(resume_skills) if resume_skills else 'None extracted'}

Job Title: {job_title}
Company: {company}
Calculated Match Score: {match_percentage}%
Job Required Skills: {', '.join(required_skills)}
Verified Matched Skills: {', '.join(matched_skills) if matched_skills else 'None'}
Verified Missing Skills: {', '.join(missing_skills) if missing_skills else 'None'}
--- END CONTEXT ---

Respond ONLY with a JSON object in this exact format:
{{
  "summary": "2-sentence summary of candidate alignment with this role.",
  "why_match": ["Reason 1 based on verified skills", "Reason 2 based on role alignment"],
  "matched_skills": {json.dumps(matched_skills[:5])},
  "missing_skills": {json.dumps(missing_skills[:5])},
  "skill_gap_priority": ["Skill to learn first", "Skill to learn second"],
  "recommendation": "Actionable career advice for student to bridge the identified missing skills."
}}
"""

class GeminiProvider:
    """Primary LLM provider using Google's official google-genai Python SDK."""
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
        self.model_name = model or model_name or settings.GEMINI_MODEL or os.getenv("GEMINI_MODEL") or "gemini-3.8-flash"
        self._client = None
        self._circuit_open_until = 0.0

    def is_circuit_open(self) -> bool:
        return time.time() < self._circuit_open_until

    def _trip_circuit_breaker(self, duration: float = 30.0) -> None:
        self._circuit_open_until = time.time() + duration

    def is_available(self) -> bool:
        if self.is_circuit_open():
            return False
        return bool(self.api_key and self.api_key.strip())

    def get_client(self):
        if self._client is None and self.is_available():
            from google import genai
            from google.genai import types
            http_opts = types.HttpOptions(
                timeout=4,
                retry_options=types.HttpRetryOptions(attempts=1)
            )
            self._client = genai.Client(api_key=self.api_key, http_options=http_opts)
        return self._client

    def generate(self, prompt: str) -> Optional[CareerExplanationSchema]:
        client = self.get_client()
        if not client:
            return None

        try:
            from google.genai import types
            config = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=600,
                response_mime_type="application/json",
                response_schema=CareerExplanationSchema
            )
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            if response and response.text:
                data = json.loads(response.text)
                self._circuit_open_until = 0.0
                return CareerExplanationSchema.model_validate(data)
        except Exception as e:
            # Trip circuit breaker for 30 seconds on failure so subsequent jobs fast-fallback immediately
            self._circuit_open_until = time.time() + 30.0
            logger.warning(f"Gemini API primary call failed: {type(e).__name__} - {e}. Falling back to Groq.")
            return None
        return None

class GroqProvider:
    """Fallback LLM provider using Groq Cloud."""
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.model_name = model_name or settings.GROQ_MODEL or os.getenv("GROQ_MODEL") or "llama-3.3-70b-versatile"
        self._client = None

    def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())

    def get_client(self):
        if self._client is None and self.is_available():
            from groq import Groq
            self._client = Groq(api_key=self.api_key, timeout=5.0)
        return self._client

    def generate(self, prompt: str) -> Optional[CareerExplanationSchema]:
        client = self.get_client()
        if not client:
            return None

        # Prioritize standard Groq models with fast fallback
        candidate_models = []
        for m in [self.model_name, "llama-3.3-70b-versatile", "llama-3.1-8b-instant"]:
            if m and m not in candidate_models and "gpt-oss" not in m:
                candidate_models.append(m)
        if not candidate_models:
            candidate_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

        for mod in candidate_models:
            try:
                chat = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a grounded career advisor. Return strictly valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    model=mod,
                    temperature=0.2,
                    max_tokens=600,
                    response_format={"type": "json_object"}
                )
                content = chat.choices[0].message.content
                if content and content.strip():
                    data = json.loads(content)
                    return CareerExplanationSchema.model_validate(data)
            except Exception as e:
                logger.info(f"Groq model {mod} fallback attempt note: {e}")
                continue
        return None

# In-memory LRU-style cache for grounded RAG career explanations
_EXPLANATION_CACHE: Dict[Tuple, str] = {}
MAX_EXPLANATION_CACHE_SIZE = 500

class UnifiedLLMService:
    """Orchestrates Gemini as PRIMARY and Groq as FALLBACK with strict Pydantic validation."""
    def __init__(self):
        self.gemini = GeminiProvider()
        self.groq = GroqProvider()

    def generate_career_explanation(
        self,
        resume_skills: List[str],
        target_role: str,
        job_title: str,
        company: str,
        required_skills: List[str],
        matched_skills: List[str],
        missing_skills: List[str],
        match_percentage: float,
        fallback_recommendation: str
    ) -> str:
        # Cache key based on job and skills alignment
        cache_key = (
            job_title.strip().lower(),
            company.strip().lower(),
            tuple(sorted(matched_skills[:4])),
            tuple(sorted(missing_skills[:3])),
            round(match_percentage, 0)
        )
        if cache_key in _EXPLANATION_CACHE:
            return _EXPLANATION_CACHE[cache_key]

        prompt = build_grounded_context_prompt(
            resume_skills=resume_skills,
            target_role=target_role,
            job_title=job_title,
            company=company,
            required_skills=required_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            match_percentage=match_percentage
        )

        # 1. Attempt Gemini PRIMARY
        if self.gemini.is_available():
            try:
                res = self.gemini.generate(prompt)
                if res:
                    out = f"{res.summary} {res.recommendation}".strip()
                    if len(_EXPLANATION_CACHE) < MAX_EXPLANATION_CACHE_SIZE:
                        _EXPLANATION_CACHE[cache_key] = out
                    return out
            except Exception as e:
                logger.warning(f"Gemini execution failed: {e}. Falling back to Groq.")

        # 2. Attempt Groq FALLBACK
        if self.groq.is_available():
            try:
                res = self.groq.generate(prompt)
                if res:
                    out = f"{res.summary} {res.recommendation}".strip()
                    if len(_EXPLANATION_CACHE) < MAX_EXPLANATION_CACHE_SIZE:
                        _EXPLANATION_CACHE[cache_key] = out
                    return out
            except Exception as e:
                logger.warning(f"Groq execution failed: {e}. Using deterministic fallback.")

        # 3. Deterministic Fallback if both fail
        logger.info("Both LLM providers unavailable; using deterministic recommendation summary.")
        return fallback_recommendation

llm_service = UnifiedLLMService()

def generate_grounded_recommendation(
    resume_skills: List[str],
    target_role: str,
    job_title: str,
    company: str,
    required_skills: List[str],
    matched_skills: List[str],
    missing_skills: List[str],
    match_percentage: float,
    fallback_recommendation: str
) -> str:
    """Direct entrypoint maintaining backward-compatibility across all modules."""
    return llm_service.generate_career_explanation(
        resume_skills=resume_skills,
        target_role=target_role,
        job_title=job_title,
        company=company,
        required_skills=required_skills,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        match_percentage=match_percentage,
        fallback_recommendation=fallback_recommendation
    )

import pytest
from unittest.mock import patch, MagicMock
from app.services.llm_service import (
    GeminiProvider,
    GroqProvider,
    UnifiedLLMService,
    CareerExplanationSchema
)
from app.core.config import settings

def test_gemini_configuration():
    """Validates Gemini configuration keys and model specification."""
    assert hasattr(settings, "GEMINI_API_KEY")
    assert hasattr(settings, "GEMINI_MODEL")
    assert settings.GEMINI_MODEL == "gemini-3.8-flash"

def test_career_explanation_schema_validation():
    """Validates Pydantic structured output contract for career explanations."""
    payload = {
        "summary": "Candidate exhibits strong Python and Machine Learning competencies.",
        "why_match": ["Hands-on experience with PyTorch", "FastAPI web services experience"],
        "matched_skills": ["Python", "FastAPI"],
        "missing_skills": ["Docker", "Kubernetes"],
        "skill_gap_priority": ["Docker containerization", "Kubernetes orchestration"],
        "recommendation": "Build and deploy a containerized ML microservice to close the gap."
    }
    validated = CareerExplanationSchema(**payload)
    assert validated.summary.startswith("Candidate exhibits")
    assert len(validated.matched_skills) == 2
    assert "Docker" in validated.missing_skills

def test_gemini_provider_mock_structured_output():
    """Tests GeminiProvider structured output generation when invoked with valid context."""
    provider = GeminiProvider(api_key="mock_key", model_name="gemini-3.8-flash")

    mock_json = """
    {
        "summary": "Exceptional fit for AI Engineer role.",
        "why_match": ["Strong deep learning background"],
        "matched_skills": ["Python", "PyTorch"],
        "missing_skills": ["AWS"],
        "skill_gap_priority": ["AWS deployment"],
        "recommendation": "Great candidate for direct interview."
    }
    """

    with patch.object(provider, "get_client") as mock_client:
        mock_response = MagicMock()
        mock_response.text = mock_json
        mock_client.return_value.models.generate_content.return_value = mock_response

        result = provider.generate("Test prompt context")

        assert result is not None
        assert isinstance(result, CareerExplanationSchema)
        assert "Exceptional fit" in result.summary
        assert "AWS" in result.missing_skills

def test_gemini_to_groq_fallback():
    """Tests that UnifiedLLMService seamlessly falls back to Groq when Gemini fails."""
    service = UnifiedLLMService()

    # Configure mock gemini that fails
    service.gemini.is_available = lambda: True
    service.gemini.generate = MagicMock(return_value=None)

    # Configure mock groq that succeeds
    service.groq.is_available = lambda: True
    mock_groq_res = CareerExplanationSchema(
        summary="Solid fit via Groq fallback analysis.",
        why_match=["Good skill alignment"],
        matched_skills=["Python"],
        missing_skills=[],
        skill_gap_priority=[],
        recommendation="Recommended for screening."
    )
    service.groq.generate = MagicMock(return_value=mock_groq_res)

    explanation = service.generate_career_explanation(
        resume_skills=["Python"],
        target_role="Software Engineer",
        job_title="Software Engineer",
        company="Tech Corp",
        required_skills=["Python"],
        matched_skills=["Python"],
        missing_skills=[],
        match_percentage=100.0,
        fallback_recommendation="Fallback recommendation"
    )

    assert "Solid fit via Groq fallback" in explanation
    assert "Recommended for screening." in explanation

def test_deterministic_fallback_when_both_fail():
    """Tests that UnifiedLLMService returns deterministic fallback string if both Gemini and Groq fail."""
    service = UnifiedLLMService()
    service.gemini.is_available = lambda: True
    service.gemini.generate = MagicMock(return_value=None)
    service.groq.is_available = lambda: True
    service.groq.generate = MagicMock(return_value=None)

    fallback_text = "Deterministic fallback recommendation summary."
    explanation = service.generate_career_explanation(
        resume_skills=["Python"],
        target_role="Developer",
        job_title="Dev",
        company="Co",
        required_skills=["Python"],
        matched_skills=["Python"],
        missing_skills=[],
        match_percentage=80.0,
        fallback_recommendation=fallback_text
    )

    assert explanation == fallback_text

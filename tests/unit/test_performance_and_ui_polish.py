import os
import pytest
from unittest.mock import MagicMock, patch

def test_frontend_health_endpoint():
    """Verify frontend has a working /health endpoint for production monitoring."""
    from frontend.app import app
    client = app.test_client()
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ok"
    assert data["service"] == "hireagent-frontend"

def test_dashboard_template_contains_all_skills_container():
    """Verify dashboard.html includes the expandable skill elements and UI tokens."""
    template_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "templates", "dashboard.html"
    )
    with open(template_path, "r", encoding="utf-8") as f:
        html = f.read()

    assert 'id="skillsCompactView"' in html
    assert 'id="allSkillsContainer"' in html
    assert 'id="toggleAllSkillsBtn"' in html
    assert 'id="collapseSkillsBtn"' in html
    assert 'badge-surface' in html
    assert 'badge-accent' in html

def test_dashboard_js_all_skills_rendering_logic():
    """Verify dashboard.js toggles compact view and all skills container."""
    js_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "frontend", "static", "js", "dashboard.js"
    )
    with open(js_path, "r", encoding="utf-8") as f:
        js = f.read()

    assert "toggleSkillsBtn" in js
    assert "collapseSkillsBtn" in js
    assert "allSkillsContainer" in js
    assert "classList.remove('d-none')" in js
    assert "classList.add('d-none')" in js

def test_batch_embedding_matching_optimization():
    """Verify matching_service uses embed_texts for candidate batching."""
    from backend.app.services.matching_service import match_and_rank_jobs
    import inspect

    source = inspect.getsource(match_and_rank_jobs)
    assert "embed_texts(" in source
    assert "batch" in source.lower() or "candidate_vectors" in source

def test_gemini_circuit_breaker():
    """Verify GeminiProvider trips circuit breaker on connection errors and fast-fails."""
    from backend.app.services.llm_service import GeminiProvider
    import time

    provider = GeminiProvider(api_key="mock_key", model="mock-model")
    assert provider.is_circuit_open() is False

    # Trip breaker
    provider._trip_circuit_breaker(duration=2.0)
    assert provider.is_circuit_open() is True

    # Fast failure without API call
    res = provider.generate("test prompt")
    assert res is None

    # Wait for reset
    time.sleep(2.1)
    assert provider.is_circuit_open() is False

def test_render_yaml_validity():
    """Verify render.yaml exists and contains backend and frontend services."""
    render_yaml_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "render.yaml"
    )
    assert os.path.exists(render_yaml_path)
    with open(render_yaml_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "hireagent-backend" in content
    assert "hireagent-frontend" in content
    assert "uvicorn" in content
    assert "gunicorn" in content
    assert "/health" in content

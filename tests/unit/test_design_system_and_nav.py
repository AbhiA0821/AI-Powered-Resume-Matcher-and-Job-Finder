"""
Tests for Design System WCAG 2.1 AA Contrast Ratios and Navigation Bar.
Validates:
1. Automated WCAG luminance & contrast ratio calculation for all :root tokens.
2. Normal text >= 4.5:1, boundary/focus >= 3:1.
3. No inline style="color:", no opacity on text in templates, no emoji icons in templates.
4. Top navigation routes, exact labels, active page indicator, aria-current="page", and mobile toggle.
"""

import os
import re
import pytest
from unittest.mock import patch
from flask import Flask

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STYLE_CSS_PATH = os.path.join(ROOT_DIR, "frontend", "static", "css", "style.css")
TEMPLATES_DIR = os.path.join(ROOT_DIR, "frontend", "templates")

def hex_to_rgb(hex_str: str):
    """Converts a hex color string to RGB tuple."""
    hex_clean = hex_str.strip().lstrip("#")
    if len(hex_clean) == 3:
        hex_clean = "".join([c * 2 for c in hex_clean])
    r = int(hex_clean[0:2], 16)
    g = int(hex_clean[2:4], 16)
    b = int(hex_clean[4:6], 16)
    return r, g, b

def relative_luminance(rgb):
    """Computes WCAG relative luminance."""
    components = []
    for c in rgb:
        s = c / 255.0
        if s <= 0.03928:
            components.append(s / 12.92)
        else:
            components.append(((s + 0.055) / 1.055) ** 2.4)
    return 0.2126 * components[0] + 0.7152 * components[1] + 0.0722 * components[2]

def contrast_ratio(hex1: str, hex2: str) -> float:
    """Computes WCAG contrast ratio between two hex colors."""
    l1 = relative_luminance(hex_to_rgb(hex1))
    l2 = relative_luminance(hex_to_rgb(hex2))
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)

def parse_root_tokens(css_path: str):
    """Parses :root CSS custom properties from stylesheet."""
    tokens = {}
    with open(css_path, "r", encoding="utf-8") as f:
        content = f.read()

    root_match = re.search(r":root\s*\{([^}]+)\}", content, re.DOTALL)
    assert root_match, "No :root token block found in style.css"
    root_block = root_match.group(1)

    for line in root_block.split(";"):
        clean_line = re.sub(r"/\*.*?\*/", "", line, flags=re.DOTALL).strip()
        if clean_line.startswith("--"):
            parts = clean_line.split(":", 1)
            if len(parts) == 2:
                name = parts[0].strip()
                val = parts[1].strip().split("/*")[0].strip()
                if val.startswith("#"):
                    tokens[name] = val
    return tokens


# ── 1. Automated WCAG Contrast Ratio Verification ──

def test_wcag_contrast_ratios_meet_aa_standards():
    """Validates that all core text and boundary tokens meet WCAG 2.1 AA contrast requirements."""
    tokens = parse_root_tokens(STYLE_CSS_PATH)

    # Core background surfaces
    bg = tokens.get("--bg", "#0A0F1E")
    surface = tokens.get("--surface", "#111A2E")
    surface_2 = tokens.get("--surface-2", "#18233B")
    input_bg = tokens.get("--input-bg", "#0D1527")

    # Core text tokens
    text_primary = tokens.get("--text-primary", "#F1F5F9")
    text_secondary = tokens.get("--text-secondary", "#CBD5E1")
    text_muted = tokens.get("--text-muted", "#94A3B8")
    placeholder = tokens.get("--placeholder", "#8FA0BA")
    accent = tokens.get("--accent", "#4F46E5")
    accent_text = tokens.get("--accent-text", "#A5B4FC")
    ai_cyan = tokens.get("--ai-cyan", "#22D3EE")

    # High-contrast button text on accent: for light accent (yellow), text is dark (accent_text); for dark accent, text is white
    button_fg = accent_text if relative_luminance(hex_to_rgb(accent)) > 0.4 else "#ffffff"
    accent_surface_fg = accent if relative_luminance(hex_to_rgb(accent)) > 0.4 else accent_text

    # Normal text test pairs (Required >= 4.5:1)
    normal_text_pairs = [
        ("text-primary on bg", text_primary, bg),
        ("text-primary on surface", text_primary, surface),
        ("text-primary on surface-2", text_primary, surface_2),
        ("text-secondary on bg", text_secondary, bg),
        ("text-secondary on surface", text_secondary, surface),
        ("text-secondary on surface-2", text_secondary, surface_2),
        ("text-muted on bg", text_muted, bg),
        ("text-muted on surface", text_muted, surface),
        ("placeholder on input-bg", placeholder, input_bg),
        ("button text on accent", button_fg, accent),
        ("accent on surface", accent_surface_fg, surface),
        ("ai-cyan on surface", ai_cyan, surface),
    ]

    for label, fg, bg_color in normal_text_pairs:
        ratio = contrast_ratio(fg, bg_color)
        assert ratio >= 4.5, f"WCAG Fail: {label} ({fg} on {bg_color}) ratio {ratio:.2f}:1 is below 4.5:1"

    # UI boundaries / focus ring pairs (Required >= 3:1)
    boundary_pairs = [
        ("ai-cyan focus ring on surface", ai_cyan, surface),
        ("border-strong on surface", tokens.get("--border-strong", "#3A4A70"), surface)
    ]

    for label, fg, bg_color in boundary_pairs:
        ratio = contrast_ratio(fg, bg_color)
        assert ratio >= 3.0, f"WCAG Boundary Fail: {label} ratio {ratio:.2f}:1 is below 3.0:1"


# ── 2. Template Style & Icon Safety ──

def test_no_inline_color_or_opacity_on_text():
    """Asserts no template uses inline style='color:' or opacity on text."""
    for root, _, files in os.walk(TEMPLATES_DIR):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                assert 'style="color:' not in content.lower(), f"Forbidden inline style='color:' found in {file}"
                # Assert no style="opacity:" on text elements
                assert 'style="opacity:' not in content.lower(), f"Forbidden inline style='opacity:' found in {file}"

def test_no_emoji_icons_remain_in_templates():
    """Asserts all emoji icons have been replaced with SVG icons in templates."""
    common_emojis = ["🎯", "✨", "🔐", "⚡", "🧭", "💼", "🏢", "📍", "📅", "💡", "⚙️", "✓", "✗", "↗"]
    for root, _, files in os.walk(TEMPLATES_DIR):
        for file in files:
            if file.endswith(".html"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                for emoji in common_emojis:
                    assert emoji not in content, f"Emoji icon '{emoji}' still present in template: {file}"


# ── 3. Top Navigation Bar Testing ──

@pytest.fixture
def flask_client():
    from frontend.app import app as flask_app
    flask_app.config["TESTING"] = True
    return flask_app.test_client()

def test_navigation_logged_out_shows_public_links(flask_client):
    """Logged out navigation must display Overview, How it works, Features, Sign In, and Create Account."""
    res = flask_client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    assert "Overview" in html
    assert "How it works" in html
    assert "Features" in html
    assert "Sign In" in html
    assert "Create Account" in html
    # Protected links should not be present
    assert 'href="/resume"' not in html
    assert 'href="/skill-gap"' not in html

def test_navigation_logged_in_shows_all_workspace_pages(flask_client):
    """Logged in navigation must show primary Resume Matcher, Jobs, History, and Sign Out links."""
    with flask_client.session_transaction() as sess:
        sess["token"] = "fake_jwt"
        sess["user"] = {"name": "Candidate", "email": "candidate@example.com"}

    with patch("frontend.api_client.get_user_resumes", return_value=([], None)), \
         patch("frontend.api_client.get_user_profile", return_value=({}, None)), \
         patch("frontend.api_client.get_jsearch_quota", return_value={"requests_used": 8, "monthly_limit": 200, "status": "normal", "requests_remaining": 192}):

        res = flask_client.get("/dashboard")
        assert res.status_code == 200
        html = res.data.decode("utf-8")

        assert "Resume Matcher" in html
        assert "Jobs" in html
        assert "History" in html
        assert "Sign Out" in html

        # Active page indicator & aria-current
        assert 'aria-current="page"' in html
        # Accessible mobile toggle button
        assert 'id="navToggle"' in html
        assert 'aria-controls="navMenu"' in html
        # Linked accessible nav script
        assert "js/nav.js" in html

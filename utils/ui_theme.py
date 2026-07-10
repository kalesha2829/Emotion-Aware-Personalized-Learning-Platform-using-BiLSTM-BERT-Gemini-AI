"""Shared styling helpers for the Streamlit UI."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  color-scheme: dark;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background: #060b16;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at top left, rgba(139, 92, 246, 0.16), transparent 26%),
    linear-gradient(135deg, #0b1020 0%, #0f172a 100%);
}

.stApp {
  background: transparent;
}

.block-container {
  padding-top: 1.4rem;
  padding-bottom: 2rem;
  max-width: 1440px;
}

# Hide only footer, header and deploy button; keep Streamlit main menu and toolbar visible
/* (removing the rules that hid the main menu/toolbar so the default page navigation appears). */
footer, header, .stDeployButton {
  visibility: hidden;
}

div[data-testid="stSidebar"] {
  background: rgba(7, 12, 24, 0.96);
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  backdrop-filter: blur(24px);
  width: 300px !important;
  min-width: 300px !important;
}

[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"],
button[kind="header"][data-testid="baseButton-headerNoPadding"] {
  display: none !important;
}

[data-testid="stSidebarContent"] {
  padding: 1rem 0.8rem 1.2rem;
}

[data-testid="stSidebarNav"] {
  display: none;
}

.left-nav-link {
  display: block;
  border-radius: 12px;
  margin: 0.2rem 0;
  padding: 0.7rem 0.8rem;
  color: rgba(255, 255, 255, 0.75);
  text-decoration: none;
  transition: all 0.2s ease;
}

.left-nav-link:hover {
  background: rgba(139, 92, 246, 0.16);
  color: #fff;
  transform: translateX(2px);
}

.left-nav-link.active {
  background: rgba(139, 92, 246, 0.22);
  color: #fff;
  border: 1px solid rgba(139, 92, 246, 0.35);
  box-shadow: 0 0 0 1px rgba(139, 92, 246, 0.12);
  font-weight: 600;
}

.left-nav-panel {
  margin-bottom: 0.9rem;
}

.stButton > button,
button[kind="primary"],
button[kind="secondary"] {
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  background: linear-gradient(135deg, #8b5cf6 0%, #6366f1 100%);
  color: white;
  padding: 0.72rem 1.1rem;
  font-weight: 600;
  box-shadow: 0 12px 30px rgba(99, 102, 241, 0.22);
  transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}

.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 36px rgba(99, 102, 241, 0.28);
  border-color: rgba(255, 255, 255, 0.16);
}

.stTextInput > div > div > input,
.stTextArea > div > textarea {
  background: rgba(17, 24, 39, 0.9);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 16px;
  color: #f8fafc;
  padding: 0.9rem 1rem;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}

.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
  border-color: rgba(139, 92, 246, 0.65);
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.16);
}

.stMetric {
  background: rgba(17, 24, 39, 0.82);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 16px;
  padding: 0.9rem 1rem;
  backdrop-filter: blur(18px);
  box-shadow: 0 12px 30px rgba(2, 6, 23, 0.2);
}

[data-testid="stMetricValue"] {
  color: #f8fafc;
}

[data-testid="stMetricLabel"] {
  color: rgba(255,255,255,0.64);
}

.hero-card,
.feature-card,
.metric-card,
.prediction-card,
.sidebar-card,
.chart-card,
.chat-shell,
.resource-card {
  background: linear-gradient(135deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.9));
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 20px;
  box-shadow: 0 18px 50px rgba(2, 6, 23, 0.28);
  backdrop-filter: blur(18px);
}

.hero-card {
  padding: 1.5rem;
  margin-bottom: 1rem;
  position: relative;
  overflow: hidden;
}

.hero-card::before {
  content: "";
  position: absolute;
  inset: auto auto -40px -40px;
  width: 180px;
  height: 180px;
  background: radial-gradient(circle, rgba(139,92,246,0.26), transparent 70%);
  filter: blur(12px);
}

.feature-card,
.metric-card,
.prediction-card,
.sidebar-card,
.chart-card,
.resource-card {
  padding: 1.1rem;
}

.feature-card:hover,
.metric-card:hover,
.prediction-card:hover,
.resource-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 24px 60px rgba(99, 102, 241, 0.16);
  border-color: rgba(139, 92, 246, 0.35);
}

.page-title {
  font-size: 2.3rem;
  font-weight: 700;
  color: #f8fafc;
  letter-spacing: -0.02em;
}

.page-subtitle {
  color: rgba(255,255,255,0.72);
  font-size: 1rem;
  margin-top: 0.35rem;
  max-width: 760px;
}

.section-label {
  color: rgba(255,255,255,0.64);
  text-transform: uppercase;
  letter-spacing: 0.22em;
  font-size: 0.75rem;
  margin-bottom: 0.7rem;
}

.glow-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  font-size: 1.2rem;
  border-radius: 14px;
  background: rgba(139, 92, 246, 0.18);
  box-shadow: 0 0 0 1px rgba(255,255,255,0.06), inset 0 1px 0 rgba(255,255,255,0.04);
}

.hero-visual {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 260px;
  border-radius: 24px;
  background: radial-gradient(circle at 30% 30%, rgba(139, 92, 246, 0.25), transparent 40%), rgba(9, 14, 28, 0.8);
  border: 1px solid rgba(255,255,255,0.08);
  position: relative;
  overflow: hidden;
}

.brain-orb {
  font-size: 6rem;
  animation: float 4s ease-in-out infinite;
  filter: drop-shadow(0 0 22px rgba(139,92,246,0.45));
}

.particle {
  position: absolute;
  width: 8px;
  height: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.7);
  opacity: 0.6;
  animation: drift 5s ease-in-out infinite;
}

.particle.one { top: 18%; right: 16%; animation-delay: 0s; }
.particle.two { top: 44%; left: 18%; animation-delay: 1s; }
.particle.three { bottom: 20%; right: 20%; animation-delay: 2s; }

.chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-top: 0.9rem;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  padding: 0.45rem 0.7rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.06);
  color: rgba(255,255,255,0.8);
  font-size: 0.9rem;
}

.chat-shell {
  padding: 1rem;
}

.chat-bubble-user,
.chat-bubble-assistant {
  border-radius: 16px;
  padding: 0.95rem 1rem;
  margin-bottom: 0.8rem;
  border: 1px solid rgba(255,255,255,0.06);
}

.chat-bubble-user {
  background: rgba(139, 92, 246, 0.16);
  margin-left: 2rem;
}

.chat-bubble-assistant {
  background: rgba(255,255,255,0.03);
  margin-right: 2rem;
}

.progress-pill {
  height: 8px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
  margin-top: 0.35rem;
}

.progress-fill {
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #8b5cf6, #6366f1);
}

@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50% { transform: translateY(-8px); }
}

@keyframes drift {
  0%, 100% { transform: translateY(0px) scale(1); opacity: 0.5; }
  50% { transform: translateY(-10px) scale(1.2); opacity: 0.9; }
}
"""


def apply_theme() -> None:
    """Inject the shared dark theme and component polish."""
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _current_nav_key() -> str:
    """Return the active navigation key based on the running page script."""
    main = sys.modules.get("__main__")
    if not main or not getattr(main, "__file__", None):
        return "home"

    stem = Path(main.__file__).stem
    if stem == "app":
        return "home"
    if "Emotion_Detection" in stem:
        return "emotion"
    if "Analytics" in stem:
        return "analytics"
    return ""


def render_sidebar() -> None:
    """Render a polished sidebar for the application."""
    active_key = _current_nav_key()
    nav_links = [
        ("home", "🏠 Home", "/"),
        ("emotion", "😊 Emotion Detection", "/Emotion_Detection"),
        ("analytics", "📊 Analytics Dashboard", "/Analytics"),
    ]

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card" style="padding: 0.95rem; margin-bottom: 0.9rem;">
              <div style="display:flex; align-items:center; gap:0.7rem; font-weight:700; font-size:1.05rem; color:#f8fafc;">
                <div class="glow-icon">🧠</div>
                <div>AI Learning Assistant</div>
              </div>
              <div style="margin-top:0.75rem; color:rgba(255,255,255,0.68); font-size:0.94rem;">
                Emotion-aware learning support for modern students.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        nav_html = ['<div class="left-nav-panel">', '<div class="section-label">Navigation</div>']
        for key, label, href in nav_links:
            active_class = " active" if key == active_key else ""
            nav_html.append(
                f'<a class="left-nav-link{active_class}" href="{href}" target="_self">{label}</a>'
            )
        nav_html.append("</div>")
        st.markdown("".join(nav_html), unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-card" style="margin-top: 1rem; padding: 0.95rem;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <div style="font-weight:700; color:#f8fafc;">Learning Streak</div>
                <div style="color:#8b5cf6; font-weight:700;">7 days</div>
              </div>
              <div style="color:rgba(255,255,255,0.7); font-size:0.92rem;">Today's motivation</div>
              <div style="margin-top:0.7rem; color:#f8fafc; font-weight:600;">Keep going — small steps create strong momentum.</div>
              <div class="progress-pill" style="margin-top:0.8rem;"><div class="progress-fill" style="width: 74%;"></div></div>
              <div style="margin-top:0.4rem; color:rgba(255,255,255,0.6); font-size:0.82rem;">Weekly progress 74%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

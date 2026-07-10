"""AI Learning Assistant — main Streamlit entry point."""

import streamlit as st

from utils.ui_theme import apply_theme, render_sidebar

st.set_page_config(
    page_title="AI Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()

st.markdown(
    """
    <div class="hero-card">
      <div class="section-label">AI LEARNING ASSISTANT</div>
      <div style="display:flex; flex-wrap:wrap; gap:1.4rem; align-items:center; justify-content:space-between;">
        <div style="flex: 1.2; min-width: 320px;">
          <div class="page-title">Emotion-Aware Personalized Learning Platform</div>
          <div class="page-subtitle" style="margin-top:0.8rem; font-size:1.02rem;">
            Understand your emotions, receive personalized AI guidance, and improve your learning journey using BiLSTM, BERT, and Gemini AI.
          </div>
          <div style="margin-top:1rem; display:flex; flex-wrap:wrap; gap:0.7rem;">
              <a href="/Emotion_Detection" target="_self"><button>Start Emotion Detection</button></a>
              <a href="/Analytics" target="_self"><button style="background: rgba(255,255,255,0.04); box-shadow:none;">View Analytics</button></a>
          </div>
          <div class="chip-row">
            <span class="chip">🧠 Emotion-aware</span>
            <span class="chip">✨ AI-guided</span>
            <span class="chip">📈 Insights</span>
          </div>
        </div>
        <div style="flex: 0.8; min-width: 280px;">
          <div class="hero-visual">
            <div class="brain-orb">🧠</div>
            <div class="particle one"></div>
            <div class="particle two"></div>
            <div class="particle three"></div>
          </div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='section-label'>Live snapshot</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'><div class='glow-icon' style='margin-bottom:0.7rem;'>😊</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc;'>Calm</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Current Mood</div></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><div class='glow-icon' style='margin-bottom:0.7rem;'>🔥</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc;'>7 Days</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Learning Streak</div></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><div class='glow-icon' style='margin-bottom:0.7rem;'>📊</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc;'>128</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Sessions Analyzed</div></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'><div class='glow-icon' style='margin-bottom:0.7rem;'>🎯</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc;'>91%</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Average Confidence</div></div>", unsafe_allow_html=True)

st.markdown("<div class='section-label' style='margin-top:1.2rem;'>Core capabilities</div>", unsafe_allow_html=True)
features = st.columns(3)
with features[0]:
  st.markdown("<div class='feature-card'><div class='glow-icon'>😊</div><h3 style='margin:0.6rem 0 0.3rem; color:#f8fafc;'>Emotion Detection</h3><p style='color:rgba(255,255,255,0.72); line-height:1.6;'>Capture subtle emotional signals from study reflections and receive an explainable prediction.</p><a href='/Emotion_Detection' target='_self'><button style='margin-top:0.6rem;'>Explore</button></a></div>", unsafe_allow_html=True)
with features[1]:
  st.markdown("<div class='feature-card'><div class='glow-icon'>🎓</div><h3 style='margin:0.6rem 0 0.3rem; color:#f8fafc;'>Learning Support</h3><p style='color:rgba(255,255,255,0.72); line-height:1.6;'>Receive calm, personalized study guidance and actionable next steps tailored to your mood.</p><a href='/Learning_Support' target='_self'><button style='margin-top:0.6rem;'>Open Coach</button></a></div>", unsafe_allow_html=True)
with features[2]:
  st.markdown("<div class='feature-card'><div class='glow-icon'>📊</div><h3 style='margin:0.6rem 0 0.3rem; color:#f8fafc;'>Analytics Dashboard</h3><p style='color:rgba(255,255,255,0.72); line-height:1.6;'>Track confidence, model agreement, and emotion distributions in a refined view.</p><a href='/Analytics' target='_self'><button style='margin-top:0.6rem;'>View Trends</button></a></div>", unsafe_allow_html=True)

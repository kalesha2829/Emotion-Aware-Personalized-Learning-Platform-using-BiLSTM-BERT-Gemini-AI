"""Personalized learning support page for the AI Learning Assistant."""

from __future__ import annotations

import streamlit as st

from utils.gemini import generate_gemini_learning_support
from utils.predict import EmotionDetectionPipeline, EmotionDetectionResult
from utils.ui_theme import apply_theme, render_sidebar

st.set_page_config(page_title="Learning Support", page_icon="💡", layout="wide", initial_sidebar_state="expanded")

apply_theme()
render_sidebar()

st.markdown("<div class='page-title'>Personalized Learning Support</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Chat with the assistant to receive targeted study advice, tailored encouragement, and concrete next steps.</div>", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> EmotionDetectionPipeline:
    return EmotionDetectionPipeline()


def show_support(result: EmotionDetectionResult, text: str) -> None:
    st.markdown("<div class='section-label'>Detected emotion</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='prediction-card'><div style='font-size:1.2rem; font-weight:700; color:#f8fafc;'>{result.final_emotion}</div><div style='color:#8b5cf6; font-weight:600; margin-top:0.3rem;'>{result.confidence_score:.0%}</div></div>", unsafe_allow_html=True)

    support = generate_gemini_learning_support(result.final_emotion, text)

    if support.error:
        st.warning(
            "Gemini AI could not generate guidance cleanly, so a fallback response is shown."
        )

    st.markdown("<div class='chat-shell' style='margin-top:1rem;'><div class='chat-bubble-user'>You: I’m feeling overwhelmed while preparing for an exam.</div><div class='chat-bubble-assistant'>Assistant: <strong>" + support.personalized_learning_guidance + "</strong></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-card'><div style='font-weight:700; color:#f8fafc;'>Encouragement</div><div style='color:rgba(255,255,255,0.74); margin-top:0.55rem; line-height:1.6;'>" + support.encouraging_response + "</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:1rem;'>Study tips</div>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-card'>" + support.study_tips + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-label' style='margin-top:1rem;'>Recommended resources</div>", unsafe_allow_html=True)
    st.markdown("<div class='prediction-card'>" + support.next_learning_steps + "</div>", unsafe_allow_html=True)

    if support.raw_response and support.error:
        with st.expander("Raw Gemini output"):
            st.code(support.raw_response)


def main() -> None:
    pipeline = get_pipeline()

    if "student_text" in st.session_state and "emotion_result" in st.session_state:
        st.markdown("<div class='prediction-card' style='margin-top:0.8rem;'><div style='font-weight:700; color:#f8fafc;'>Previous input</div><div style='color:rgba(255,255,255,0.72); margin-top:0.45rem;'>" + st.session_state["student_text"] + "</div></div>", unsafe_allow_html=True)
        if st.button("Re-run guidance on this input"):
            with st.spinner("Regenerating Gemini guidance..."):
                show_support(st.session_state["emotion_result"], st.session_state["student_text"])
        else:
            show_support(st.session_state["emotion_result"], st.session_state["student_text"])
        return

    with st.form("support_form"):
        text = st.text_area(
            "",
            height=220,
            placeholder="Enter study-related reflections, stressors, or learning questions...",
        )
        submit = st.form_submit_button("Generate guidance")

    if submit:
        if not text.strip():
            st.warning("Please enter study-related text to get personalized learning support.")
            return

        with st.spinner("Analyzing emotion and requesting Gemini guidance..."):
            try:
                result = pipeline.predict(text)
                show_support(result, text)
                st.session_state["student_text"] = text
                st.session_state["emotion_result"] = result
            except Exception as exc:
                st.error("Unable to generate guidance at this time. Please try again later.")
                st.error(str(exc))


if __name__ == "__main__":
    main()

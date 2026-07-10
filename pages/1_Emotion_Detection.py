"""Emotion detection page for the AI Learning Assistant."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any

from utils.gemini import generate_gemini_learning_support
from utils.logger import log_interaction
from utils.predict import EmotionDetectionPipeline, EmotionDetectionResult
from utils.ui_theme import apply_theme, render_sidebar

st.set_page_config(page_title="Emotion Detection", page_icon="🎭", layout="wide", initial_sidebar_state="expanded")

apply_theme()
render_sidebar()

st.markdown("<div class='page-title'>Emotion Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Enter study-related text and receive a refined emotion analysis from BiLSTM, BERT, and Gemini AI.</div>", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline() -> EmotionDetectionPipeline:
    return EmotionDetectionPipeline()


def render_prediction_summary(result: EmotionDetectionResult) -> None:
    st.markdown("<div class='section-label'>Prediction overview</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='prediction-card'><div class='glow-icon'>🧠</div><div style='font-size:0.9rem; color:rgba(255,255,255,0.64); margin-top:0.5rem;'>BiLSTM</div><div style='font-size:1.2rem; font-weight:700; color:#f8fafc; margin-top:0.2rem;'>{result.bilstm_prediction}</div><div style='color:#8b5cf6; font-weight:600;'>{result.bilstm_confidence:.0%}</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='prediction-card'><div class='glow-icon'>🤖</div><div style='font-size:0.9rem; color:rgba(255,255,255,0.64); margin-top:0.5rem;'>BERT</div><div style='font-size:1.2rem; font-weight:700; color:#f8fafc; margin-top:0.2rem;'>{result.bert_prediction}</div><div style='color:#8b5cf6; font-weight:600;'>{result.bert_confidence:.0%}</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='prediction-card'><div class='glow-icon'>✨</div><div style='font-size:0.9rem; color:rgba(255,255,255,0.64); margin-top:0.5rem;'>Final</div><div style='font-size:1.2rem; font-weight:700; color:#f8fafc; margin-top:0.2rem;'>{result.final_emotion}</div><div style='color:#8b5cf6; font-weight:600;'>{result.confidence_score:.0%}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label' style='margin-top:1.1rem;'>Mixed emotion breakdown</div>", unsafe_allow_html=True)
    if result.mixed_emotion_breakdown:
        breakdown_df = pd.DataFrame(result.mixed_emotion_breakdown)
        breakdown_df["confidence"] = breakdown_df["confidence"].map(lambda x: f"{x:.1%}")
        st.dataframe(breakdown_df.rename(columns={"emotion": "Emotion", "confidence": "Confidence"}), use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='prediction-card' style='margin-top:0.5rem;'>No mixed emotions were detected for this input.</div>", unsafe_allow_html=True)


def render_gemini_support(support: Any) -> None:
    st.markdown("<div class='section-label' style='margin-top:1.1rem;'>Gemini recommendation</div>", unsafe_allow_html=True)
    if support.error:
        st.warning(
            "Gemini AI returned an unexpected response or could not be reached. "
            "A fallback message is displayed below."
        )

    guidance_cols = st.columns(2)
    with guidance_cols[0]:
        st.markdown("<div class='prediction-card'><div style='font-weight:700; color:#f8fafc;'>Guidance</div><div style='color:rgba(255,255,255,0.74); margin-top:0.55rem; line-height:1.6;'>" + support.personalized_learning_guidance + "</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='prediction-card' style='margin-top:0.7rem;'><div style='font-weight:700; color:#f8fafc;'>Encouragement</div><div style='color:rgba(255,255,255,0.74); margin-top:0.55rem; line-height:1.6;'>" + support.encouraging_response + "</div></div>", unsafe_allow_html=True)
    with guidance_cols[1]:
        st.markdown("<div class='prediction-card'><div style='font-weight:700; color:#f8fafc;'>Study tips</div><div style='color:rgba(255,255,255,0.74); margin-top:0.55rem; line-height:1.6;'>" + support.study_tips + "</div></div>", unsafe_allow_html=True)
        st.markdown("<div class='prediction-card' style='margin-top:0.7rem;'><div style='font-weight:700; color:#f8fafc;'>Next learning steps</div><div style='color:rgba(255,255,255,0.74); margin-top:0.55rem; line-height:1.6;'>" + support.next_learning_steps + "</div></div>", unsafe_allow_html=True)

    if support.raw_response and support.error:
        with st.expander("Raw Gemini output"):
            st.code(support.raw_response)


def main() -> None:
    pipeline = get_pipeline()

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown("<div class='prediction-card'><div class='section-label'>Text input</div><div style='font-weight:700; color:#f8fafc; font-size:1.02rem;'>Describe how you feel about your current learning experience.</div></div>", unsafe_allow_html=True)
        with st.form("emotion_form"):
            text = st.text_area("", height=220, placeholder="Type study-related thoughts, questions, or reflections here...")
            st.markdown("<div class='chip-row'><span class='chip'>💬 Need motivation?</span><span class='chip'>📝 Want clarity?</span><span class='chip'>😌 Feeling stuck?</span></div>", unsafe_allow_html=True)
            analyze = st.form_submit_button("Analyze emotion")

        if not text.strip() and not analyze:
            st.markdown("<div class='prediction-card' style='margin-top:0.8rem;'>Enter a sentence or paragraph about studying to begin emotion detection.</div>", unsafe_allow_html=True)
            return

    with right_col:
        st.markdown("<div class='prediction-card'><div class='section-label'>Prediction card</div><div style='font-size:1.15rem; font-weight:700; color:#f8fafc;'>Detected Emotion</div><div style='color:rgba(255,255,255,0.72); margin-top:0.4rem;'>Confidence, mixed emotion signals, and Gemini guidance will appear here.</div></div>", unsafe_allow_html=True)

    if analyze:
        if not text.strip():
            st.warning("Please enter study-related text before analyzing.")
            return

        with st.spinner("Analyzing the input and generating guidance..."):
            try:
                result = pipeline.predict(text)
                support = generate_gemini_learning_support(result.final_emotion, text)
                st.session_state["emotion_result"] = result
                st.session_state["gemini_support"] = support
                st.session_state["student_text"] = text

                logged, log_error = log_interaction(
                    student_input=text,
                    bilstm_prediction=result.bilstm_prediction,
                    bert_prediction=result.bert_prediction,
                    final_emotion=result.final_emotion,
                    confidence_score=result.confidence_score,
                    mixed_emotion_breakdown=result.mixed_emotion_breakdown,
                    gemini_support=support,
                )
                if not logged:
                    st.warning(
                        "Prediction completed, but interaction logging failed. "
                        "Check file permissions for logs/interactions.csv."
                    )
                    if log_error:
                        st.write(f"Logging error: {log_error}")
            except Exception as exc:
                st.error(
                    "Something went wrong while analyzing your text. Please try again."
                )
                st.error(str(exc))
                return

        with right_col:
            render_prediction_summary(st.session_state["emotion_result"])
            render_gemini_support(st.session_state["gemini_support"])

    elif "emotion_result" in st.session_state:
        with right_col:
            render_prediction_summary(st.session_state["emotion_result"])
            render_gemini_support(st.session_state["gemini_support"])




if __name__ == "__main__":
    main()

import streamlit as st

from utils.analytics import (
    build_emotion_distribution_chart,
    build_confidence_distribution_chart,
    build_model_comparison_chart,
    build_mixed_emotion_frequency_chart,
    build_top_mixed_pairs_chart,
    compute_prediction_analytics,
    load_analysis_dataframe,
)
from utils.predict import EmotionDetectionPipeline
from utils.ui_theme import apply_theme, render_sidebar


st.set_page_config(
    page_title="Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()
render_sidebar()

st.markdown("<div class='page-title'>Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='page-subtitle'>Explore emotion distributions, confidence trends, and model comparisons in a premium dashboard view.</div>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[list[str], list[str]]:
    df = load_analysis_dataframe(max_rows=1000)
    return df, df["target_emotion"].unique().tolist() if not df.empty else []


@st.cache_resource(show_spinner=False)
def load_pipeline() -> EmotionDetectionPipeline:
    return EmotionDetectionPipeline()


data_frame, available_labels = load_data()
if data_frame.empty:
    st.warning("No analytics dataset could be loaded. Please check dataset paths and try again.")
    st.stop()

st.sidebar.markdown("<div class='section-label'>Dataset overview</div>", unsafe_allow_html=True)
st.sidebar.markdown(f"<div class='prediction-card'><div style='color:#f8fafc; font-weight:700;'>Rows available</div><div style='font-size:1.2rem; color:#8b5cf6; font-weight:700; margin-top:0.3rem;'>{len(data_frame):,}</div></div>", unsafe_allow_html=True)
if available_labels:
    st.sidebar.markdown(f"<div class='prediction-card' style='margin-top:0.7rem;'><div style='color:#f8fafc; font-weight:700;'>Emotion labels</div><div style='color:rgba(255,255,255,0.72); margin-top:0.4rem;'>{', '.join(sorted(available_labels))}</div></div>", unsafe_allow_html=True)

predictions_df = None
prediction_error = None
with st.spinner("Calculating analytics with model predictions..."):
    try:
        pipeline = load_pipeline()
        predictions_df = compute_prediction_analytics(data_frame, pipeline, max_rows=200)
    except Exception as exc:
        predictions_df = None
        prediction_error = str(exc)

st.markdown("<div class='section-label' style='margin-top:1rem;'>Top metrics</div>", unsafe_allow_html=True)
metrics = st.columns(4)
with metrics[0]:
    st.markdown("<div class='metric-card'><div class='glow-icon'>📊</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc; margin-top:0.5rem;'>" + str(len(data_frame)) + "</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Dataset Rows</div></div>", unsafe_allow_html=True)
with metrics[1]:
    st.markdown("<div class='metric-card'><div class='glow-icon'>🎯</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc; margin-top:0.5rem;'>Confidence</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Model Trend</div></div>", unsafe_allow_html=True)
with metrics[2]:
    st.markdown("<div class='metric-card'><div class='glow-icon'>🧠</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc; margin-top:0.5rem;'>BiLSTM</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>BERT</div></div>", unsafe_allow_html=True)
with metrics[3]:
    st.markdown("<div class='metric-card'><div class='glow-icon'>✨</div><div style='font-size:1.4rem; font-weight:700; color:#f8fafc; margin-top:0.5rem;'>Mixed</div><div style='color:rgba(255,255,255,0.66); font-size:0.9rem;'>Emotion Insights</div></div>", unsafe_allow_html=True)

st.markdown("<div class='section-label' style='margin-top:1rem;'>Emotion Distribution</div>", unsafe_allow_html=True)
st.plotly_chart(build_emotion_distribution_chart(data_frame), width="stretch")

if predictions_df is None or predictions_df.empty:
    st.warning(
        "Model prediction analytics could not be generated. "
        "Verify model artifacts are present in the `models/` folder."
    )
    if prediction_error:
        st.write(f"**Error:** {prediction_error}")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-label'>Confidence Trend</div>", unsafe_allow_html=True)
        st.plotly_chart(build_confidence_distribution_chart(predictions_df), width="stretch")
    with col2:
        st.markdown("<div class='section-label'>Model Comparison</div>", unsafe_allow_html=True)
        st.plotly_chart(build_model_comparison_chart(predictions_df), width="stretch")

    st.markdown("<div class='section-label' style='margin-top:1rem;'>Recent Predictions</div>", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(build_mixed_emotion_frequency_chart(predictions_df), width="stretch")
    with col4:
        st.plotly_chart(build_top_mixed_pairs_chart(predictions_df), width="stretch")

st.markdown(
    "---\n"
    "**Notes:** The analytics dashboard samples a subset of dataset texts for prediction analysis to keep performance responsive. "
    "If the dataset is large, the current charts reflect the first available rows."
)

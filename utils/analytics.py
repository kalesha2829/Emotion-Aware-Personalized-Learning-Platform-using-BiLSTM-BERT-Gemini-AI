"""Analytics utilities for the AI Learning Assistant dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.constants import (
    CLEANED_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    TEST_DATASET_PATH,
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
)
from utils.predict import EmotionDetectionPipeline

ANALYTICS_DATA_PATHS = [
    PROCESSED_DATASET_PATH,
    CLEANED_DATASET_PATH,
    TEST_DATASET_PATH,
    VAL_DATASET_PATH,
    TRAIN_DATASET_PATH,
]


def load_analysis_dataframe(max_rows: int = 1000) -> pd.DataFrame:
    """Load the most appropriate dataset available for analytics."""
    for path in ANALYTICS_DATA_PATHS:
        if path.exists():
            df = pd.read_csv(path)
            if "text" not in df.columns:
                continue
            if df.empty:
                continue
            df = df.copy()
            if "target_emotion" in df.columns:
                df["target_emotion"] = df["target_emotion"].astype(str)
            elif "label" in df.columns:
                df["target_emotion"] = df["label"].astype(str)
            else:
                df["target_emotion"] = "unknown"
            return df.head(max_rows).reset_index(drop=True)
    return pd.DataFrame(columns=["text", "target_emotion"])


def compute_prediction_analytics(
    df: pd.DataFrame,
    pipeline: EmotionDetectionPipeline,
    max_rows: int = 200,
) -> pd.DataFrame:
    """Run model predictions for a sample of rows and prepare analytics data."""
    if df.empty or "text" not in df.columns:
        return pd.DataFrame()

    sample_df = df.head(max_rows).copy().reset_index(drop=True)
    records: list[dict[str, Any]] = []

    for _, row in sample_df.iterrows():
        text = str(row["text"] or "").strip()
        if not text:
            continue
        prediction = pipeline.predict(text)
        mixed_emotions = prediction.mixed_emotion_breakdown
        records.append(
            {
                "text": text,
                "target_emotion": row.get("target_emotion", "unknown"),
                "bilstm_prediction": prediction.bilstm_prediction,
                "bert_prediction": prediction.bert_prediction,
                "final_emotion": prediction.final_emotion,
                "bilstm_confidence": prediction.bilstm_confidence,
                "bert_confidence": prediction.bert_confidence,
                "ensemble_confidence": prediction.confidence_score,
                "mixed_count": len(mixed_emotions),
                "mixed_emotion_pair": (
                    " / ".join([entry["emotion"] for entry in mixed_emotions])
                    if len(mixed_emotions) > 1
                    else ""
                ),
            }
        )

    return pd.DataFrame(records)


def build_emotion_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Build a Plotly bar chart for target emotion distribution."""
    if df.empty or "target_emotion" not in df.columns:
        return go.Figure()

    distribution = df["target_emotion"].value_counts().rename_axis("emotion").reset_index(name="count")
    fig = px.bar(
        distribution,
        x="emotion",
        y="count",
        color="emotion",
        title="Emotion Distribution",
        labels={"count": "Number of examples", "emotion": "Emotion"},
    )
    fig.update_layout(showlegend=False, xaxis_title="Emotion", yaxis_title="Count")
    return fig


def build_confidence_distribution_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly histogram of confidence scores from both models."""
    if predictions_df.empty:
        return go.Figure()

    confidence_df = predictions_df.melt(
        id_vars=["text"],
        value_vars=["bilstm_confidence", "bert_confidence", "ensemble_confidence"],
        var_name="model",
        value_name="confidence",
    )
    model_labels = {
        "bilstm_confidence": "BiLSTM",
        "bert_confidence": "BERT",
        "ensemble_confidence": "Ensemble",
    }
    confidence_df["model"] = confidence_df["model"].map(model_labels)

    fig = px.histogram(
        confidence_df,
        x="confidence",
        color="model",
        nbins=20,
        barmode="overlay",
        opacity=0.75,
        title="Confidence Score Distribution",
        labels={"confidence": "Confidence", "model": "Model"},
        marginal="rug",
    )
    fig.update_layout(xaxis=dict(tickformat=".0%"), yaxis_title="Frequency")
    return fig


def build_model_comparison_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly heatmap comparing BiLSTM and BERT predictions."""
    if predictions_df.empty:
        return go.Figure()

    cross_tab = pd.crosstab(predictions_df["bilstm_prediction"], predictions_df["bert_prediction"])
    fig = px.imshow(
        cross_tab,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title="BiLSTM vs. BERT Prediction Comparison",
        labels={"x": "BERT prediction", "y": "BiLSTM prediction", "color": "Count"},
    )
    fig.update_layout(xaxis_side="top")
    return fig


def build_mixed_emotion_frequency_chart(predictions_df: pd.DataFrame) -> go.Figure:
    """Build a Plotly chart showing how often mixed emotions are detected."""
    if predictions_df.empty:
        return go.Figure()

    mixed_counts = (
        predictions_df["mixed_count"].apply(lambda count: "Mixed detected" if count > 1 else "Single dominant emotion")
        .value_counts()
        .rename_axis("type")
        .reset_index(name="count")
    )
    fig = px.bar(
        mixed_counts,
        x="type",
        y="count",
        color="type",
        title="Mixed Emotion Frequency",
        labels={"type": "Category", "count": "Number of examples"},
    )
    fig.update_layout(showlegend=False, xaxis_title="Detection type", yaxis_title="Count")
    return fig


def build_top_mixed_pairs_chart(predictions_df: pd.DataFrame, max_pairs: int = 10) -> go.Figure:
    """Build a Plotly bar chart for the most common mixed emotion pairs."""
    if predictions_df.empty or "mixed_emotion_pair" not in predictions_df.columns:
        return go.Figure()

    pairs = (
        predictions_df[predictions_df["mixed_emotion_pair"] != ""]["mixed_emotion_pair"]
        .value_counts()
        .nlargest(max_pairs)
        .rename_axis("pair")
        .reset_index(name="count")
    )

    if pairs.empty:
        return go.Figure()

    fig = px.bar(
        pairs,
        x="count",
        y="pair",
        orientation="h",
        title="Most Common Mixed Emotion Pairs",
        labels={"pair": "Emotion pair", "count": "Count"},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

"""
Plotly visualizations for emotion label distribution.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.constants import EMOTION_BAR_CHART_PATH, EMOTION_PIE_CHART_PATH, TARGET_EMOTIONS


def _prepare_distribution(df: pd.DataFrame, label_col: str = "target_emotion") -> pd.DataFrame:
    """Build a distribution DataFrame with counts and percentages."""
    counts = df[label_col].value_counts().reindex(TARGET_EMOTIONS, fill_value=0)
    total = counts.sum()
    return pd.DataFrame(
        {
            "emotion": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(2) if total else 0,
        }
    )


def plot_emotion_bar_chart(
    df: pd.DataFrame,
    label_col: str = "target_emotion",
    save_path: Path | None = None,
    show: bool = False,
) -> go.Figure:
    """
    Create a bar chart showing the count of each target emotion.

    Saves an interactive HTML file to ``assets/emotion_distribution_bar.html``.
    """
    dist = _prepare_distribution(df, label_col)

    fig = px.bar(
        dist,
        x="emotion",
        y="count",
        color="emotion",
        title="Student Emotion Distribution (Bar Chart)",
        labels={"emotion": "Emotion", "count": "Number of Samples"},
        text="count",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        xaxis_categoryorder="array",
        xaxis={"categoryorder": "array", "categoryarray": TARGET_EMOTIONS},
        yaxis_title="Sample Count",
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    output = save_path or EMOTION_BAR_CHART_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f"  Bar chart saved to: {output}")

    if show:
        fig.show()

    return fig


def plot_emotion_pie_chart(
    df: pd.DataFrame,
    label_col: str = "target_emotion",
    save_path: Path | None = None,
    show: bool = False,
) -> go.Figure:
    """
    Create a pie chart showing the percentage of each target emotion.

    Saves an interactive HTML file to ``assets/emotion_distribution_pie.html``.
    """
    dist = _prepare_distribution(df, label_col)
    dist = dist[dist["count"] > 0]

    fig = px.pie(
        dist,
        names="emotion",
        values="count",
        title="Student Emotion Distribution (Pie Chart)",
        color="emotion",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.35,
    )
    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    )
    fig.update_layout(
        legend_title="Emotion",
        uniformtext_minsize=10,
    )

    output = save_path or EMOTION_PIE_CHART_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    print(f"  Pie chart saved to: {output}")

    if show:
        fig.show()

    return fig


def create_all_visualizations(df: pd.DataFrame, label_col: str = "target_emotion") -> None:
    """Generate and save all emotion distribution visualizations."""
    print("\n--- Generating Visualizations ---")
    plot_emotion_bar_chart(df, label_col)
    plot_emotion_pie_chart(df, label_col)

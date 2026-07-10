"""
Dataset loading utilities for the AI Learning Assistant.

Downloads GoEmotions from Hugging Face (equivalent to the Kaggle release) or loads
a previously saved local CSV copy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.constants import (
    CLEANED_DATASET_PATH,
    HF_DATASET_CONFIG,
    HF_DATASET_NAME,
    RAW_DATASET_PATH,
)
from utils.emotion_mapping import (
    GOEMOTIONS_LABEL_NAMES,
    label_ids_to_names,
    map_source_labels_to_target,
)


def load_goemotions_from_hub() -> pd.DataFrame:
    """
    Download and combine all GoEmotions simplified splits from Hugging Face.

    Returns a DataFrame with columns:
        text, labels, comment_id, split, source_label, source_labels
    """
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET_NAME, HF_DATASET_CONFIG)

    frames: list[pd.DataFrame] = []
    for split_name in dataset.keys():
        split_df = dataset[split_name].to_pandas()
        split_df["split"] = split_name
        frames.append(split_df)

    df = pd.concat(frames, ignore_index=True)
    return _enrich_with_source_labels(df)


def _normalize_label_ids(label_ids: object) -> list[int]:
    """Convert Hugging Face / pandas label values to a flat list of ints."""
    import numpy as np

    if label_ids is None:
        return []
    if isinstance(label_ids, (list, tuple)):
        return [int(v) for v in label_ids]
    if isinstance(label_ids, np.ndarray):
        return [int(v) for v in label_ids.tolist()]
    return [int(label_ids)]


def _enrich_with_source_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add human-readable source label columns derived from GoEmotions label IDs."""
    enriched = df.copy()

    def resolve_labels(label_ids: object) -> tuple[str, str]:
        ids = _normalize_label_ids(label_ids)
        names = label_ids_to_names(ids)
        primary = names[0] if names else "unknown"
        return primary, "|".join(names)

    label_info = enriched["labels"].apply(resolve_labels)
    enriched["source_label"] = label_info.apply(lambda x: x[0])
    enriched["source_labels"] = label_info.apply(lambda x: x[1])
    return enriched


def load_raw_dataset(local_path: Path | None = None) -> pd.DataFrame:
    """
    Load the raw GoEmotions dataset.

    Uses the local CSV at ``local_path`` when present; otherwise downloads from
    Hugging Face and caches the result to ``dataset/go_emotions_raw.csv``.
    """
    path = local_path or RAW_DATASET_PATH

    if path.exists():
        df = pd.read_csv(path)
        if "source_label" not in df.columns:
            df = _enrich_from_csv_labels(df)
        return df

    df = load_goemotions_from_hub()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _enrich_from_csv_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct source label columns when loading from a saved CSV."""
    enriched = df.copy()

    def parse_labels(value: object) -> list[int]:
        if isinstance(value, list):
            return [int(v) for v in value]
        if isinstance(value, str):
            cleaned = value.strip("[]")
            if not cleaned:
                return []
            return [int(v.strip()) for v in cleaned.split(",")]
        return [int(value)]

    enriched["labels"] = enriched["labels"].apply(parse_labels)
    return _enrich_with_source_labels(enriched)


def apply_target_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map GoEmotions source labels to the five target study emotion classes.

    Adds a ``target_emotion`` column. Rows without a mappable label are dropped.
    """
    mapped = df.copy()
    mapped["target_emotion"] = mapped["source_labels"].apply(
        lambda labels: map_source_labels_to_target(labels.split("|"))
    )
    mapped = mapped.dropna(subset=["target_emotion"]).reset_index(drop=True)
    return mapped


def clean_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform minimal cleaning without text preprocessing.

    - Drop rows with missing text
    - Remove exact duplicate text entries (keep first occurrence)
    - Strip leading/trailing whitespace from text only
    """
    cleaned = df.copy()
    cleaned = cleaned.dropna(subset=["text"])
    cleaned["text"] = cleaned["text"].astype(str).str.strip()
    cleaned = cleaned[cleaned["text"].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=["text"], keep="first")
    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def save_cleaned_dataset(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Persist the cleaned raw dataset (mapped labels, no tokenization)."""
    output_path = path or CLEANED_DATASET_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        col
        for col in [
            "text",
            "target_emotion",
            "source_label",
            "source_labels",
            "comment_id",
            "split",
        ]
        if col in df.columns
    ]
    df[columns].to_csv(output_path, index=False)
    return output_path


def get_dataset_metadata() -> dict[str, str]:
    """Return metadata about the selected dataset for reports."""
    return {
        "name": "GoEmotions (Simplified)",
        "kaggle_url": "https://www.kaggle.com/datasets/dehbmultani/go-emotions-dataset",
        "huggingface_id": HF_DATASET_NAME,
        "config": HF_DATASET_CONFIG,
        "description": (
            "58k Reddit comments annotated with 27 fine-grained emotions plus Neutral. "
            "Adapted to five student learning emotions via label mapping."
        ),
        "source_label_count": str(len(GOEMOTIONS_LABEL_NAMES)),
    }

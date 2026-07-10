"""
Dataset Collection & Exploration entry point.

Downloads the GoEmotions dataset (Kaggle / Hugging Face), maps labels to five
student learning emotions, performs exploratory analysis, and saves the cleaned
raw dataset.

Usage:
    py scripts/collect_explore_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import CLEANED_DATASET_PATH, RAW_DATASET_PATH
from utils.dataset_explorer import (
    check_duplicates,
    check_missing_values,
    display_head,
    display_info,
    generate_observations,
    show_label_distribution,
    show_sample_count,
)
from utils.dataset_loader import (
    apply_target_mapping,
    clean_raw_dataset,
    get_dataset_metadata,
    load_raw_dataset,
    save_cleaned_dataset,
)
from utils.emotion_mapping import print_mapping_summary
from utils.visualizations import create_all_visualizations


def main() -> None:
    """Run the full dataset collection and exploration pipeline."""
    # Ensure Unicode text (emojis in Reddit comments) prints on Windows consoles
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    metadata = get_dataset_metadata()

    print("=" * 70)
    print("AI Learning Assistant — Dataset Collection & Exploration")
    print("=" * 70)
    print(f"\nDataset  : {metadata['name']}")
    print(f"Source   : {metadata['huggingface_id']} ({metadata['config']})")
    print(f"Kaggle   : {metadata['kaggle_url']}")
    print(f"Details  : {metadata['description']}")

    print_mapping_summary()

    # --- Load dataset ---
    print("Loading dataset (downloads on first run)...")
    raw_df = load_raw_dataset()
    print(f"Raw dataset loaded: {len(raw_df):,} records -> {RAW_DATASET_PATH}")

    # --- Map to target emotions ---
    mapped_df = apply_target_mapping(raw_df)
    print(f"After label mapping: {len(mapped_df):,} records with target emotions")

    # --- Clean without text preprocessing ---
    cleaned_df = clean_raw_dataset(mapped_df)
    print(f"After deduplication : {len(cleaned_df):,} records")

    # --- Exploratory analysis ---
    display_head(cleaned_df)
    display_info(cleaned_df)
    duplicate_stats = check_duplicates(cleaned_df)
    check_missing_values(cleaned_df)
    show_sample_count(cleaned_df)
    distribution = show_label_distribution(cleaned_df)

    # --- Visualizations ---
    create_all_visualizations(cleaned_df)

    # --- Observations ---
    generate_observations(cleaned_df, distribution, duplicate_stats)

    # --- Save cleaned dataset ---
    saved_path = save_cleaned_dataset(cleaned_df)
    print(f"Cleaned dataset saved to: {saved_path}")
    print(f"  ({len(cleaned_df):,} rows, no text preprocessing applied)")


if __name__ == "__main__":
    main()

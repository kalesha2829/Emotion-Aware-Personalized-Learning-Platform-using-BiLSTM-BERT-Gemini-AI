"""
Data Preprocessing entry point for the BiLSTM pipeline.

Loads the cleaned dataset, applies text preprocessing, encodes labels,
performs stratified splitting, and saves all artifacts.

Usage:
    py scripts/preprocess_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.constants import LABEL_ENCODER_PATH
from utils.preprocess import (
    encode_labels,
    load_cleaned_dataset,
    preprocess_dataframe,
    print_preprocessing_summary,
    save_processed_datasets,
    stratified_train_val_test_split,
)


def main() -> None:
    """Run the full preprocessing pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("AI Learning Assistant — Data Preprocessing (BiLSTM Pipeline)")
    print("=" * 70)

    # 1. Load cleaned dataset
    print("\nLoading cleaned dataset...")
    df = load_cleaned_dataset()
    print(f"  Loaded {len(df):,} records")

    # 2. Apply text preprocessing (preserves original text column)
    print("\nApplying text preprocessing pipeline...")
    processed_df = preprocess_dataframe(df)
    empty_count = processed_df["processed_text"].str.len().eq(0).sum()
    print(f"  Added 'processed_text' column")
    if empty_count:
        print(f"  Warning: {empty_count:,} samples produced empty processed text")

    # 3. Encode emotion labels
    print("\nEncoding emotion labels...")
    encoded_df, encoder = encode_labels(processed_df)
    print(f"  Saved LabelEncoder -> {LABEL_ENCODER_PATH}")

    # 4. Stratified train / validation / test split
    print("\nSplitting dataset (80% / 10% / 10%, stratified)...")
    train_df, val_df, test_df = stratified_train_val_test_split(encoded_df)

    # 5. Save processed datasets
    print("\nSaving processed datasets...")
    paths = save_processed_datasets(encoded_df, train_df, val_df, test_df)
    for name, path in paths.items():
        print(f"  {name:<12} -> {path}")

    # 6. Print summary report
    print_preprocessing_summary(encoded_df, encoder, train_df, val_df, test_df)


if __name__ == "__main__":
    main()

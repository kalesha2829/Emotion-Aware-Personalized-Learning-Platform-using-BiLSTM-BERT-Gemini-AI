"""
Exploratory data analysis utilities for the student emotion dataset.
"""

from __future__ import annotations

import pandas as pd


def display_head(df: pd.DataFrame, n: int = 5) -> None:
    """Print the first n rows of the dataset."""
    print(f"\n--- First {n} Rows ---")
    preview_cols = [
        c for c in ["text", "target_emotion", "source_label", "source_labels", "split"]
        if c in df.columns
    ]
    print(df[preview_cols].head(n).to_string(index=False))


def display_info(df: pd.DataFrame) -> None:
    """Print dataset shape, column names, dtypes, and memory usage."""
    print("\n--- Dataset Information ---")
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns        : {list(df.columns)}")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nColumn dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"  {col:<20} {dtype}")


def check_duplicates(df: pd.DataFrame, subset: str = "text") -> dict[str, int]:
    """Check for duplicate records based on the text column."""
    total = len(df)
    unique = df[subset].nunique()
    duplicate_count = total - unique

    print("\n--- Duplicate Check ---")
    print(f"Total records       : {total:,}")
    print(f"Unique {subset} values : {unique:,}")
    print(f"Duplicate records   : {duplicate_count:,}")

    return {
        "total_records": total,
        "unique_texts": unique,
        "duplicate_records": duplicate_count,
    }


def check_missing_values(df: pd.DataFrame) -> pd.Series:
    """Report missing values per column."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    print("\n--- Missing Values ---")
    if missing.sum() == 0:
        print("No missing values found.")
    else:
        report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
        print(report[report["missing_count"] > 0].to_string())

    return missing


def show_sample_count(df: pd.DataFrame) -> int:
    """Print and return the total number of samples."""
    count = len(df)
    print(f"\n--- Total Samples ---\n{count:,}")
    return count


def show_label_distribution(
    df: pd.DataFrame, label_col: str = "target_emotion"
) -> pd.Series:
    """Print and return the distribution of emotion labels."""
    distribution = df[label_col].value_counts().sort_index()
    percentages = (df[label_col].value_counts(normalize=True) * 100).sort_index()

    print(f"\n--- Emotion Label Distribution ({label_col}) ---")
    for label in distribution.index:
        print(f"  {label:<12} {distribution[label]:>6,}  ({percentages[label]:.2f}%)")

    return distribution


def compute_class_imbalance_ratio(distribution: pd.Series) -> float:
    """Return the ratio of the largest class to the smallest class."""
    if distribution.empty:
        return 0.0
    return round(distribution.max() / distribution.min(), 2)


def generate_observations(
    df: pd.DataFrame,
    distribution: pd.Series,
    duplicate_stats: dict[str, int],
) -> None:
    """
    Print observations about dataset quality, class imbalance, and preprocessing needs.
    """
    imbalance_ratio = compute_class_imbalance_ratio(distribution)
    avg_text_len = df["text"].str.len().mean()
    max_text_len = df["text"].str.len().max()
    min_text_len = df["text"].str.len().min()

    print("\n" + "=" * 70)
    print("OBSERVATIONS & RECOMMENDATIONS")
    print("=" * 70)

    print("\n1. Dataset Quality")
    print("-" * 40)
    if duplicate_stats["duplicate_records"] == 0:
        print("  - No duplicate text entries remain after cleaning.")
    else:
        print(
            f"  - {duplicate_stats['duplicate_records']:,} duplicate texts were removed "
            "during cleaning."
        )
    print(f"  - Average text length: {avg_text_len:.1f} characters")
    print(f"  - Text length range : {min_text_len} - {max_text_len} characters")
    print(
        "  - Source data comes from Reddit comments; language is informal and may "
        "contain masked tokens (e.g. [NAME], [RELIGION])."
    )
    if "split" in df.columns:
        split_counts = df["split"].value_counts().to_dict()
        print(f"  - Split distribution: {split_counts}")

    print("\n2. Class Imbalance")
    print("-" * 40)
    majority = distribution.idxmax()
    minority = distribution.idxmin()
    print(f"  - Majority class : {majority} ({distribution[majority]:,} samples)")
    print(f"  - Minority class : {minority} ({distribution[minority]:,} samples)")
    print(f"  - Imbalance ratio (max/min): {imbalance_ratio}:1")
    if imbalance_ratio > 3:
        print(
            "  - Significant class imbalance detected. Consider class weighting, "
            "oversampling (e.g. SMOTE), or stratified splitting during model training."
        )
    else:
        print("  - Class distribution is relatively balanced for initial experiments.")

    print("\n3. Required Preprocessing (for future modules - NOT applied here)")
    print("-" * 40)
    print("  - Lowercasing and normalization of masked tokens ([NAME] -> placeholder)")
    print("  - Removal of URLs, excessive whitespace, and special characters")
    print("  - Tokenization and sequence padding for BiLSTM input")
    print("  - Subword tokenization via BERT tokenizer (max_length truncation)")
    print("  - Stratified train/validation/test split preserving label proportions")
    print("  - Address class imbalance via weighted loss or resampling")
    print("=" * 70 + "\n")

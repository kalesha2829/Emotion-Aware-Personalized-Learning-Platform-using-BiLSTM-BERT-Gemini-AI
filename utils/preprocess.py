"""
Text preprocessing utilities for the BiLSTM emotion classification pipeline.

Applies a deterministic cleaning pipeline to raw text while preserving the
original ``text`` column. Produces ``processed_text`` suitable for downstream
tokenization and sequence modeling (tokenization is handled in a later module).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import joblib
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from utils.constants import (
    CLEANED_DATASET_PATH,
    LABEL_ENCODER_PATH,
    PROCESSED_DATASET_PATH,
    RANDOM_STATE,
    TEST_DATASET_PATH,
    TEST_RATIO,
    TRAIN_DATASET_PATH,
    TRAIN_RATIO,
    VAL_DATASET_PATH,
    VAL_RATIO,
)

# ---------------------------------------------------------------------------
# NLTK resource management
# ---------------------------------------------------------------------------


def ensure_nltk_resources() -> None:
    """Download required NLTK corpora if they are not already present."""
    resource_paths = {
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource_name, lookup_path in resource_paths.items():
        try:
            nltk.data.find(lookup_path)
        except LookupError:
            nltk.download(resource_name, quiet=True)


def get_stop_words(language: str = "english") -> set[str]:
    """Return the NLTK stopword set for the given language."""
    ensure_nltk_resources()
    return set(stopwords.words(language))


def get_lemmatizer() -> WordNetLemmatizer:
    """Return an initialized WordNet lemmatizer."""
    ensure_nltk_resources()
    return WordNetLemmatizer()


# ---------------------------------------------------------------------------
# Common English contractions
# ---------------------------------------------------------------------------

CONTRACTIONS: dict[str, str] = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "could've": "could have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "mightn't": "might not",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "should've": "should have",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "won't": "will not",
    "wouldn't": "would not",
    "would've": "would have",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# ---------------------------------------------------------------------------
# Individual preprocessing steps
# ---------------------------------------------------------------------------

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<[^>]+>")
EMAIL_PATTERN = re.compile(r"\S+@\S+")
NUMBER_PATTERN = re.compile(r"\b\d+\b")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s]", re.UNICODE)
SPECIAL_CHAR_PATTERN = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


CONTRACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in CONTRACTIONS) + r")\b"
)


def expand_contractions(text: str) -> str:
    """Expand common English contractions to their full forms."""
    return CONTRACTION_PATTERN.sub(lambda match: CONTRACTIONS[match.group(1)], text)


def remove_urls(text: str) -> str:
    """Remove HTTP/HTTPS and www URLs from text."""
    return URL_PATTERN.sub(" ", text)


def remove_html_tags(text: str) -> str:
    """Remove HTML/XML tags from text."""
    return HTML_PATTERN.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return EMAIL_PATTERN.sub(" ", text)


def remove_numbers(text: str) -> str:
    """Remove standalone numeric tokens from text."""
    return NUMBER_PATTERN.sub(" ", text)


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters from text."""
    return PUNCTUATION_PATTERN.sub(" ", text)


def remove_special_characters(text: str) -> str:
    """Remove non-alphabetic characters (retain letters and whitespace)."""
    return SPECIAL_CHAR_PATTERN.sub(" ", text)


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace and strip leading/trailing spaces."""
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_stopwords_from_tokens(
    tokens: Iterable[str], stop_word_set: set[str]
) -> list[str]:
    """Filter out NLTK stopwords from a token list."""
    return [token for token in tokens if token not in stop_word_set]


def lemmatize_tokens(
    tokens: Iterable[str], lemmatizer: WordNetLemmatizer | None = None
) -> list[str]:
    """Lemmatize tokens using the NLTK WordNet lemmatizer."""
    lemmatizer = lemmatizer or get_lemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


def preprocess_text(
    text: str,
    stop_word_set: set[str] | None = None,
    lemmatizer: WordNetLemmatizer | None = None,
) -> str:
    """
    Apply the full BiLSTM preprocessing pipeline to a single text string.

    The original text column is never modified; this function returns a new
    cleaned string for the ``processed_text`` column.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    stop_word_set = stop_word_set or get_stop_words()
    lemmatizer = lemmatizer or get_lemmatizer()

    # Order: lowercase -> contractions -> structural noise -> numeric/punct cleanup
    cleaned = to_lowercase(text)
    cleaned = expand_contractions(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_emails(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_punctuation(cleaned)
    cleaned = remove_special_characters(cleaned)
    cleaned = normalize_whitespace(cleaned)

    tokens = cleaned.split()
    tokens = remove_stopwords_from_tokens(tokens, stop_word_set)
    tokens = lemmatize_tokens(tokens, lemmatizer)

    return normalize_whitespace(" ".join(tokens))


def preprocess_dataframe(
    df: pd.DataFrame,
    text_col: str = "text",
    processed_col: str = "processed_text",
) -> pd.DataFrame:
    """
    Add a ``processed_text`` column without altering the original text column.

    Reuses a single stopword set and lemmatizer instance for efficiency.
    """
    result = df.copy()
    stop_word_set = get_stop_words()
    lemmatizer = get_lemmatizer()

    result[processed_col] = result[text_col].apply(
        lambda text: preprocess_text(text, stop_word_set, lemmatizer)
    )
    return result


# ---------------------------------------------------------------------------
# Dataset loading, encoding, splitting, and persistence
# ---------------------------------------------------------------------------


def load_cleaned_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the cleaned student emotions CSV produced by the EDA module."""
    dataset_path = path or CLEANED_DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {dataset_path}. "
            "Run scripts/collect_explore_dataset.py first."
        )
    return pd.read_csv(dataset_path)


def encode_labels(
    df: pd.DataFrame,
    label_col: str = "target_emotion",
    encoded_col: str = "label",
    encoder_path: Path | None = None,
) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode string emotion labels with sklearn LabelEncoder.

    Saves the fitted encoder to ``models/label_encoder.joblib`` for inference.
    """
    output_path = encoder_path or LABEL_ENCODER_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = LabelEncoder()
    encoded_df = df.copy()
    encoded_df[encoded_col] = encoder.fit_transform(encoded_df[label_col])

    joblib.dump(encoder, output_path)
    return encoded_df, encoder


def stratified_train_val_test_split(
    df: pd.DataFrame,
    label_col: str = "label",
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train / validation / test sets with stratified sampling.

    Default ratios: 80% train, 10% validation, 10% test.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    # First split: train vs (val + test)
    holdout_ratio = val_ratio + test_ratio
    train_df, holdout_df = train_test_split(
        df,
        test_size=holdout_ratio,
        stratify=df[label_col],
        random_state=random_state,
    )

    # Second split: val vs test (equal proportions of holdout)
    relative_test_size = test_ratio / holdout_ratio
    val_df, test_df = train_test_split(
        holdout_df,
        test_size=relative_test_size,
        stratify=holdout_df[label_col],
        random_state=random_state,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df


def save_processed_datasets(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    full_path: Path | None = None,
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
) -> dict[str, Path]:
    """Persist the full processed dataset and stratified splits to CSV."""
    paths = {
        "full": full_path or PROCESSED_DATASET_PATH,
        "train": train_path or TRAIN_DATASET_PATH,
        "validation": val_path or VAL_DATASET_PATH,
        "test": test_path or TEST_DATASET_PATH,
    }

    for key, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)

    full_df.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["validation"], index=False)
    test_df.to_csv(paths["test"], index=False)

    return paths


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def compute_vocabulary_stats(processed_texts: pd.Series) -> dict[str, float | int]:
    """Compute vocabulary statistics from processed text samples."""
    token_lists = processed_texts.fillna("").str.split()
    total_tokens = int(token_lists.str.len().sum())
    sample_count = len(processed_texts)

    all_tokens: list[str] = []
    for tokens in token_lists:
        all_tokens.extend(tokens)

    unique_tokens = set(all_tokens)
    non_empty = processed_texts.fillna("").str.strip().ne("").sum()

    return {
        "sample_count": sample_count,
        "non_empty_samples": int(non_empty),
        "empty_samples": int(sample_count - non_empty),
        "total_tokens": total_tokens,
        "unique_tokens": len(unique_tokens),
        "avg_tokens_per_sample": round(total_tokens / sample_count, 2) if sample_count else 0.0,
        "avg_tokens_per_non_empty_sample": round(total_tokens / non_empty, 2) if non_empty else 0.0,
        "min_tokens": int(token_lists.str.len().min()) if sample_count else 0,
        "max_tokens": int(token_lists.str.len().max()) if sample_count else 0,
    }


def print_preprocessing_summary(
    df: pd.DataFrame,
    encoder: LabelEncoder,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_index: int = 0,
) -> None:
    """Print sample texts, vocabulary statistics, class info, and split sizes."""
    stats = compute_vocabulary_stats(df["processed_text"])

    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print("\n--- Sample Original vs Processed Text ---")
    sample = df.iloc[sample_index]
    print(f"Original  : {sample['text']}")
    print(f"Processed : {sample['processed_text']}")
    print(f"Emotion   : {sample['target_emotion']} (label={sample['label']})")

    print("\n--- Vocabulary Statistics ---")
    for key, value in stats.items():
        print(f"  {key:<32} {value:,}" if isinstance(value, int) else f"  {key:<32} {value}")

    print("\n--- Label Encoding ---")
    print(f"  Number of classes : {len(encoder.classes_)}")
    for idx, class_name in enumerate(encoder.classes_):
        print(f"    {idx} -> {class_name}")

    print("\n--- Dataset Sizes (Stratified Splits) ---")
    total = len(train_df) + len(val_df) + len(test_df)
    print(f"  Training   : {len(train_df):>6,}  ({len(train_df) / total * 100:.1f}%)")
    print(f"  Validation : {len(val_df):>6,}  ({len(val_df) / total * 100:.1f}%)")
    print(f"  Test       : {len(test_df):>6,}  ({len(test_df) / total * 100:.1f}%)")
    print(f"  Total      : {total:>6,}")

    print("\n--- Class Distribution per Split ---")
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        counts = split_df["target_emotion"].value_counts().sort_index()
        print(f"  {split_name}:")
        for emotion, count in counts.items():
            pct = count / len(split_df) * 100
            print(f"    {emotion:<12} {count:>5,}  ({pct:.1f}%)")

    print("=" * 70 + "\n")

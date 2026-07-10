"""
BiLSTM emotion classification model — training, evaluation, and artifact helpers.

Provides reusable functions for building, training, saving, and loading the BiLSTM
model used in the AI Learning Assistant. Does not include BERT, Gemini, or Streamlit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

from utils.constants import (
    ASSETS_DIR,
    BATCH_SIZE,
    BILSTM_CONFUSION_MATRIX_PATH,
    BILSTM_HISTORY_PATH,
    BILSTM_MAX_LEN_PATH,
    BILSTM_MODEL_PATH,
    BILSTM_TOKENIZER_PATH,
    BILSTM_TRAINING_CURVES_PATH,
    DENSE_UNITS,
    DROPOUT_RATE,
    EMBEDDING_DIM,
    EMPTY_TEXT_PLACEHOLDER,
    EPOCHS,
    LABEL_ENCODER_PATH,
    LEARNING_RATE,
    LSTM_UNITS,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    NUM_CLASSES,
    TEST_DATASET_PATH,
    TRAIN_DATASET_PATH,
    VAL_DATASET_PATH,
)

# ---------------------------------------------------------------------------
# Dataset loading and empty-text handling
# ---------------------------------------------------------------------------


def load_split_datasets(
    train_path: Path | None = None,
    val_path: Path | None = None,
    test_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load preprocessed train, validation, and test CSV splits."""
    train_df = pd.read_csv(train_path or TRAIN_DATASET_PATH)
    val_df = pd.read_csv(val_path or VAL_DATASET_PATH)
    test_df = pd.read_csv(test_path or TEST_DATASET_PATH)
    return train_df, val_df, test_df


def handle_empty_processed_text(
    df: pd.DataFrame,
    text_col: str = "processed_text",
    placeholder: str = EMPTY_TEXT_PLACEHOLDER,
) -> tuple[pd.DataFrame, int]:
    """
    Replace empty or whitespace-only processed_text values with a placeholder token.

    Approach
    --------
    We **replace** rather than remove empty records because:

    1. Removing rows would desynchronize splits from the saved CSV files and discard
       valid labels that may still be useful for learning edge-case behaviour.
    2. A dedicated ``<EMPTY>`` token lets the Tokenizer assign a fixed index so the
       model receives a consistent input instead of a zero-length sequence.
    3. Empty texts typically arise when stopword removal eliminates every token;
       the placeholder preserves sample count and enables explicit handling at
       inference time.

    Returns the updated DataFrame and the number of replaced records.
    """
    result = df.copy()
    empty_mask = result[text_col].fillna("").astype(str).str.strip().eq("")
    replaced_count = int(empty_mask.sum())

    if replaced_count:
        result.loc[empty_mask, text_col] = placeholder

    return result, replaced_count


# ---------------------------------------------------------------------------
# Tokenization and sequence preparation
# ---------------------------------------------------------------------------


def create_and_fit_tokenizer(
    texts: list[str] | pd.Series,
    num_words: int | None = None,
    oov_token: str = "<OOV>",
) -> Tokenizer:
    """
    Build and fit a Keras Tokenizer on training text only.

    ``num_words=None`` retains the full vocabulary; a cap can be added later if needed.
    """
    tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_sequences(
    texts: list[str] | pd.Series, tokenizer: Tokenizer
) -> list[list[int]]:
    """Convert raw text strings into integer token sequences."""
    return tokenizer.texts_to_sequences(texts)


def pad_token_sequences(
    sequences: list[list[int]],
    max_len: int | None = None,
    padding: str = "post",
    truncating: str = "post",
) -> np.ndarray:
    """Pad integer sequences to a uniform length for BiLSTM input."""
    length = max_len or MAX_SEQUENCE_LENGTH
    return pad_sequences(
        sequences, maxlen=length, padding=padding, truncating=truncating
    )


def prepare_features(
    texts: list[str] | pd.Series,
    tokenizer: Tokenizer,
    max_len: int,
) -> np.ndarray:
    """Convert text to padded integer sequences ready for model input."""
    sequences = texts_to_sequences(texts, tokenizer)
    return pad_token_sequences(sequences, max_len=max_len)


def infer_max_sequence_length(
    sequences: list[list[int]], percentile: float = 99.0
) -> int:
    """Suggest max sequence length from the given percentile of sequence lengths."""
    lengths = [len(seq) for seq in sequences if seq]
    if not lengths:
        return MAX_SEQUENCE_LENGTH
    return int(np.ceil(np.percentile(lengths, percentile)))


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


def build_bilstm_model(
    vocab_size: int,
    max_len: int,
    num_classes: int = NUM_CLASSES,
    embedding_dim: int = EMBEDDING_DIM,
    lstm_units: int = LSTM_UNITS,
    dense_units: int = DENSE_UNITS,
    dropout_rate: float = DROPOUT_RATE,
    learning_rate: float = LEARNING_RATE,
) -> Sequential:
    """
    Build a BiLSTM classifier for emotion detection.

    Architecture
    ------------
    Embedding -> Bidirectional LSTM -> Dropout -> Dense (ReLU) -> Dense (Softmax)
    """
    model = Sequential(
        [
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_len,
                name="embedding",
            ),
            Bidirectional(LSTM(lstm_units), name="bilstm"),
            Dropout(dropout_rate, name="dropout"),
            Dense(dense_units, activation="relu", name="dense_hidden"),
            Dense(num_classes, activation="softmax", name="output"),
        ],
        name="bilstm_emotion_classifier",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Training, evaluation, and visualization
# ---------------------------------------------------------------------------


def get_training_callbacks(
    model_path: Path | None = None,
    monitor: str = "val_loss",
    patience: int = 3,
) -> list[Any]:
    """Return EarlyStopping and ModelCheckpoint callbacks."""
    path = model_path or BILSTM_MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )
    checkpoint = ModelCheckpoint(
        filepath=str(path),
        monitor=monitor,
        save_best_only=True,
        verbose=1,
    )
    return [early_stopping, checkpoint]


def train_bilstm_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    callbacks: list[Any] | None = None,
) -> Any:
    """Train the BiLSTM model using the validation set for monitoring."""
    callbacks = callbacks or get_training_callbacks()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def evaluate_bilstm_model(
    model: Sequential,
    x_test: np.ndarray,
    y_test: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, Any]:
    """
    Evaluate the model on the test set and return metrics, confusion matrix,
    and classification report.
    """
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=list(label_encoder.classes_),
        digits=4,
        zero_division=0,
    )

    return {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_test,
        "y_pred": y_pred,
    }


def plot_training_history(
    history: Any,
    save_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot training/validation accuracy and loss curves."""
    output_path = save_path or BILSTM_TRAINING_CURVES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Accuracy vs Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Loss vs Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot and save a confusion matrix heatmap."""
    output_path = save_path or BILSTM_CONFUSION_MATRIX_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BiLSTM Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def print_training_report(
    history: Any,
    eval_results: dict[str, Any],
    label_encoder: LabelEncoder,
) -> None:
    """Print final training, validation, and test metrics."""
    best_epoch = int(np.argmin(history.history["val_loss"]))
    train_acc = history.history["accuracy"][best_epoch]
    val_acc = history.history["val_accuracy"][best_epoch]
    train_loss = history.history["loss"][best_epoch]
    val_loss = history.history["val_loss"][best_epoch]

    print("\n" + "=" * 70)
    print("BiLSTM TRAINING & EVALUATION REPORT")
    print("=" * 70)
    print(f"\n  Best Epoch (val_loss) : {best_epoch + 1}")
    print(f"  Training Accuracy     : {train_acc:.4f}")
    print(f"  Validation Accuracy   : {val_acc:.4f}")
    print(f"  Test Accuracy         : {eval_results['test_accuracy']:.4f}")
    print(f"\n  Training Loss         : {train_loss:.4f}")
    print(f"  Validation Loss       : {val_loss:.4f}")
    print(f"  Test Loss             : {eval_results['test_loss']:.4f}")
    print("\n--- Confusion Matrix (Test Set) ---")
    print(eval_results["confusion_matrix"])
    print("\n--- Classification Report (Test Set) ---")
    print(eval_results["classification_report"])
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Artifact persistence
# ---------------------------------------------------------------------------


def save_training_history(history: Any, path: Path | None = None) -> Path:
    """Serialize Keras training history to JSON."""
    output_path = path or BILSTM_HISTORY_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {key: [float(v) for v in values] for key, values in history.history.items()}
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(serializable, file, indent=2)
    return output_path


def save_tokenizer(tokenizer: Tokenizer, path: Path | None = None) -> Path:
    """Persist the fitted Keras Tokenizer."""
    output_path = path or BILSTM_TOKENIZER_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(tokenizer, output_path)
    return output_path


def save_max_len(max_len: int, path: Path | None = None) -> Path:
    """Persist the sequence padding length used during training."""
    output_path = path or BILSTM_MAX_LEN_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(max_len, output_path)
    return output_path


def load_label_encoder(path: Path | None = None) -> LabelEncoder:
    """Load the fitted LabelEncoder saved during preprocessing."""
    encoder_path = path or LABEL_ENCODER_PATH
    return joblib.load(encoder_path)


def load_bilstm_tokenizer(path: Path | None = None) -> Tokenizer:
    """Load the fitted Keras Tokenizer."""
    return joblib.load(path or BILSTM_TOKENIZER_PATH)


def load_bilstm_max_len(path: Path | None = None) -> int:
    """Load the max sequence length used during training."""
    return joblib.load(path or BILSTM_MAX_LEN_PATH)


def load_bilstm_model(path: Path | None = None) -> Sequential:
    """Load the trained BiLSTM Keras model."""
    return load_model(path or BILSTM_MODEL_PATH)


def load_bilstm_artifacts() -> dict[str, Any]:
    """
    Load all BiLSTM artifacts needed for later inference integration.

    Returns a dict with keys: model, tokenizer, label_encoder, max_len.
    """
    return {
        "model": load_bilstm_model(),
        "tokenizer": load_bilstm_tokenizer(),
        "label_encoder": load_label_encoder(),
        "max_len": load_bilstm_max_len(),
    }


def predict_emotion_labels(
    processed_texts: list[str] | pd.Series,
    artifacts: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict emotion labels from preprocessed text strings.

    Expects text that has already passed through the preprocessing pipeline
    (``processed_text`` format). Returns predicted integer labels and probabilities.

    This is a lightweight helper for later integration — not a full prediction pipeline.
    """
    artifacts = artifacts or load_bilstm_artifacts()
    model = artifacts["model"]
    tokenizer = artifacts["tokenizer"]
    max_len = artifacts["max_len"]

    texts = pd.Series(processed_texts).fillna("").astype(str)
    empty_mask = texts.str.strip().eq("")
    if empty_mask.any():
        texts = texts.copy()
        texts.loc[empty_mask] = EMPTY_TEXT_PLACEHOLDER

    x = prepare_features(texts, tokenizer, max_len)
    probabilities = model.predict(x, verbose=0)
    predicted_labels = np.argmax(probabilities, axis=1)
    return predicted_labels, probabilities


def decode_emotion_labels(
    label_ids: np.ndarray,
    label_encoder: LabelEncoder | None = None,
) -> list[str]:
    """Convert numeric label IDs to human-readable emotion names."""
    encoder = label_encoder or load_label_encoder()
    return list(encoder.inverse_transform(label_ids))

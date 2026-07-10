"""
BERT emotion classification — fine-tuning, evaluation, and inference helpers.

Uses Hugging Face ``bert-base-uncased`` on the original ``text`` column with the
project's encoded labels. Does not include Streamlit, Gemini, or pipeline logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from transformers import (
    BertForSequenceClassification,
    BertTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    logging as transformers_logging,
)
from huggingface_hub.utils import disable_progress_bars as hf_disable_progress_bars, enable_progress_bars as hf_enable_progress_bars

from utils.bilstm_model import load_split_datasets
from utils.constants import (
    BERT_BATCH_SIZE,
    BERT_CHECKPOINT_DIR,
    BERT_CONFUSION_MATRIX_PATH,
    BERT_EARLY_STOPPING_PATIENCE,
    BERT_EMPTY_TEXT_PLACEHOLDER,
    BERT_EPOCHS,
    BERT_EVAL_BATCH_SIZE,
    BERT_HISTORY_PATH,
    BERT_LABEL_MAPPING_PATH,
    BERT_LEARNING_RATE,
    BERT_MAX_LENGTH,
    BERT_MODEL_DIR,
    BERT_MODEL_NAME,
    BERT_TRAINING_CURVES_PATH,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    LABEL_ENCODER_PATH,
    NUM_CLASSES,
    TARGET_EMOTIONS,
)

# ---------------------------------------------------------------------------
# Dataset utilities
# ---------------------------------------------------------------------------


class EmotionTextDataset(Dataset):
    """PyTorch Dataset wrapping tokenized inputs and integer labels."""

    def __init__(self, encodings: dict[str, list], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def handle_empty_original_text(
    df: pd.DataFrame,
    text_col: str = "text",
    placeholder: str = BERT_EMPTY_TEXT_PLACEHOLDER,
) -> tuple[pd.DataFrame, int]:
    """Replace empty original text values with a placeholder for BERT tokenization."""
    result = df.copy()
    empty_mask = result[text_col].fillna("").astype(str).str.strip().eq("")
    replaced_count = int(empty_mask.sum())
    if replaced_count:
        result.loc[empty_mask, text_col] = placeholder
    return result, replaced_count


def build_label_mapping(label_encoder: LabelEncoder | None = None) -> dict[str, Any]:
    """
    Build id2label / label2id mappings aligned with the preprocessing LabelEncoder.

    Keys in ``id2label`` are strings (Hugging Face JSON convention) so the mapping
    works both in memory and after ``json.dump`` / ``json.load``.

    Falls back to ``TARGET_EMOTIONS`` order when no encoder is supplied.
    """
    if label_encoder is not None:
        classes = [str(name) for name in label_encoder.classes_]
    else:
        classes = list(TARGET_EMOTIONS)

    id2label = {str(i): name for i, name in enumerate(classes)}
    label2id = {name: i for i, name in enumerate(classes)}
    return {"id2label": id2label, "label2id": label2id}


def get_class_names_from_mapping(label_mapping: dict[str, Any]) -> list[str]:
    """Return ordered class names from a label mapping dict."""
    id2label = label_mapping["id2label"]
    return [id2label[str(i)] for i in sorted(int(k) for k in id2label)]


def save_label_mapping(
    mapping: dict[str, Any],
    path: Path | None = None,
) -> Path:
    """Persist label mapping JSON for inference."""
    output_path = path or BERT_LABEL_MAPPING_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2)
    return output_path


def load_label_mapping(path: Path | None = None) -> dict[str, Any]:
    """Load saved label mapping JSON or fall back to available label encoder data."""
    mapping_path = path or BERT_LABEL_MAPPING_PATH
    try:
        with open(mapping_path, encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        try:
            encoder = load_label_encoder()
            return build_label_mapping(encoder)
        except Exception:
            return build_label_mapping()


def load_label_encoder(path: Path | None = None) -> LabelEncoder:
    """Load the LabelEncoder saved during preprocessing."""
    import joblib

    return joblib.load(path or LABEL_ENCODER_PATH)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def load_bert_tokenizer(model_name: str = BERT_MODEL_NAME) -> BertTokenizer:
    """Load the pretrained BERT tokenizer."""
    transformers_logging.disable_progress_bar()
    hf_disable_progress_bars()
    try:
        return BertTokenizer.from_pretrained(model_name)
    finally:
        transformers_logging.enable_progress_bar()
        hf_enable_progress_bars()


def tokenize_texts(
    texts: list[str] | pd.Series,
    tokenizer: BertTokenizer,
    max_length: int = BERT_MAX_LENGTH,
) -> dict[str, list]:
    """Tokenize raw text with BertTokenizer (truncation + padding)."""
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors=None,
    )


def create_emotion_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: BertTokenizer,
    text_col: str = "text",
    label_col: str = "label",
    max_length: int = BERT_MAX_LENGTH,
) -> tuple[EmotionTextDataset, EmotionTextDataset, EmotionTextDataset]:
    """Build train, validation, and test PyTorch datasets from DataFrames."""
    train_enc = tokenize_texts(train_df[text_col], tokenizer, max_length)
    val_enc = tokenize_texts(val_df[text_col], tokenizer, max_length)
    test_enc = tokenize_texts(test_df[text_col], tokenizer, max_length)

    train_dataset = EmotionTextDataset(train_enc, train_df[label_col].tolist())
    val_dataset = EmotionTextDataset(val_enc, val_df[label_col].tolist())
    test_dataset = EmotionTextDataset(test_enc, test_df[label_col].tolist())
    return train_dataset, val_dataset, test_dataset


# ---------------------------------------------------------------------------
# Model creation and metrics
# ---------------------------------------------------------------------------


def create_bert_classifier(
    num_labels: int = NUM_CLASSES,
    model_name: str = BERT_MODEL_NAME,
    label_mapping: dict[str, Any] | None = None,
) -> BertForSequenceClassification:
    """Instantiate a BERT sequence classification head with ``num_labels`` outputs."""
    kwargs: dict[str, Any] = {"num_labels": num_labels}
    if label_mapping:
        kwargs["id2label"] = {str(k): v for k, v in label_mapping["id2label"].items()}
        kwargs["label2id"] = label_mapping["label2id"]

    transformers_logging.disable_progress_bar()
    hf_disable_progress_bars()
    try:
        return BertForSequenceClassification.from_pretrained(model_name, **kwargs)
    finally:
        transformers_logging.enable_progress_bar()
        hf_enable_progress_bars()


def get_device() -> torch.device:
    """Return CUDA device when available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for the Hugging Face Trainer."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(
            precision_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, average="weighted", zero_division=0)
        ),
        "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def build_training_arguments(
    output_dir: Path | None = None,
    num_epochs: int = BERT_EPOCHS,
    batch_size: int = BERT_BATCH_SIZE,
    eval_batch_size: int = BERT_EVAL_BATCH_SIZE,
    learning_rate: float = BERT_LEARNING_RATE,
    weight_decay: float = BERT_WEIGHT_DECAY,
    warmup_ratio: float = BERT_WARMUP_RATIO,
) -> TrainingArguments:
    """Configure TrainingArguments with AdamW (default) and LR scheduler."""
    device = get_device()
    use_cuda = device.type == "cuda"
    return TrainingArguments(
        output_dir=str(output_dir or BERT_CHECKPOINT_DIR),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        fp16=use_cuda,
        dataloader_num_workers=0,
        seed=42,
    )


def train_bert_model(
    model: BertForSequenceClassification,
    train_dataset: EmotionTextDataset,
    val_dataset: EmotionTextDataset,
    training_args: TrainingArguments | None = None,
    early_stopping_patience: int = BERT_EARLY_STOPPING_PATIENCE,
) -> tuple[Trainer, Any]:
    """
    Fine-tune BERT using the Hugging Face Trainer with validation and early stopping.
    """
    args = training_args or build_training_arguments()
    callbacks = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_result = trainer.train()
    return trainer, train_result


def evaluate_bert_on_test(
    trainer: Trainer,
    test_dataset: EmotionTextDataset,
    label_mapping: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate fine-tuned BERT on the held-out test set."""
    predictions_output = trainer.predict(test_dataset)
    logits = predictions_output.predictions
    y_true = predictions_output.label_ids
    y_pred = np.argmax(logits, axis=-1)

    class_names = get_class_names_from_mapping(label_mapping)

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_precision": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_recall": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "test_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "confusion_matrix": cm,
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def extract_epoch_metrics(trainer: Trainer) -> dict[str, list[float]]:
    """Extract per-epoch train/eval metrics from Trainer log history."""
    epoch_metrics: dict[float, dict[str, float]] = {}

    for entry in trainer.state.log_history:
        if "epoch" not in entry:
            continue
        epoch = float(entry["epoch"])
        if epoch not in epoch_metrics:
            epoch_metrics[epoch] = {}

        if "loss" in entry and "eval_loss" not in entry:
            epoch_metrics[epoch]["train_loss"] = float(entry["loss"])
        if "eval_loss" in entry:
            epoch_metrics[epoch]["eval_loss"] = float(entry["eval_loss"])
            for key in ("eval_accuracy", "eval_precision", "eval_recall", "eval_f1"):
                if key in entry:
                    epoch_metrics[epoch][key] = float(entry[key])

    sorted_epochs = sorted(epoch_metrics.keys())
    history: dict[str, list[float]] = {
        "epoch": sorted_epochs,
        "train_loss": [],
        "eval_loss": [],
        "eval_accuracy": [],
        "eval_precision": [],
        "eval_recall": [],
        "eval_f1": [],
    }

    for epoch in sorted_epochs:
        metrics = epoch_metrics[epoch]
        history["train_loss"].append(metrics.get("train_loss", float("nan")))
        history["eval_loss"].append(metrics.get("eval_loss", float("nan")))
        history["eval_accuracy"].append(metrics.get("eval_accuracy", float("nan")))
        history["eval_precision"].append(metrics.get("eval_precision", float("nan")))
        history["eval_recall"].append(metrics.get("eval_recall", float("nan")))
        history["eval_f1"].append(metrics.get("eval_f1", float("nan")))

    return history


def save_training_history(history: dict[str, Any], path: Path | None = None) -> Path:
    """Serialize training history to JSON."""
    output_path = path or BERT_HISTORY_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)
    return output_path


def compute_train_accuracy(trainer: Trainer, train_dataset: EmotionTextDataset) -> float:
    """Compute training accuracy after fine-tuning."""
    predictions = trainer.predict(train_dataset)
    y_pred = np.argmax(predictions.predictions, axis=-1)
    return float(accuracy_score(predictions.label_ids, y_pred))


def plot_bert_training_history(
    history: dict[str, Any],
    save_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot training/validation loss and validation accuracy per epoch."""
    output_path = save_path or BERT_TRAINING_CURVES_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = history.get("epoch", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.get("train_loss", []), marker="o", label="Train Loss")
    axes[0].plot(epochs, history.get("eval_loss", []), marker="o", label="Validation Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history.get("eval_accuracy", []), marker="o", color="green", label="Validation Accuracy")
    axes[1].set_title("Validation Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def plot_bert_confusion_matrix(
    confusion_mat: np.ndarray,
    class_names: list[str],
    save_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Plot and save BERT test-set confusion matrix."""
    output_path = save_path or BERT_CONFUSION_MATRIX_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("BERT Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def print_bert_report(
    trainer: Trainer,
    train_dataset: EmotionTextDataset,
    eval_results: dict[str, Any],
    history: dict[str, Any],
) -> None:
    """Print training, validation, and test metrics."""
    train_acc = compute_train_accuracy(trainer, train_dataset)
    eval_losses = history.get("eval_loss", [])
    eval_accs = history.get("eval_accuracy", [])
    train_losses = history.get("train_loss", [])

    best_idx = int(np.nanargmin(eval_losses)) if eval_losses else 0

    print("\n" + "=" * 70)
    print("BERT TRAINING & EVALUATION REPORT")
    print("=" * 70)
    print(f"\n  Device               : {get_device()}")
    print(f"  Best Epoch (val_loss): {history.get('epoch', [0])[best_idx]}")
    print(f"\n  Training Accuracy    : {train_acc:.4f}")
    print(f"  Validation Accuracy  : {eval_accs[best_idx]:.4f}")
    print(f"  Test Accuracy        : {eval_results['test_accuracy']:.4f}")
    print(f"\n  Training Loss        : {train_losses[best_idx]:.4f}")
    print(f"  Validation Loss      : {eval_losses[best_idx]:.4f}")
    print(f"  Test Loss            : {eval_results.get('test_loss', float('nan')):.4f}")
    print(f"\n  Test Precision       : {eval_results['test_precision']:.4f}")
    print(f"  Test Recall          : {eval_results['test_recall']:.4f}")
    print(f"  Test F1 Score        : {eval_results['test_f1']:.4f}")
    print("\n--- Confusion Matrix (Test Set) ---")
    print(eval_results["confusion_matrix"])
    print("\n--- Classification Report (Test Set) ---")
    print(eval_results["classification_report"])
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Artifact persistence and inference helpers
# ---------------------------------------------------------------------------


def save_bert_model(
    model: BertForSequenceClassification,
    tokenizer: BertTokenizer,
    model_dir: Path | None = None,
) -> Path:
    """Save fine-tuned BERT model and tokenizer to disk."""
    output_dir = model_dir or BERT_MODEL_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return output_dir


def _resolve_bert_model_directory(model_dir: Path | None = None) -> Path:
    if model_dir is not None and model_dir.exists():
        return model_dir

    if BERT_MODEL_DIR.exists():
        return BERT_MODEL_DIR

    if BERT_CHECKPOINT_DIR.exists():
        checkpoint_dirs = [
            child
            for child in BERT_CHECKPOINT_DIR.iterdir()
            if child.is_dir() and child.name.startswith("checkpoint-")
        ]
        if checkpoint_dirs:
            latest_checkpoint = sorted(
                checkpoint_dirs,
                key=lambda directory: int(directory.name.split("-")[-1]),
            )[-1]
            # Auto-export latest checkpoint to deployment folder
            _export_checkpoint_to_deployment(latest_checkpoint)
            return BERT_MODEL_DIR

    raise FileNotFoundError(
        f"No BERT model directory found. Expected {BERT_MODEL_DIR} or a checkpoint under {BERT_CHECKPOINT_DIR}."
    )


def _export_checkpoint_to_deployment(checkpoint_dir: Path) -> None:
    """Export a checkpoint to the deployment folder if it does not exist."""
    if BERT_MODEL_DIR.exists():
        return

    import shutil

    BERT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    for file_name in ["config.json", "model.safetensors", "training_args.bin"]:
        src = checkpoint_dir / file_name
        if src.exists():
            shutil.copy2(src, BERT_MODEL_DIR / file_name)

    # Load and save tokenizer from checkpoint locally.
    transformers_logging.disable_progress_bar()
    hf_disable_progress_bars()
    try:
        try:
            tokenizer = BertTokenizer.from_pretrained(checkpoint_dir, local_files_only=True)
        except OSError as exc:
            raise FileNotFoundError(
                "Unable to export the tokenizer from the local checkpoint. "
                "Ensure the checkpoint contains tokenizer files such as tokenizer.json, "
                "tokenizer_config.json, and special_tokens_map.json."
            ) from exc
        tokenizer.save_pretrained(BERT_MODEL_DIR)
        _ensure_special_tokens_map(BERT_MODEL_DIR, tokenizer)
    finally:
        transformers_logging.enable_progress_bar()
        hf_enable_progress_bars()


def _ensure_special_tokens_map(directory: Path, tokenizer: BertTokenizer) -> None:
    """Save a special_tokens_map.json if it is missing from the tokenizer directory."""
    target_path = directory / "special_tokens_map.json"
    if target_path.exists():
        return

    special_tokens = getattr(tokenizer, "special_tokens_map", None)
    if not special_tokens:
        return

    with target_path.open("w", encoding="utf-8") as file:
        json.dump(special_tokens, file, indent=2)


def load_bert_model(
    model_dir: Path | None = None,
) -> tuple[BertForSequenceClassification, BertTokenizer, dict[str, Any]]:
    """
    Load the fine-tuned BERT model, tokenizer, and label mapping.

    Returns (model, tokenizer, label_mapping).
    """
    directory = _resolve_bert_model_directory(model_dir)
    label_mapping = load_label_mapping()

    if not (directory / "config.json").exists():
        raise FileNotFoundError(f"Missing BERT config.json in {directory}")
    if not (directory / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing BERT model.safetensors in {directory}")

    transformers_logging.disable_progress_bar()
    hf_disable_progress_bars()
    try:
        model = BertForSequenceClassification.from_pretrained(directory, local_files_only=True)
    finally:
        transformers_logging.enable_progress_bar()
        hf_enable_progress_bars()

    try:
        transformers_logging.disable_progress_bar()
        hf_disable_progress_bars()
        tokenizer = BertTokenizer.from_pretrained(directory, local_files_only=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"Unable to load tokenizer from local directory {directory}. "
            "Ensure tokenizer.json, tokenizer_config.json, and special_tokens_map.json exist."
        ) from exc
    finally:
        transformers_logging.enable_progress_bar()
        hf_enable_progress_bars()

    _ensure_special_tokens_map(directory, tokenizer)
    device = get_device()
    model.to(device)
    model.eval()
    return model, tokenizer, label_mapping


def load_bert_artifacts(
    model_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Load all BERT artifacts needed for later dual-model integration (Step 6).

    Returns a dict with keys: model, tokenizer, label_mapping.
    Mirrors ``load_bilstm_artifacts()`` from the BiLSTM module.
    """
    model, tokenizer, label_mapping = load_bert_model(model_dir)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "label_mapping": label_mapping,
    }


def _prepare_bert_inputs(
    texts: list[str] | pd.Series,
    tokenizer: BertTokenizer,
    max_length: int = BERT_MAX_LENGTH,
) -> dict[str, torch.Tensor]:
    """Tokenize texts and move tensors to the appropriate device."""
    cleaned = pd.Series(texts).fillna("").astype(str)
    empty_mask = cleaned.str.strip().eq("")
    if empty_mask.any():
        cleaned = cleaned.copy()
        cleaned.loc[empty_mask] = BERT_EMPTY_TEXT_PLACEHOLDER

    encodings = tokenizer(
        cleaned.tolist(),
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    device = get_device()
    return {key: value.to(device) for key, value in encodings.items()}


def predict_bert_probabilities(
    texts: list[str] | pd.Series,
    model: BertForSequenceClassification | None = None,
    tokenizer: BertTokenizer | None = None,
    max_length: int = BERT_MAX_LENGTH,
) -> np.ndarray:
    """
    Return class probability vectors for the given raw text inputs.

    Expects original (unprocessed) text strings.
    """
    if model is None or tokenizer is None:
        model, tokenizer, _ = load_bert_model()

    inputs = _prepare_bert_inputs(texts, tokenizer, max_length)
    device = get_device()

    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)

    return probabilities.cpu().numpy()


def predict_bert(
    texts: list[str] | pd.Series,
    model: BertForSequenceClassification | None = None,
    tokenizer: BertTokenizer | None = None,
    label_mapping: dict[str, Any] | None = None,
    max_length: int = BERT_MAX_LENGTH,
    return_label_names: bool = True,
) -> np.ndarray:
    """
    Predict emotion labels for raw text inputs.

    Returns string emotion names when ``return_label_names=True`` (default),
    otherwise returns integer label IDs.
    """
    if label_mapping is None:
        label_mapping = load_label_mapping()

    probabilities = predict_bert_probabilities(texts, model, tokenizer, max_length)
    predicted_ids = np.argmax(probabilities, axis=-1)

    if not return_label_names:
        return predicted_ids

    id2label = label_mapping["id2label"]
    return np.array([id2label[str(label_id)] for label_id in predicted_ids])


def prepare_bert_training_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    """
    Load splits, handle empty text, and build label mapping.

    Convenience wrapper for training scripts and notebooks.
    """
    train_df, val_df, test_df = load_split_datasets()
    train_df, _ = handle_empty_original_text(train_df)
    val_df, _ = handle_empty_original_text(val_df)
    test_df, _ = handle_empty_original_text(test_df)

    label_encoder = load_label_encoder()
    label_mapping = build_label_mapping(label_encoder)
    return train_df, val_df, test_df, label_mapping

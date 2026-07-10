"""
BERT emotion classifier fine-tuning entry point.

Usage:
    py scripts/train_bert.py
"""

from __future__ import annotations

import joblib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.bilstm_model import load_split_datasets
from utils.bert_model import (
    build_label_mapping,
    build_training_arguments,
    create_bert_classifier,
    create_emotion_datasets,
    evaluate_bert_on_test,
    extract_epoch_metrics,
    get_class_names_from_mapping,
    get_device,
    handle_empty_original_text,
    load_bert_tokenizer,
    load_label_encoder,
    plot_bert_confusion_matrix,
    plot_bert_training_history,
    print_bert_report,
    save_bert_model,
    save_label_mapping,
    save_training_history,
    train_bert_model,
)
from utils.constants import (
    BERT_CONFUSION_MATRIX_PATH,
    BERT_HISTORY_PATH,
    BERT_LABEL_MAPPING_PATH,
    BERT_MODEL_DIR,
    BERT_TRAINING_CURVES_PATH,
    LABEL_ENCODER_PATH,
)


def prepare_bert_training_data_with_schema():
    train_df, val_df, test_df = load_split_datasets()
    train_df, _ = handle_empty_original_text(train_df)
    val_df, _ = handle_empty_original_text(val_df)
    test_df, _ = handle_empty_original_text(test_df)

    if 'target_emotion' not in train_df.columns:
        raise ValueError('Dataset must contain `target_emotion` column for BERT training.')

    if Path(LABEL_ENCODER_PATH).exists():
        label_encoder = load_label_encoder()
    else:
        label_encoder = LabelEncoder()
        label_encoder.fit(
            pd.concat(
                [
                    train_df['target_emotion'].astype(str),
                    val_df['target_emotion'].astype(str),
                    test_df['target_emotion'].astype(str),
                ],
                ignore_index=True,
            )
        )
        joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    for df in (train_df, val_df, test_df):
        df['target_emotion_id'] = label_encoder.transform(df['target_emotion'].astype(str))

    return train_df, val_df, test_df, build_label_mapping(label_encoder)


def main() -> None:
    """Run the full BERT fine-tuning and evaluation pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("AI Learning Assistant — BERT Emotion Classification Training")
    print("=" * 70)
    print(f"\n  Device: {get_device()}")
    print("  Model : bert-base-uncased")
    print("  Text  : original 'text' column (BERT tokenizer handles tokenization)")

    # 1. Load datasets and label mapping
    print("\n[1/7] Loading datasets...")
    train_df, val_df, test_df, label_mapping = prepare_bert_training_data_with_schema()
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # 2. Tokenize with BertTokenizer
    print("\n[2/7] Tokenizing with BertTokenizer...")
    tokenizer = load_bert_tokenizer()
    train_dataset, val_dataset, test_dataset = create_emotion_datasets(
        train_df,
        val_df,
        test_df,
        tokenizer,
        label_col='target_emotion_id',
    )

    # 3. Build sequence classification model
    print("\n[3/7] Building BERT sequence classification model (5 classes)...")
    model = create_bert_classifier(label_mapping=label_mapping)

    # 4. Fine-tune with AdamW, LR scheduler, and early stopping
    print("\n[4/7] Fine-tuning BERT...")
    training_args = build_training_arguments()
    trainer, _ = train_bert_model(model, train_dataset, val_dataset, training_args)

    # 5. Evaluate on test set
    print("\n[5/7] Evaluating on test set...")
    eval_results = evaluate_bert_on_test(trainer, test_dataset, label_mapping)
    test_metrics = trainer.evaluate(test_dataset)
    eval_results["test_loss"] = float(test_metrics.get("eval_loss", float("nan")))

    # 6. Save artifacts
    print("\n[6/7] Saving model, tokenizer, label mapping, and history...")
    save_bert_model(trainer.model, tokenizer, BERT_MODEL_DIR)
    save_label_mapping(label_mapping, BERT_LABEL_MAPPING_PATH)
    history = extract_epoch_metrics(trainer)
    history["test_accuracy"] = eval_results["test_accuracy"]
    history["test_precision"] = eval_results["test_precision"]
    history["test_recall"] = eval_results["test_recall"]
    history["test_f1"] = eval_results["test_f1"]
    history["test_loss"] = eval_results["test_loss"]
    save_training_history(history, BERT_HISTORY_PATH)

    class_names = get_class_names_from_mapping(label_mapping)
    plot_bert_training_history(history, BERT_TRAINING_CURVES_PATH)
    plot_bert_confusion_matrix(
        eval_results["confusion_matrix"], class_names, BERT_CONFUSION_MATRIX_PATH
    )

    print(f"  Model saved         -> {BERT_MODEL_DIR}")
    print(f"  Label mapping saved -> {BERT_LABEL_MAPPING_PATH}")
    print(f"  History saved       -> {BERT_HISTORY_PATH}")
    print(f"  Curves saved        -> {BERT_TRAINING_CURVES_PATH}")
    print(f"  Confusion matrix    -> {BERT_CONFUSION_MATRIX_PATH}")

    # 7. Print report
    print("\n[7/7] Generating report...")
    print_bert_report(trainer, train_dataset, eval_results, history)


if __name__ == "__main__":
    main()

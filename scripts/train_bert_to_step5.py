"""
Run BERT fine-tuning pipeline up to Step 5 (evaluation) and stop.

This script mirrors `scripts/train_bert.py` but intentionally stops after
evaluating on the test set (Step 5) so it can be used to continue/inspect
training without saving artifacts or generating plots.

Usage:
    py scripts/train_bert_to_step5.py
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
    get_class_names_from_mapping,
    get_device,
    handle_empty_original_text,
    load_bert_tokenizer,
    load_label_encoder,
    train_bert_model,
)
from utils.constants import LABEL_ENCODER_PATH


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
    """Execute BERT pipeline through Step 5 (evaluation) and then exit."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("BERT: Run up to Step 5 (evaluation)")
    print("=" * 70)
    print(f"\n  Device: {get_device()}")

    # 1. Load datasets and label mapping
    print("\n[1/5] Loading datasets...")
    train_df, val_df, test_df, label_mapping = prepare_bert_training_data_with_schema()
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # 2. Tokenize with BertTokenizer
    print("\n[2/5] Tokenizing with BertTokenizer...")
    tokenizer = load_bert_tokenizer()
    train_dataset, val_dataset, test_dataset = create_emotion_datasets(
        train_df,
        val_df,
        test_df,
        tokenizer,
        label_col='target_emotion_id',
    )

    # 3. Build sequence classification model
    print("\n[3/5] Building BERT sequence classification model...")
    model = create_bert_classifier(label_mapping=label_mapping)

    # 4. Fine-tune with AdamW, LR scheduler, and early stopping
    print("\n[4/5] Fine-tuning BERT (training)...")
    training_args = build_training_arguments()
    trainer, _ = train_bert_model(model, train_dataset, val_dataset, training_args)

    # 5. Evaluate on test set and print summary, then stop
    print("\n[5/5] Evaluating on test set (stopping after this step)...")
    eval_results = evaluate_bert_on_test(trainer, test_dataset, label_mapping)
    test_metrics = trainer.evaluate(test_dataset)
    eval_results["test_loss"] = float(test_metrics.get("eval_loss", float("nan")))

    class_names = get_class_names_from_mapping(label_mapping)
    print("\nEvaluation results:")
    print(f"  Test Accuracy : {eval_results['test_accuracy']:.4f}")
    print(f"  Test Precision: {eval_results['test_precision']:.4f}")
    print(f"  Test Recall   : {eval_results['test_recall']:.4f}")
    print(f"  Test F1       : {eval_results['test_f1']:.4f}")
    print("\n--- Confusion Matrix (Test Set) ---")
    print(eval_results["confusion_matrix"]) 
    print("\n--- Classification Report (Test Set) ---")
    print(eval_results["classification_report"]) 


if __name__ == "__main__":
    main()

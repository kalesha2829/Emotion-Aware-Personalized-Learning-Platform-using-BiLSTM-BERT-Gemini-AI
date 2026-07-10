"""
GPU-safe BERT training runner for Colab / Kaggle.

This script enforces GPU availability and reads hyperparameters from
`configs/bert_training_kaggle.json`. It will abort if no CUDA device is found,
preventing accidental CPU training.

Do not run this on a machine without GPU.
"""

from __future__ import annotations

import json
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
    get_device,
    handle_empty_original_text,
    load_bert_tokenizer,
    load_label_encoder,
    save_bert_model,
    save_label_mapping,
    save_training_history,
    train_bert_model,
)
from utils.constants import (
    BERT_LABEL_MAPPING_PATH,
    BERT_MODEL_DIR,
    BERT_HISTORY_PATH,
    LABEL_ENCODER_PATH,
)
import torch


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


def load_config(path: Path | None = None) -> dict:
    cfg_path = Path(path or PROJECT_ROOT / "configs/bert_training_kaggle.json")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    cfg = load_config()

    # Enforce GPU runtime
    if not torch.cuda.is_available():
        print("ERROR: No CUDA-capable device found. Aborting to avoid CPU training.")
        sys.exit(2)

    print("GPU detected — proceeding with training on CUDA.")

    train_df, val_df, test_df, label_mapping = prepare_bert_training_data_with_schema()
    tokenizer = load_bert_tokenizer()
    train_dataset, val_dataset, test_dataset = create_emotion_datasets(
        train_df,
        val_df,
        test_df,
        tokenizer,
        label_col='target_emotion_id',
        max_length=cfg.get("max_length", 128),
    )

    model = create_bert_classifier(label_mapping=label_mapping)

    training_args = build_training_arguments(
        num_epochs=cfg.get("epochs", 3),
        batch_size=cfg.get("batch_size", 32),
        eval_batch_size=cfg.get("eval_batch_size", 64),
        learning_rate=cfg.get("learning_rate", 2e-5),
        weight_decay=cfg.get("weight_decay", 0.01),
        warmup_ratio=cfg.get("warmup_ratio", 0.1),
    )

    trainer, train_result = train_bert_model(
        model,
        train_dataset,
        val_dataset,
        training_args,
        early_stopping_patience=cfg.get("early_stopping_patience", 2),
    )

    # Evaluate and save artifacts
    eval_results = evaluate_bert_on_test(trainer, test_dataset, label_mapping)

    save_bert_model(trainer.model, tokenizer, Path(cfg.get("save_dir", BERT_MODEL_DIR)))
    save_label_mapping(label_mapping, Path(BERT_LABEL_MAPPING_PATH))

    history = trainer.state.log_history
    # Convert history extraction to existing helper for consistency
    history_obj = {"trainer_log_history": history}
    save_training_history(history_obj, Path(BERT_HISTORY_PATH))

    print("Training complete — artifacts saved to configured paths.")


if __name__ == "__main__":
    main()

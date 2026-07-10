"""
BiLSTM emotion classifier training entry point.

Usage:
    py scripts/train_bilstm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.bilstm_model import (
    build_bilstm_model,
    create_and_fit_tokenizer,
    evaluate_bilstm_model,
    get_training_callbacks,
    handle_empty_processed_text,
    infer_max_sequence_length,
    load_label_encoder,
    load_split_datasets,
    plot_confusion_matrix,
    plot_training_history,
    prepare_features,
    print_training_report,
    save_max_len,
    save_tokenizer,
    save_training_history,
    texts_to_sequences,
    train_bilstm_model,
)
from utils.constants import (
    BATCH_SIZE,
    BILSTM_CONFUSION_MATRIX_PATH,
    BILSTM_HISTORY_PATH,
    BILSTM_MAX_LEN_PATH,
    BILSTM_MODEL_PATH,
    BILSTM_TOKENIZER_PATH,
    BILSTM_TRAINING_CURVES_PATH,
    EPOCHS,
    MAX_SEQUENCE_LENGTH,
)


def main() -> None:
    """Run the full BiLSTM training and evaluation pipeline."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 70)
    print("AI Learning Assistant — BiLSTM Emotion Classification Training")
    print("=" * 70)

    # 1. Load datasets
    print("\n[1/8] Loading train, validation, and test datasets...")
    train_df, val_df, test_df = load_split_datasets()
    print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

    # 2. Handle empty processed_text with placeholder token
    print("\n[2/8] Handling empty processed_text records...")
    print(
        "  Strategy: Replace empty strings with '<EMPTY>' placeholder token.\n"
        "  Rationale: Preserves labeled samples and gives the Tokenizer a dedicated\n"
        "  index instead of producing zero-length sequences after stopword removal."
    )
    train_df, train_empty = handle_empty_processed_text(train_df)
    val_df, val_empty = handle_empty_processed_text(val_df)
    test_df, test_empty = handle_empty_processed_text(test_df)
    print(f"  Replaced -> Train: {train_empty}, Val: {val_empty}, Test: {test_empty}")

    # 3. Fit Tokenizer on training text only
    print("\n[3/8] Fitting Keras Tokenizer on training data...")
    tokenizer = create_and_fit_tokenizer(train_df["processed_text"])
    vocab_size = len(tokenizer.word_index) + 1  # +1 for padding index 0
    print(f"  Vocabulary size: {vocab_size:,}")

    train_sequences = texts_to_sequences(train_df["processed_text"], tokenizer)
    max_len = max(
        infer_max_sequence_length(train_sequences),
        MAX_SEQUENCE_LENGTH,
    )
    print(f"  Max sequence length: {max_len}")

    # 4. Convert to padded integer sequences
    print("\n[4/8] Converting text to padded integer sequences...")
    x_train = prepare_features(train_df["processed_text"], tokenizer, max_len)
    x_val = prepare_features(val_df["processed_text"], tokenizer, max_len)
    x_test = prepare_features(test_df["processed_text"], tokenizer, max_len)

    y_train = train_df["label"].to_numpy()
    y_val = val_df["label"].to_numpy()
    y_test = test_df["label"].to_numpy()
    print(f"  Shapes -> Train: {x_train.shape}, Val: {x_val.shape}, Test: {x_test.shape}")

    # 5. Build BiLSTM model
    print("\n[5/8] Building BiLSTM architecture...")
    model = build_bilstm_model(vocab_size=vocab_size, max_len=max_len)
    model.summary()

    # 6. Train with EarlyStopping and ModelCheckpoint
    print("\n[6/8] Training model...")
    callbacks = get_training_callbacks(model_path=BILSTM_MODEL_PATH)
    history = train_bilstm_model(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    # 7. Evaluate on test set
    print("\n[7/8] Evaluating on test set...")
    label_encoder = load_label_encoder()
    eval_results = evaluate_bilstm_model(model, x_test, y_test, label_encoder)

    # 8. Save artifacts and generate plots/reports
    print("\n[8/8] Saving artifacts and generating plots...")
    save_tokenizer(tokenizer, BILSTM_TOKENIZER_PATH)
    save_max_len(max_len, BILSTM_MAX_LEN_PATH)
    save_training_history(history, BILSTM_HISTORY_PATH)
    plot_training_history(history, BILSTM_TRAINING_CURVES_PATH)
    plot_confusion_matrix(
        eval_results["confusion_matrix"],
        list(label_encoder.classes_),
        BILSTM_CONFUSION_MATRIX_PATH,
    )

    print(f"  Model saved    -> {BILSTM_MODEL_PATH}")
    print(f"  Tokenizer saved-> {BILSTM_TOKENIZER_PATH}")
    print(f"  History saved  -> {BILSTM_HISTORY_PATH}")
    print(f"  Curves saved   -> {BILSTM_TRAINING_CURVES_PATH}")
    print(f"  Confusion matrix -> {BILSTM_CONFUSION_MATRIX_PATH}")

    print_training_report(history, eval_results, label_encoder)


if __name__ == "__main__":
    main()

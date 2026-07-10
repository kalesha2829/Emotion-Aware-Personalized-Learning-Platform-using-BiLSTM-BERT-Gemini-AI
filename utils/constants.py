"""Project-wide constants for dataset collection and exploration."""

from pathlib import Path

# Project root (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Target emotion classes for the AI Learning Assistant
TARGET_EMOTIONS = [
    "Bored",
    "Confident",
    "Confused",
    "Curious",
    "Frustrated",
]

# Dataset paths
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_DATASET_PATH = DATASET_DIR / "go_emotions_raw.csv"
CLEANED_DATASET_PATH = DATASET_DIR / "student_emotions_cleaned.csv"

# Preprocessed dataset paths
PROCESSED_DATASET_PATH = DATASET_DIR / "processed_full.csv"
TRAIN_DATASET_PATH = DATASET_DIR / "train.csv"
VAL_DATASET_PATH = DATASET_DIR / "validation.csv"
TEST_DATASET_PATH = DATASET_DIR / "test.csv"

# Visualization output paths
ASSETS_DIR = PROJECT_ROOT / "assets"
EMOTION_BAR_CHART_PATH = ASSETS_DIR / "emotion_distribution_bar.html"
EMOTION_PIE_CHART_PATH = ASSETS_DIR / "emotion_distribution_pie.html"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
BILSTM_MODEL_PATH = MODELS_DIR / "bilstm_emotion.keras"
BILSTM_TOKENIZER_PATH = MODELS_DIR / "bilstm_tokenizer.joblib"
BILSTM_HISTORY_PATH = MODELS_DIR / "bilstm_training_history.json"
BILSTM_MAX_LEN_PATH = MODELS_DIR / "bilstm_max_len.joblib"
BILSTM_TRAINING_CURVES_PATH = ASSETS_DIR / "bilstm_training_curves.png"
BILSTM_CONFUSION_MATRIX_PATH = ASSETS_DIR / "bilstm_confusion_matrix.png"

# BERT model artifacts
BERT_MODEL_NAME = "bert-base-uncased"
BERT_MODEL_DIR = MODELS_DIR / "bert_emotion"
BERT_CHECKPOINT_DIR = MODELS_DIR / "bert_checkpoints"
BERT_LABEL_MAPPING_PATH = MODELS_DIR / "bert_label_mapping.json"
BERT_HISTORY_PATH = MODELS_DIR / "bert_training_history.json"
BERT_TRAINING_CURVES_PATH = ASSETS_DIR / "bert_training_curves.png"
BERT_CONFUSION_MATRIX_PATH = ASSETS_DIR / "bert_confusion_matrix.png"

# BERT hyperparameters
BERT_MAX_LENGTH = 128
BERT_BATCH_SIZE = 32
BERT_EVAL_BATCH_SIZE = 64
BERT_EPOCHS = 3
BERT_LEARNING_RATE = 2e-5
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_EARLY_STOPPING_PATIENCE = 2
BERT_EMPTY_TEXT_PLACEHOLDER = "[EMPTY]"

# BiLSTM hyperparameters
EMPTY_TEXT_PLACEHOLDER = "<EMPTY>"
NUM_CLASSES = len(TARGET_EMOTIONS)
EMBEDDING_DIM = 128
LSTM_UNITS = 64
DENSE_UNITS = 64
DROPOUT_RATE = 0.3
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 1e-3
MAX_SEQUENCE_LENGTH = 25  # Covers ~99th percentile of tokenized training lengths

# Train / validation / test split ratios
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_STATE = 42

# Hugging Face dataset (mirrors Kaggle: google-research-datasets/go-emotions)
HF_DATASET_NAME = "google-research-datasets/go_emotions"
HF_DATASET_CONFIG = "simplified"

"""Reusable CSV logging utilities for prediction interactions."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.gemini import GeminiLearningSupport

LOG_DIRECTORY = Path(__file__).resolve().parents[1] / "logs"
INTERACTIONS_CSV = LOG_DIRECTORY / "interactions.csv"
FIELDNAMES = [
    "timestamp",
    "student_input",
    "bilstm_prediction",
    "bert_prediction",
    "final_emotion",
    "confidence_score",
    "mixed_emotion_breakdown",
    "gemini_response",
]


def _ensure_log_file() -> Path:
    """Ensure the interactions CSV file exists with a header row."""
    LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
    if not INTERACTIONS_CSV.exists():
        with INTERACTIONS_CSV.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writeheader()
    return INTERACTIONS_CSV


def _serialize_mixed_breakdown(mixed_emotion_breakdown: list[dict[str, Any]]) -> str:
    try:
        return json.dumps(mixed_emotion_breakdown, ensure_ascii=False)
    except Exception:
        return str(mixed_emotion_breakdown)


def _normalize_gemini_response(support: GeminiLearningSupport) -> str:
    if support.raw_response:
        return support.raw_response
    if support.error:
        return f"ERROR: {support.error}"

    payload = {
        "personalized_learning_guidance": support.personalized_learning_guidance,
        "encouraging_response": support.encouraging_response,
        "study_tips": support.study_tips,
        "next_learning_steps": support.next_learning_steps,
    }
    try:
        return json.dumps(payload, ensure_ascii=False)
    except Exception:
        return str(payload)


def log_interaction(
    student_input: str,
    bilstm_prediction: str,
    bert_prediction: str,
    final_emotion: str,
    confidence_score: float,
    mixed_emotion_breakdown: list[dict[str, Any]],
    gemini_support: GeminiLearningSupport,
) -> tuple[bool, str | None]:
    """Append a prediction interaction to the CSV log file."""
    try:
        log_file = _ensure_log_file()
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "student_input": student_input,
            "bilstm_prediction": bilstm_prediction,
            "bert_prediction": bert_prediction,
            "final_emotion": final_emotion,
            "confidence_score": f"{confidence_score:.6f}",
            "mixed_emotion_breakdown": _serialize_mixed_breakdown(mixed_emotion_breakdown),
            "gemini_response": _normalize_gemini_response(gemini_support),
        }
        with log_file.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            writer.writerow(row)
        return True, None
    except OSError as exc:
        return False, str(exc)
    except Exception as exc:
        return False, str(exc)


GEMINI_ERRORS_CSV = LOG_DIRECTORY / "gemini_errors.csv"
GEMINI_ERR_FIELDNAMES = [
    "timestamp",
    "model_used",
    "http_status",
    "exception_type",
    "error_message",
    "complete_error",
]


def log_gemini_error(
    model_used: str,
    http_status: str | int,
    exception_type: str,
    error_message: str,
    complete_error: str,
) -> None:
    """Log Gemini API errors to logs/gemini_errors.csv."""
    try:
        LOG_DIRECTORY.mkdir(parents=True, exist_ok=True)
        file_exists = GEMINI_ERRORS_CSV.exists()
        with GEMINI_ERRORS_CSV.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=GEMINI_ERR_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model_used": model_used,
                "http_status": str(http_status),
                "exception_type": exception_type,
                "error_message": error_message,
                "complete_error": complete_error,
            })
    except Exception as e:
        print(f"[LOGGER ERROR] Failed to log Gemini error: {e}")


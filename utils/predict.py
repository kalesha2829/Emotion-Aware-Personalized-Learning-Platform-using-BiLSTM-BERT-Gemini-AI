"""Emotion detection pipeline for the AI Learning Assistant.

This module loads trained BiLSTM and BERT artifacts, preprocesses student text
inputs for each model, computes prediction probabilities, and combines model
outputs into a final emotion decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from utils.bilstm_model import (
    decode_emotion_labels,
    load_bilstm_artifacts,
    predict_emotion_labels,
)
from utils.bert_model import load_bert_artifacts, predict_bert_probabilities
from utils.preprocess import preprocess_text


@dataclass
class EmotionDetectionResult:
    bilstm_prediction: str
    bert_prediction: str
    final_emotion: str
    confidence_score: float
    mixed_emotion_breakdown: list[dict[str, Any]]
    bilstm_confidence: float
    bert_confidence: float


class EmotionDetectionPipeline:
    """Pipeline for combined BiLSTM + BERT emotion detection."""

    MIXED_EMOTION_THRESHOLD = 0.18
    AGREEMENT_MARGIN = 0.15

    def __init__(self, bilstm_artifacts: dict[str, Any] | None = None, bert_artifacts: dict[str, Any] | None = None):
        self.bilstm_artifacts = bilstm_artifacts or load_bilstm_artifacts()
        self.bert_artifacts = bert_artifacts or load_bert_artifacts()
        self.label_names = self._load_label_names()

    def _load_label_names(self) -> list[str]:
        label_mapping = self.bert_artifacts["label_mapping"]
        return [label_mapping["id2label"][str(i)] for i in sorted(int(k) for k in label_mapping["id2label"])]

    def predict(self, text: str) -> EmotionDetectionResult:
        """Run the emotion detection pipeline for a single student input text."""
        text = str(text or "")

        bilstm_label, bilstm_confidence, bilstm_probs = self._predict_bilstm(text)
        bert_label, bert_confidence, bert_probs = self._predict_bert(text)

        combined_probs = self._combine_probabilities(bilstm_probs, bert_probs)
        blended_breakdown = self._build_mixed_breakdown(combined_probs)
        final_emotion, final_confidence = self._resolve_final_emotion(
            bilstm_label,
            bert_label,
            bilstm_confidence,
            bert_confidence,
            combined_probs,
        )

        return EmotionDetectionResult(
            bilstm_prediction=bilstm_label,
            bert_prediction=bert_label,
            final_emotion=final_emotion,
            confidence_score=final_confidence,
            mixed_emotion_breakdown=blended_breakdown,
            bilstm_confidence=bilstm_confidence,
            bert_confidence=bert_confidence,
        )

    def _predict_bilstm(self, text: str) -> tuple[str, float, np.ndarray]:
        processed_text = preprocess_text(text)
        label_ids, probabilities = predict_emotion_labels(
            [processed_text], self.bilstm_artifacts
        )
        label_name = decode_emotion_labels(
            label_ids, self.bilstm_artifacts["label_encoder"]
        )[0]
        confidence = float(np.max(probabilities[0]))
        return label_name, confidence, probabilities[0]

    def _predict_bert(self, text: str) -> tuple[str, float, np.ndarray]:
        probabilities = predict_bert_probabilities(
            [text],
            model=self.bert_artifacts["model"],
            tokenizer=self.bert_artifacts["tokenizer"],
        )
        best_idx = int(np.argmax(probabilities[0]))
        label_name = self.bert_artifacts["label_mapping"]["id2label"][str(best_idx)]
        confidence = float(np.max(probabilities[0]))
        return label_name, confidence, probabilities[0]

    def _combine_probabilities(self, bilstm_probs: np.ndarray, bert_probs: np.ndarray) -> np.ndarray:
        """Average BiLSTM and BERT probability vectors to form an ensemble signal."""
        return (bilstm_probs + bert_probs) / 2.0

    def _build_mixed_breakdown(self, combined_probs: np.ndarray) -> list[dict[str, Any]]:
        top_indices = np.argsort(-combined_probs)[:2]
        breakdown: list[dict[str, Any]] = []

        for rank, idx in enumerate(top_indices, start=1):
            score = float(combined_probs[idx])
            if rank == 2 and score < self.MIXED_EMOTION_THRESHOLD:
                break
            breakdown.append(
                {
                    "emotion": self.label_names[idx],
                    "confidence": score,
                }
            )

        return breakdown

    def _resolve_final_emotion(
        self,
        bilstm_label: str,
        bert_label: str,
        bilstm_confidence: float,
        bert_confidence: float,
        combined_probs: np.ndarray,
    ) -> tuple[str, float]:
        if bilstm_label == bert_label:
            return bilstm_label, float((bilstm_confidence + bert_confidence) / 2.0)

        if abs(bilstm_confidence - bert_confidence) >= self.AGREEMENT_MARGIN:
            best_label = bilstm_label if bilstm_confidence > bert_confidence else bert_label
            best_confidence = max(bilstm_confidence, bert_confidence)
            return best_label, best_confidence

        best_idx = int(np.argmax(combined_probs))
        return self.label_names[best_idx], float(combined_probs[best_idx])

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.features import tokenize

STOPWORDS = {
    "a", "al", "and", "con", "de", "del", "el", "en", "es", "fue", "la", "las",
    "lo", "los", "me", "mi", "no", "para", "por", "se", "sin", "the", "un",
    "una", "y",
}


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[str]


class SentimentModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.artifact: dict[str, Any] | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            msg = (
                f"No existe el modelo en {self.model_path}. "
                "Ejecuta: uv run python scripts/train_model.py"
            )
            raise FileNotFoundError(msg)
        self.artifact = json.loads(self.model_path.read_text(encoding="utf-8"))

    @property
    def is_loaded(self) -> bool:
        return self.artifact is not None

    def predict(self, text: str) -> Prediction:
        if self.artifact is None:
            raise RuntimeError("El modelo no está cargado.")

        labels: list[str] = self.artifact["labels"]
        priors: dict[str, float] = self.artifact["priors"]
        token_log_probs: dict[str, dict[str, float]] = self.artifact["token_log_probs"]
        unknown_log_probs: dict[str, float] = self.artifact["unknown_log_probs"]

        tokens = tokenize(text)
        scores: dict[str, float] = {}

        for label in labels:
            score = math.log(priors[label])
            for token in tokens:
                score += token_log_probs[label].get(token, unknown_log_probs[label])
            scores[label] = score

        probabilities = _softmax(scores)
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]
        explanation = self._explain_prediction(text, label, tokens)

        return Prediction(
            label=label,
            confidence=round(confidence, 4),
            probabilities={key: round(value, 4) for key, value in probabilities.items()},
            explanation=explanation,
        )

    def _explain_prediction(self, text: str, predicted_label: str, tokens: list[str]) -> list[str]:
        if self.artifact is None:
            return []

        token_log_probs: dict[str, dict[str, float]] = self.artifact["token_log_probs"]
        unknown_log_probs: dict[str, float] = self.artifact["unknown_log_probs"]
        labels: list[str] = self.artifact["labels"]
        other_labels = [label for label in labels if label != predicted_label]

        explanations: list[str] = []
        token_counter = Counter(tokens)
        contributions: list[tuple[str, float]] = []
        for token, count in token_counter.items():
            if token in STOPWORDS or len(token) <= 2:
                continue
            predicted_log_prob = token_log_probs[predicted_label].get(
                token,
                unknown_log_probs[predicted_label],
            )
            other_average = sum(
                token_log_probs[label].get(token, unknown_log_probs[label]) for label in other_labels
            ) / max(len(other_labels), 1)
            contribution = (predicted_log_prob - other_average) * count
            if contribution > 0:
                contributions.append((token, contribution))

        for token, _ in sorted(contributions, key=lambda item: item[1], reverse=True)[:4]:
            explanations.append(
                f"La palabra '{token}' aumenta la probabilidad de la clase '{predicted_label}'."
            )

        if not explanations:
            explanations.append(
                "La predicción se basa en la combinación estadística de palabras aprendidas; "
                "no hubo una palabra dominante clara."
            )
        return explanations[:5]


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    max_score = max(scores.values())
    exp_scores = {label: math.exp(score - max_score) for label, score in scores.items()}
    total = sum(exp_scores.values())
    return {label: value / total for label, value in exp_scores.items()}

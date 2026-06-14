from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

from app.features import tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "training_texts.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_model.json"
ALPHA = 1.0


def load_training_data(path: Path = DATA_PATH) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        required_columns = {"text", "label"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing_columns)}")
        for row in reader:
            text = (row.get("text") or "").strip()
            label = (row.get("label") or "").strip()
            if text and label:
                rows.append({"text": text, "label": label})
    if not rows:
        raise ValueError("El dataset de entrenamiento está vacío.")
    return rows


def train_and_save_model() -> dict[str, object]:
    rows = load_training_data()
    labels = sorted({row["label"] for row in rows})
    label_counts = Counter(row["label"] for row in rows)
    token_counts_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    total_tokens_by_label: Counter[str] = Counter()
    vocabulary: set[str] = set()

    for row in rows:
        label = row["label"]
        tokens = tokenize(row["text"])
        token_counts_by_label[label].update(tokens)
        total_tokens_by_label[label] += len(tokens)
        vocabulary.update(tokens)

    vocab_size = len(vocabulary)
    total_rows = len(rows)
    priors = {label: label_counts[label] / total_rows for label in labels}
    token_log_probs: dict[str, dict[str, float]] = {}
    unknown_log_probs: dict[str, float] = {}

    for label in labels:
        denominator = total_tokens_by_label[label] + ALPHA * (vocab_size + 1)
        token_log_probs[label] = {
            token: math.log((token_counts_by_label[label][token] + ALPHA) / denominator)
            for token in sorted(vocabulary)
        }
        unknown_log_probs[label] = math.log(ALPHA / denominator)

    metadata = {
        "model_type": "Multinomial Naive Bayes pure Python",
        "problem_type": "text_sentiment_classification",
        "labels": labels,
        "train_rows": total_rows,
        "vocab_size": vocab_size,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "notes": "Modelo Naive Bayes entrenado desde data/training_texts.csv; sin reglas de decisión hardcodeadas ni API keys.",
    }

    artifact = {
        "labels": labels,
        "priors": priors,
        "token_log_probs": token_log_probs,
        "unknown_log_probs": unknown_log_probs,
        "metadata": metadata,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    metadata = train_and_save_model()
    print("Modelo entrenado y guardado correctamente.")
    print(f"Ruta: {MODEL_PATH}")
    print(f"Filas de entrenamiento: {metadata['train_rows']}")
    print(f"Vocabulario: {metadata['vocab_size']} tokens")


if __name__ == "__main__":
    main()

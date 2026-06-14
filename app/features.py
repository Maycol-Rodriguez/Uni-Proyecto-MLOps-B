from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    return normalized


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    return re.findall(r"[a-zñü]+", normalized)

from dataclasses import dataclass
from functools import lru_cache
from os import getenv
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_env: str
    model_path: Path
    model_version: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        app_name=getenv("APP_NAME", "ML Sentiment API"),
        app_env=getenv("APP_ENV", "production"),
        model_path=Path(getenv("MODEL_PATH", "models/sentiment_model.json")),
        model_version=getenv("MODEL_VERSION", "0.1.0"),
    )

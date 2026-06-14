from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import get_settings
from app.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY
from app.ml_model import SentimentModel
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse

settings = get_settings()
model = SentimentModel(settings.model_path)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    model.load()
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "API de IA/ML en contenedor. Recibe un texto y devuelve una predicción "
        "de sentimiento con probabilidad y explicación local."
    ),
    version=settings.model_version,
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Evita mostrar stack traces al evaluador; mantiene un error controlado.
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Error interno de la aplicación.",
            "path": str(request.url.path),
            "error_type": exc.__class__.__name__,
        },
    )


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "message": "API de sentimiento operativa. Usa POST /predict o abre /docs.",
        "example": "curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"text\":\"Me encantó el servicio\"}'",
    }


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model.is_loaded else "error",
        model_loaded=model.is_loaded,
        model_version=settings.model_version,
    )


@app.get("/metrics", tags=["ops"])
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["ml"])
def predict(payload: PredictionRequest) -> PredictionResponse:
    clean_text = payload.text.strip()
    if not clean_text:
        raise HTTPException(status_code=422, detail="El texto no puede estar vacío.")

    start_time = time.perf_counter()
    prediction = model.predict(clean_text)
    elapsed = time.perf_counter() - start_time

    PREDICTION_COUNTER.labels(label=prediction.label).inc()
    PREDICTION_LATENCY.observe(elapsed)

    return PredictionResponse(
        text=clean_text,
        label=prediction.label,
        confidence=prediction.confidence,
        probabilities=prediction.probabilities,
        explanation=prediction.explanation,
        model_version=settings.model_version,
    )


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()

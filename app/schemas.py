from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        max_length=5_000,
        examples=["Me encantó el servicio, fue rápido y muy útil."],
    )


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    explanation: list[str]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str

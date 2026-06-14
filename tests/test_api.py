from app.main import app
from fastapi.testclient import TestClient


def test_healthcheck_loads_model() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True


def test_predict_positive_text() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Me encantó el servicio, fue rápido y excelente"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "positivo"
    assert payload["confidence"] >= 0.5
    assert "positivo" in payload["probabilities"]
    assert payload["explanation"]


def test_predict_negative_text() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "La aplicación falla, es lenta y terrible"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] == "negativo"
    assert payload["confidence"] >= 0.5


def test_metrics_endpoint() -> None:
    with TestClient(app) as client:
        client.post("/predict", json={"text": "El servicio fue normal y estándar"})
        response = client.get("/metrics")
    assert response.status_code == 200
    assert "sentiment_predictions_total" in response.text

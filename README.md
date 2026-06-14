# ML Sentiment API — App de IA/ML en contenedor

API pequeña de IA/ML construida con **FastAPI**, **uv** y **Docker**. Recibe un texto y devuelve una predicción de sentimiento: `positivo`, `negativo` o `neutral`, incluyendo probabilidades y una explicación local simple. El modelo es un Multinomial Naive Bayes de texto, entrenado desde `data/training_texts.csv` e implementado en Python puro para evitar dependencias pesadas dentro del contenedor.

Este proyecto no usa API keys. Por eso no hay secretos dentro de la imagen Docker.

## 1. Qué hace

Endpoint principal:

```text
POST /predict
```

Entrada:

```json
{
  "text": "Me encantó el servicio, fue rápido y excelente"
}
```

Salida esperada aproximada:

```json
{
  "text": "Me encantó el servicio, fue rápido y excelente",
  "label": "positivo",
  "confidence": 0.99,
  "probabilities": {
    "negativo": 0.0,
    "neutral": 0.0,
    "positivo": 0.99
  },
  "explanation": [
    "La palabra 'encanto' aumenta la probabilidad de la clase 'positivo'.",
    "La palabra 'excelente' aumenta la probabilidad de la clase 'positivo'.",
    "La palabra 'rapido' aumenta la probabilidad de la clase 'positivo'."
  ],
  "model_version": "0.1.0"
}
```

## 2. Estructura del proyecto

```text
.
├── app/                       # Código FastAPI y carga del modelo
├── data/                      # Dataset pequeño de entrenamiento
├── models/                    # Artefacto entrenado .json
├── scripts/                   # Script de entrenamiento
├── tests/                     # Tests de API
├── .github/workflows/ci.yml   # CI: lint, tests, build y push opcional a GHCR
├── Dockerfile                 # Imagen productiva con python slim + uv
├── docker-compose.yml
├── pyproject.toml
└── README.md
```


### Nota importante sobre `uv.lock`

El `uv.lock` anterior se retiró porque apuntaba a un índice interno que rompía el `docker build` fuera de ese entorno. Si quieres recuperar el punto extra de lockfile antes de subirlo a GitHub, ejecútalo localmente y verifica que no existan URLs internas:

```bash
rm -f uv.lock
uv lock --default-index https://pypi.org/simple
grep -En "applied-caas|internal.api.openai" uv.lock || echo "lockfile OK"
```

Luego puedes cambiar en el Dockerfile `uv sync` por `uv sync --locked`. Para la entrega inmediata, el Dockerfile actual prioriza que la imagen construya sin depender de un lock contaminado.

## 3. Correr con Docker

Construir la imagen:

```bash
docker build -t mlops-sentiment-api:local .
```

Correr la API con un solo comando:

```bash
docker run --rm -p 8000:8000 mlops-sentiment-api:local
```

Abrir documentación interactiva:

```text
http://localhost:8000/docs
```

Probar healthcheck:

```bash
curl http://localhost:8000/health
```

Probar predicción:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"Me encantó el servicio, fue rápido y excelente"}'
```

Otro ejemplo negativo:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"La aplicación falla, es lenta y terrible"}'
```

## 4. Correr con Docker Compose

```bash
docker compose up --build
```

## 5. Desarrollo local con uv

Instalar dependencias:

```bash
uv sync
```

Entrenar nuevamente el modelo:

```bash
uv run python scripts/train_model.py
```

Levantar API local:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Ejecutar tests:

```bash
uv run pytest
```

Ejecutar lint:

```bash
uv run ruff check .
```

## 6. Métricas

La API expone métricas estilo Prometheus en:

```text
GET /metrics
```

Ejemplo:

```bash
curl http://localhost:8000/metrics
```

Métricas incluidas:

- `sentiment_predictions_total`: total de predicciones por clase.
- `sentiment_prediction_latency_seconds`: latencia de predicción.

## 7. Entrega como archivo .tar

Si quieres entregar la imagen como archivo:

```bash
docker build -t mlops-sentiment-api:local .
docker save mlops-sentiment-api:local -o mlops-sentiment-api.tar
```

El profesor puede cargarla así:

```bash
docker load -i mlops-sentiment-api.tar
docker run --rm -p 8000:8000 mlops-sentiment-api:local
```

## 8. Publicación opcional en GHCR

El workflow `.github/workflows/ci.yml` construye la imagen y puede publicarla en GitHub Container Registry cuando hagas push a `main`.

Luego se podría correr así, reemplazando `OWNER` y `SHA`:

```bash
docker run --rm -p 8000:8000 ghcr.io/OWNER/mlops-sentiment-api:SHA
```

## 9. Notas importantes

- No se suben secretos al repositorio.
- No se necesita `API_KEY` porque el modelo es local.
- El modelo incluido es pequeño y suficiente para demostrar despliegue de ML en contenedor.
- El objetivo principal es que la imagen arranque, responda y sea fácil de evaluar.

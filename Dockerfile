FROM python:3.12-slim AS builder

# uv oficial para instalar dependencias de forma rápida en la imagen.
COPY --from=ghcr.io/astral-sh/uv:0.10.0 /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_DEFAULT_INDEX=https://pypi.org/simple

WORKDIR /app

# Primero dependencias para aprovechar cache de capas.
# Nota: no usamos --locked porque se retiró el uv.lock contaminado con un índice interno.
# En tu máquina puedes regenerarlo con: uv lock --default-index https://pypi.org/simple
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Luego código y artefactos del modelo.
COPY . .
RUN uv sync --no-dev

FROM python:3.12-slim AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    APP_ENV=production \
    MODEL_PATH=models/sentiment_model.json

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser
COPY --from=builder /app /app

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

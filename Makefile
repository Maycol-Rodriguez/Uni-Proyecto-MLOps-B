IMAGE_NAME ?= mlops-sentiment-api:local

.PHONY: lock train test lint run docker-build docker-run docker-save

lock:
	uv lock --default-index https://pypi.org/simple

train:
	uv run python scripts/train_model.py

test:
	uv run pytest

lint:
	uv run ruff check .

run:
	uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8000:8000 $(IMAGE_NAME)

docker-save:
	mkdir -p docker-image
	docker save $(IMAGE_NAME) -o docker-image/mlops-sentiment-api.tar

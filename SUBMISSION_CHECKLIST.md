# Checklist de cumplimiento

| Requisito del trabajo | Evidencia en este repo |
|---|---|
| Imagen Docker arranca con un solo comando | `Dockerfile` + `docker run --rm -p 8000:8000 mlops-sentiment-api:local` |
| Servicio recibe entrada y devuelve salida IA/ML | `POST /predict` con modelo sklearn entrenado |
| README con build, run y ejemplo entrada → respuesta | `README.md` secciones 3 y 4 |
| API key en runtime si aplica | No aplica: el modelo es local y no usa secretos |
| Entrega por repo, GHCR o `.tar` | README sección 7 explica `docker save` |
| Repo limpio sin secretos | `.gitignore` + `.dockerignore` |
| Dockerfile producción slim + uv + lockfile | `Dockerfile`, `pyproject.toml`, `uv / lockfile público regenerable` |
| CI con GitHub Actions | `.github/workflows/ci.yml` |
| Tests | `tests/test_api.py` |
| Lint | `ruff` configurado en `pyproject.toml` |
| Métricas | `GET /metrics` con Prometheus |

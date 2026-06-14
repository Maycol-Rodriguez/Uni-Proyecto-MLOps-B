from prometheus_client import Counter, Histogram

PREDICTION_COUNTER = Counter(
    "sentiment_predictions_total",
    "Total de predicciones realizadas por la API.",
    ["label"],
)

PREDICTION_LATENCY = Histogram(
    "sentiment_prediction_latency_seconds",
    "Latencia de las predicciones de sentimiento.",
)

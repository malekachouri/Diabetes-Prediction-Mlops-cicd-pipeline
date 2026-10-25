"""Prometheus metrics of the prediction API, exposed on ``/metrics``.

Each app gets its own registry, so that tests can create several apps without
"duplicated timeseries" errors and without sharing counters.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Probability buckets: the share of patients close to the 0.5 threshold matters
PROBABILITY_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class ApiMetrics:
    """The metrics of one app and the registry they are exposed from."""

    registry: CollectorRegistry
    requests: Counter
    latency: Histogram
    predictions: Counter
    probability: Histogram
    missing_inputs: Counter
    model_info: Gauge


def create_metrics() -> ApiMetrics:
    """Create the API metrics in a new registry."""
    registry = CollectorRegistry()
    return ApiMetrics(
        registry=registry,
        requests=Counter(
            "diabetes_api_requests",
            "HTTP requests handled by the API",
            ["method", "endpoint", "status"],
            registry=registry,
        ),
        latency=Histogram(
            "diabetes_api_request_duration_seconds",
            "Time spent handling an HTTP request",
            ["endpoint"],
            registry=registry,
        ),
        predictions=Counter(
            "diabetes_api_predictions",
            "Predictions returned, by predicted class",
            ["diabetic"],
            registry=registry,
        ),
        probability=Histogram(
            "diabetes_api_prediction_probability",
            "Predicted probability of diabetes",
            buckets=PROBABILITY_BUCKETS,
            registry=registry,
        ),
        missing_inputs=Counter(
            "diabetes_api_missing_inputs",
            "Measurements sent as null (unknown) in prediction requests",
            ["feature"],
            registry=registry,
        ),
        model_info=Gauge(
            "diabetes_api_model_info",
            "Version of the served model (always 1)",
            ["version"],
            registry=registry,
        ),
    )

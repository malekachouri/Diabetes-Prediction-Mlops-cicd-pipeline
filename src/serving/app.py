"""Prediction API: serve the diabetes model over HTTP with FastAPI.

Run locally with ``make serve``, then open http://127.0.0.1:8000/docs.
The model is loaded once at startup; the app does not start without it.
Prometheus metrics are exposed on ``/metrics``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.routing import Match

from src.serving.metrics import create_metrics
from src.serving.model import ServingSettings, load_model
from src.serving.schemas import FEATURE_COLUMNS, Health, PatientFeatures, Prediction


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """Build the API. Settings default to the ``MODEL_*`` environment variables."""
    metrics = create_metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or ServingSettings.from_env()
        app.state.model = load_model(resolved.model_path, FEATURE_COLUMNS.values())
        app.state.model_version = resolved.model_version
        metrics.model_info.labels(version=resolved.model_version).set(1)
        yield

    app = FastAPI(
        title="Diabetes Prediction API",
        description="Predict the diabetes risk of a patient from 5 health indicators.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def record_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Label with the route template, never the raw URL: unknown paths would
        # otherwise create one time series each
        endpoint = next(
            (
                route.path
                for route in request.app.routes
                if route.matches(request.scope)[0] == Match.FULL
            ),
            "other",
        )
        start = time.perf_counter()
        response = await call_next(request)
        metrics.latency.labels(endpoint=endpoint).observe(time.perf_counter() - start)
        metrics.requests.labels(
            method=request.method, endpoint=endpoint, status=str(response.status_code)
        ).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics() -> Response:
        """Expose the metrics in the Prometheus text format."""
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health")
    def health(request: Request) -> Health:
        """Report that the API is up and which model version it serves."""
        return Health(status="ok", model_version=request.app.state.model_version)

    # Plain "def" endpoints run in a thread pool: the CPU-bound prediction does
    # not block the event loop
    @app.post("/predict")
    def predict(patient: PatientFeatures, request: Request) -> Prediction:
        """Predict whether a patient is diabetic, with the predicted probability."""
        model = request.app.state.model
        features = patient.to_frame()[list(model.feature_names_in_)]
        probability = float(model.predict_proba(features)[0, 1])
        diabetic = bool(model.predict(features)[0] == 1)

        metrics.predictions.labels(diabetic=str(diabetic).lower()).inc()
        metrics.probability.observe(probability)
        for field, value in patient.model_dump().items():
            if value is None:
                metrics.missing_inputs.labels(feature=field).inc()

        return Prediction(
            diabetic=diabetic,
            probability=round(probability, 4),
            model_version=request.app.state.model_version,
        )

    return app


app = create_app()

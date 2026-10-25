"""Prediction API: serve the diabetes model over HTTP with FastAPI.

Run locally with ``make serve``, then open http://127.0.0.1:8000/docs.
The model is loaded once at startup; the app does not start without it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.serving.model import ServingSettings, load_model
from src.serving.schemas import FEATURE_COLUMNS, Health, PatientFeatures, Prediction


def create_app(settings: ServingSettings | None = None) -> FastAPI:
    """Build the API. Settings default to the ``MODEL_*`` environment variables."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        resolved = settings or ServingSettings.from_env()
        app.state.model = load_model(resolved.model_path, FEATURE_COLUMNS.values())
        app.state.model_version = resolved.model_version
        yield

    app = FastAPI(
        title="Diabetes Prediction API",
        description="Predict the diabetes risk of a patient from 5 health indicators.",
        version="0.1.0",
        lifespan=lifespan,
    )

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
        return Prediction(
            diabetic=bool(model.predict(features)[0] == 1),
            probability=round(probability, 4),
            model_version=request.app.state.model_version,
        )

    return app


app = create_app()

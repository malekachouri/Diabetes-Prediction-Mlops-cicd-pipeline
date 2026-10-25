"""Integration tests of the prediction API through HTTP requests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.serving.app import create_app
from src.serving.model import ServingSettings
from src.serving.schemas import EXAMPLE_PATIENT

pytestmark = pytest.mark.integration


@pytest.fixture
def client(model_file: Path) -> Iterator[TestClient]:
    app = create_app(ServingSettings(model_path=model_file, model_version="test"))
    with TestClient(app) as test_client:  # "with" runs the startup: the model is loaded
        yield test_client


def test_health_reports_the_model_version(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test"}


def test_predict_returns_a_prediction(client: TestClient) -> None:
    response = client.post("/predict", json=EXAMPLE_PATIENT)

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"diabetic", "probability", "model_version"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["model_version"] == "test"


@pytest.mark.parametrize(("glucose", "diabetic"), [(190, True), (90, False)])
def test_predict_uses_the_glucose_of_the_request(
    client: TestClient, glucose: float, diabetic: bool
) -> None:
    response = client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": glucose})

    assert response.json()["diabetic"] is diabetic


def test_predict_accepts_unknown_measurements(client: TestClient) -> None:
    patient = {**EXAMPLE_PATIENT, "glucose": None, "blood_pressure": None, "bmi": None}

    response = client.post("/predict", json=patient)

    assert response.status_code == 200


def test_predict_rejects_an_invalid_patient_with_422(client: TestClient) -> None:
    response = client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": 0})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "glucose"]


def test_predict_rejects_a_missing_required_field(client: TestClient) -> None:
    patient = {key: value for key, value in EXAMPLE_PATIENT.items() if key != "age"}

    response = client.post("/predict", json=patient)

    assert response.status_code == 422


def test_app_does_not_start_without_a_model(tmp_path: Path) -> None:
    app = create_app(ServingSettings(model_path=tmp_path / "missing.joblib", model_version="x"))

    with pytest.raises(FileNotFoundError), TestClient(app):
        pass

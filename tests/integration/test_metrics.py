"""Integration tests of the Prometheus metrics exposed on /metrics."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families

from src.serving.app import create_app
from src.serving.model import ServingSettings
from src.serving.schemas import EXAMPLE_PATIENT

pytestmark = pytest.mark.integration


@pytest.fixture
def client(model_file: Path) -> Iterator[TestClient]:
    app = create_app(ServingSettings(model_path=model_file, model_version="test"))
    with TestClient(app) as test_client:
        yield test_client


def sample(client: TestClient, name: str, **labels: str) -> float:
    """Return the value of one sample read from /metrics (0 if absent)."""
    text = client.get("/metrics").text
    for family in text_string_to_metric_families(text):
        for metric_sample in family.samples:
            if metric_sample.name == name and metric_sample.labels == labels:
                return float(metric_sample.value)
    return 0.0


def test_metrics_are_exposed_in_the_prometheus_format(client: TestClient) -> None:
    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_model_version_is_exposed(client: TestClient) -> None:
    assert sample(client, "diabetes_api_model_info", version="test") == 1.0


def test_predictions_are_counted_by_class(client: TestClient) -> None:
    client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": 190})
    client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": 190})
    client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": 90})

    assert sample(client, "diabetes_api_predictions_total", diabetic="true") == 2.0
    assert sample(client, "diabetes_api_predictions_total", diabetic="false") == 1.0
    assert sample(client, "diabetes_api_prediction_probability_count") == 3.0


def test_unknown_measurements_are_counted(client: TestClient) -> None:
    client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": None, "bmi": None})

    assert sample(client, "diabetes_api_missing_inputs_total", feature="glucose") == 1.0
    assert sample(client, "diabetes_api_missing_inputs_total", feature="bmi") == 1.0
    assert sample(client, "diabetes_api_missing_inputs_total", feature="age") == 0.0


def test_requests_are_counted_by_endpoint_and_status(client: TestClient) -> None:
    client.post("/predict", json=EXAMPLE_PATIENT)
    client.post("/predict", json={**EXAMPLE_PATIENT, "glucose": 0})

    def requests(status: str) -> float:
        return sample(
            client, "diabetes_api_requests_total", method="POST", endpoint="/predict", status=status
        )

    assert requests("200") == 1.0
    assert requests("422") == 1.0
    assert sample(client, "diabetes_api_request_duration_seconds_count", endpoint="/predict") == 2.0


def test_unknown_paths_share_one_label(client: TestClient) -> None:
    client.get("/does-not-exist")
    client.get("/another/unknown/path")

    assert (
        sample(client, "diabetes_api_requests_total", method="GET", endpoint="other", status="404")
        == 2.0
    )


def test_each_app_has_its_own_metrics(client: TestClient, model_file: Path) -> None:
    client.post("/predict", json=EXAMPLE_PATIENT)
    other_app = create_app(ServingSettings(model_path=model_file, model_version="other"))

    with TestClient(other_app) as other_client:
        assert sample(other_client, "diabetes_api_prediction_probability_count") == 0.0

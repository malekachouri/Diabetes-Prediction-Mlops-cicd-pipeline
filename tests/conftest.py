"""Shared pytest fixtures."""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_mlflow_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never log to a real MLflow server from the tests.

    A developer may have loaded ``.env`` in their shell: without this fixture,
    the tests would create runs on DagsHub.
    """
    for name in (
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(name, raising=False)

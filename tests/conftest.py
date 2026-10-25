"""Shared pytest fixtures."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import TrainConfig
from src.models.train import build_pipeline


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


@pytest.fixture(scope="session")
def model_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small model trained on synthetic data, saved like the real one.

    Diabetes depends only on glucose here, so the expected prediction of a
    patient is known and the tests do not need the real data.
    """
    rng = np.random.default_rng(0)
    size = 200
    features = pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, size),
            "Glucose": rng.uniform(70, 200, size),
            "BloodPressure": rng.uniform(50, 100, size),
            "BMI": rng.uniform(18, 45, size),
            "Age": rng.integers(21, 70, size),
        }
    )
    target = (features["Glucose"] > 140).astype(int)
    config = TrainConfig(
        model_path=Path("unused.joblib"), n_estimators=20, max_depth=3, random_state=0
    )
    model = build_pipeline(config).fit(features, target)

    path = tmp_path_factory.mktemp("models") / "model.joblib"
    joblib.dump(model, path)
    return path

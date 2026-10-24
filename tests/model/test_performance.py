"""Minimum performance of the trained model (quality gate)."""

import json

import pytest

from src.config import Config

# Below these values the model is not good enough for screening.
MIN_METRICS = {
    "recall": 0.75,  # at most 1 diabetic patient missed out of 4
    "cv_recall_mean": 0.70,
    "precision": 0.50,
    "roc_auc": 0.75,
}


@pytest.fixture(scope="module")
def metrics(config: Config) -> dict[str, float]:
    path = config.evaluate.metrics_path
    if not path.exists():
        pytest.skip("metrics not available: run `make repro`")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(("name", "minimum"), MIN_METRICS.items())
def test_metric_meets_the_minimum(metrics: dict[str, float], name: str, minimum: float) -> None:
    assert metrics[name] >= minimum, f"{name}={metrics[name]} is below the minimum {minimum}"


def test_cross_validation_is_stable(metrics: dict[str, float]) -> None:
    assert metrics["cv_recall_std"] <= 0.10, "recall varies too much between folds"

"""Unit tests for the evaluation stage."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import Config, DataConfig, EvaluateConfig, TrainConfig
from src.data.prepare import TEST_FILENAME, TRAIN_FILENAME
from src.models.evaluate import compute_metrics, evaluate, main
from src.models.train import train

FEATURES = ["Glucose", "BMI", "Age"]
TARGET = "Outcome"
METRIC_KEYS = {
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "true_negatives",
    "false_positives",
    "false_negatives",
    "true_positives",
}


def make_config(tmp_path: Path) -> Config:
    return Config(
        data=DataConfig(
            raw_path=tmp_path / "raw.csv",
            processed_dir=tmp_path / "processed",
            target=TARGET,
            features=FEATURES,
            zero_as_missing=["Glucose", "BMI"],
            test_size=0.25,
            random_state=42,
        ),
        train=TrainConfig(
            model_path=tmp_path / "models" / "model.joblib",
            n_estimators=10,
            max_depth=3,
            random_state=42,
        ),
        evaluate=EvaluateConfig(metrics_path=tmp_path / "reports" / "metrics.json"),
    )


def make_split(n_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    glucose = rng.integers(80, 200, n_rows).astype(float)
    return pd.DataFrame(
        {
            "Glucose": glucose,
            "BMI": rng.uniform(18, 45, n_rows).round(1),
            "Age": rng.integers(21, 80, n_rows),
            TARGET: (glucose > 140).astype(int),  # learnable signal
        }
    )


def prepare_trained_model(config: Config) -> None:
    config.data.processed_dir.mkdir(parents=True)
    make_split(80, seed=0).to_csv(config.data.processed_dir / TRAIN_FILENAME, index=False)
    make_split(20, seed=1).to_csv(config.data.processed_dir / TEST_FILENAME, index=False)
    train(config)


def test_compute_metrics_perfect_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, y_true, np.array([0.1, 0.2, 0.8, 0.9]))

    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert metrics[name] == 1.0
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0


def test_compute_metrics_known_values() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])  # 1 false positive, 1 false negative
    y_proba = np.array([0.1, 0.6, 0.2, 0.9, 0.4, 0.7])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 2
    assert metrics["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["recall"] == pytest.approx(2 / 3, abs=1e-4)
    assert metrics["roc_auc"] == pytest.approx(8 / 9, abs=1e-4)


def test_compute_metrics_never_predicting_positive() -> None:
    y_true = np.array([0, 0, 1, 1])
    metrics = compute_metrics(y_true, np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics["accuracy"] == 0.5
    assert metrics["recall"] == 0.0  # the model misses every diabetic patient
    assert metrics["precision"] == 0.0  # zero_division=0 instead of a warning


def test_evaluate_writes_metrics_json(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    prepare_trained_model(config)

    metrics = evaluate(config)

    saved = json.loads(config.evaluate.metrics_path.read_text(encoding="utf-8"))
    assert saved == metrics
    assert set(saved) == METRIC_KEYS
    confusion = ("true_negatives", "false_positives", "false_negatives", "true_positives")
    assert sum(saved[name] for name in confusion) == 20  # every test row, and only test rows
    for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert 0.0 <= saved[name] <= 1.0


def test_main_reads_params_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    prepare_trained_model(config)
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": str(config.data.raw_path),
                    "processed_dir": str(config.data.processed_dir),
                    "target": TARGET,
                    "features": FEATURES,
                    "zero_as_missing": config.data.zero_as_missing,
                    "test_size": config.data.test_size,
                    "random_state": config.data.random_state,
                },
                "train": {
                    "model_path": str(config.train.model_path),
                    "n_estimators": config.train.n_estimators,
                    "max_depth": config.train.max_depth,
                    "random_state": config.train.random_state,
                },
                "evaluate": {"metrics_path": str(config.evaluate.metrics_path)},
            }
        ),
        encoding="utf-8",
    )

    main(params_path)

    assert config.evaluate.metrics_path.exists()


def test_evaluate_logs_to_mlflow_only_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = make_config(tmp_path)
    prepare_trained_model(config)
    logged: list[dict] = []

    def fake_log_run(config: Config, metrics: dict, *args: object) -> str:
        logged.append(metrics)
        return "fake-run-id"

    monkeypatch.setattr("src.models.evaluate.log_run", fake_log_run)

    evaluate(config)
    assert logged == []  # no MLFLOW_TRACKING_URI: nothing is logged

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.invalid")
    metrics = evaluate(config)
    assert logged == [metrics]

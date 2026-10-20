"""Unit tests for the MLflow tracking helpers."""

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from src.config import Config, DataConfig, EvaluateConfig, TrainConfig
from src.models.train import build_pipeline
from src.tracking import (
    MODEL_ARTIFACT_PATH,
    flatten_params,
    is_tracking_enabled,
    log_run,
    read_dvc_md5,
)

RAW_MD5 = "f2906818eda8fcfc8f8416557ab1e6df"
METRICS = {"recall": 0.6, "roc_auc": 0.8, "true_positives": 3}


def make_config(tmp_path: Path) -> Config:
    raw_path = tmp_path / "raw.csv"
    raw_path.with_name("raw.csv.dvc").write_text(
        f"outs:\n- md5: {RAW_MD5}\n  size: 10\n  hash: md5\n  path: raw.csv\n",
        encoding="utf-8",
    )
    return Config(
        data=DataConfig(
            raw_path=raw_path,
            processed_dir=tmp_path / "processed",
            target="Outcome",
            features=["Glucose", "BMI"],
            zero_as_missing=["Glucose"],
            test_size=0.2,
            random_state=42,
        ),
        train=TrainConfig(
            model_path=tmp_path / "model.joblib",
            n_estimators=10,
            max_depth=None,
            random_state=42,
        ),
        evaluate=EvaluateConfig(metrics_path=tmp_path / "metrics.json"),
    )


def make_model(config: Config) -> tuple[Pipeline, pd.DataFrame]:
    """Train the real project pipeline (imputer + random forest) on a tiny dataset."""
    X = pd.DataFrame({"Glucose": [90.0, 150.0, 100.0, 180.0], "BMI": [22.0, 31.0, 25.0, 35.0]})
    model = build_pipeline(config.train)
    model.fit(X, np.array([0, 1, 0, 1]))
    return model, X


def test_tracking_is_disabled_without_tracking_uri() -> None:
    assert not is_tracking_enabled()


def test_tracking_is_enabled_with_tracking_uri(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    assert is_tracking_enabled()


def test_flatten_params_uses_section_prefix(tmp_path: Path) -> None:
    params = flatten_params(make_config(tmp_path))

    assert params["train.n_estimators"] == "10"
    assert params["train.max_depth"] == "None"
    assert params["data.features"] == "['Glucose', 'BMI']"
    assert all(isinstance(value, str) for value in params.values())


def test_read_dvc_md5_reads_the_pointer_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    assert read_dvc_md5(config.data.raw_path) == RAW_MD5


def test_log_run_records_params_metrics_tags_and_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "unit-test")
    mlflow.create_experiment("unit-test", artifact_location=(tmp_path / "artifacts").as_uri())
    config = make_config(tmp_path)
    model, X = make_model(config)

    run_id = log_run(config, METRICS, model, X)

    run = MlflowClient().get_run(run_id)
    assert run.data.params["train.n_estimators"] == "10"
    assert run.data.metrics["recall"] == pytest.approx(0.6)
    assert run.data.metrics["true_positives"] == 3
    assert run.data.tags["data.raw_md5"] == RAW_MD5
    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}")
    np.testing.assert_array_equal(loaded.predict(X), model.predict(X))

"""Unit tests for the model registration and promotion."""

from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src.config import Config, DataConfig, EvaluateConfig, TrainConfig
from src.models.register import (
    CHAMPION_ALIAS,
    REGISTERED_MODEL_NAME,
    build_run_filter,
    find_latest_run,
    main,
    register_and_promote,
)
from src.models.train import build_pipeline
from src.tracking import log_run

RAW_MD5 = "f2906818eda8fcfc8f8416557ab1e6df"


@pytest.fixture
def mlflow_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MlflowClient:
    """A temporary MLflow tracking server and model registry (SQLite)."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "unit-test")
    mlflow.create_experiment("unit-test", artifact_location=(tmp_path / "artifacts").as_uri())
    return MlflowClient()


def make_config(tmp_path: Path, max_depth: int = 3) -> Config:
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
            n_estimators=5,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
        ),
        evaluate=EvaluateConfig(metrics_path=tmp_path / "metrics.json"),
    )


def log_candidate(config: Config, cv_recall: float) -> str:
    X = pd.DataFrame({"Glucose": [90.0, 150.0, 100.0, 180.0], "BMI": [22.0, 31.0, 25.0, 35.0]})
    model = build_pipeline(config.train).fit(X, np.array([0, 1, 0, 1]))
    return log_run(config, {"cv_recall_mean": cv_recall, "recall": 0.8}, model, X)


def champion_version(client: MlflowClient) -> str:
    return str(client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS).version)


def test_build_run_filter_uses_train_params_and_data_hash(tmp_path: Path) -> None:
    run_filter = build_run_filter(make_config(tmp_path))

    assert "params.`train.max_depth` = '3'" in run_filter
    assert "params.`train.class_weight` = 'balanced'" in run_filter
    assert f"tags.`data.raw_md5` = '{RAW_MD5}'" in run_filter
    assert "attributes.status = 'FINISHED'" in run_filter


def test_find_latest_run_selects_the_current_configuration(
    tmp_path: Path, mlflow_store: MlflowClient
) -> None:
    config = make_config(tmp_path, max_depth=3)
    run_id = log_candidate(config, cv_recall=0.7)
    log_candidate(make_config(tmp_path, max_depth=8), cv_recall=0.9)  # other configuration

    assert find_latest_run(mlflow_store, config).info.run_id == run_id


def test_find_latest_run_fails_without_matching_run(
    tmp_path: Path, mlflow_store: MlflowClient
) -> None:
    log_candidate(make_config(tmp_path, max_depth=8), cv_recall=0.9)

    with pytest.raises(LookupError, match="no finished MLflow run"):
        find_latest_run(mlflow_store, make_config(tmp_path, max_depth=3))


def test_find_latest_run_fails_without_experiment(
    tmp_path: Path, mlflow_store: MlflowClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "missing")

    with pytest.raises(LookupError, match="does not exist"):
        find_latest_run(mlflow_store, make_config(tmp_path))


def test_first_model_becomes_champion(tmp_path: Path, mlflow_store: MlflowClient) -> None:
    config = make_config(tmp_path)
    log_candidate(config, cv_recall=0.7)

    version, promoted = register_and_promote(config)

    assert promoted
    assert champion_version(mlflow_store) == version
    model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}@{CHAMPION_ALIAS}")
    assert model.named_steps["classifier"].class_weight == "balanced"


def test_registering_the_same_run_twice_does_not_duplicate_it(
    tmp_path: Path, mlflow_store: MlflowClient
) -> None:
    config = make_config(tmp_path)
    log_candidate(config, cv_recall=0.7)

    first, _ = register_and_promote(config)
    second, promoted = register_and_promote(config)

    assert second == first
    assert not promoted  # equal score: the champion stays in place
    assert len(mlflow_store.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'")) == 1


def test_worse_model_is_registered_but_not_promoted(
    tmp_path: Path, mlflow_store: MlflowClient
) -> None:
    champion_config = make_config(tmp_path, max_depth=3)
    log_candidate(champion_config, cv_recall=0.8)
    champion, _ = register_and_promote(champion_config)

    worse_config = replace(champion_config, train=replace(champion_config.train, max_depth=8))
    log_candidate(worse_config, cv_recall=0.6)
    version, promoted = register_and_promote(worse_config)

    assert not promoted
    assert version != champion
    assert champion_version(mlflow_store) == champion


def test_better_model_replaces_the_champion(tmp_path: Path, mlflow_store: MlflowClient) -> None:
    first_config = make_config(tmp_path, max_depth=3)
    log_candidate(first_config, cv_recall=0.6)
    register_and_promote(first_config)

    better_config = replace(first_config, train=replace(first_config.train, max_depth=8))
    log_candidate(better_config, cv_recall=0.8)
    version, promoted = register_and_promote(better_config)

    assert promoted
    assert champion_version(mlflow_store) == version


def test_main_requires_a_tracking_server() -> None:
    with pytest.raises(SystemExit, match="MLFLOW_TRACKING_URI"):
        main()


def test_main_registers_the_project_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[Config] = []

    def fake_register_and_promote(config: Config) -> tuple[str, bool]:
        registered.append(config)
        return "1", True

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://tracking.invalid")
    monkeypatch.setattr("src.models.register.register_and_promote", fake_register_and_promote)

    main()

    assert registered[0].data.target == "Outcome"  # the real params.yaml was loaded

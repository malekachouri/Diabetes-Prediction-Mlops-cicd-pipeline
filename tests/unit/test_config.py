"""Unit tests for the configuration loader."""

from pathlib import Path

import pytest
import yaml

from src.config import load_config

VALID_PARAMS = {
    "data": {
        "raw_path": "data/raw/diabetes.csv",
        "processed_dir": "data/processed",
        "target": "Outcome",
        "features": ["Glucose", "BMI", "Age"],
        "zero_as_missing": ["Glucose", "BMI"],
        "test_size": 0.2,
        "random_state": 42,
    },
    "train": {
        "model_path": "models/model.joblib",
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42,
    },
    "evaluate": {"metrics_path": "reports/metrics.json"},
}


def write_params(tmp_path: Path, params: dict) -> Path:
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(params), encoding="utf-8")
    return path


def test_project_params_file_is_valid() -> None:
    config = load_config("params.yaml")
    assert config.data.target == "Outcome"
    assert config.data.target not in config.data.features


def test_load_config_returns_typed_values(tmp_path: Path) -> None:
    config = load_config(write_params(tmp_path, VALID_PARAMS))
    assert config.data.raw_path == Path("data/raw/diabetes.csv")
    assert config.data.test_size == pytest.approx(0.2)
    assert config.train.n_estimators == 10
    assert config.evaluate.metrics_path == Path("reports/metrics.json")


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("data", "test_size", 1.5, "test_size"),
        ("data", "features", [], "features"),
        ("data", "features", ["Outcome", "BMI"], "target"),
        ("data", "zero_as_missing", ["Insulin"], "zero_as_missing"),
        ("train", "n_estimators", 0, "n_estimators"),
        ("train", "max_depth", 0, "max_depth"),
    ],
)
def test_invalid_params_are_rejected(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    params = {name: dict(values) for name, values in VALID_PARAMS.items()}
    params[section][key] = value
    with pytest.raises(ValueError, match=message):
        load_config(write_params(tmp_path, params))

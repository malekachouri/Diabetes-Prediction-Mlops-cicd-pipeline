"""Unit tests of the serving code: settings, model loading and request schema."""

from pathlib import Path

import joblib
import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.dummy import DummyClassifier

from src.serving.model import DEFAULT_MODEL_PATH, ServingSettings, load_model
from src.serving.schemas import EXAMPLE_PATIENT, FEATURE_COLUMNS, PatientFeatures


# ---------- Settings ----------
def test_settings_default_to_the_local_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)

    settings = ServingSettings.from_env()

    assert settings == ServingSettings(model_path=DEFAULT_MODEL_PATH, model_version="dev")


def test_settings_are_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_PATH", "/app/model.joblib")
    monkeypatch.setenv("MODEL_VERSION", "3")

    settings = ServingSettings.from_env()

    assert settings == ServingSettings(model_path=Path("/app/model.joblib"), model_version="3")


# ---------- Model loading ----------
def test_load_model_returns_the_pipeline(model_file: Path) -> None:
    model = load_model(model_file, FEATURE_COLUMNS.values())

    assert set(model.feature_names_in_) == set(FEATURE_COLUMNS.values())


def test_load_model_fails_when_the_file_is_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="model file not found"):
        load_model(tmp_path / "missing.joblib", FEATURE_COLUMNS.values())


def test_load_model_rejects_an_object_that_is_not_a_pipeline(tmp_path: Path) -> None:
    path = tmp_path / "model.joblib"
    joblib.dump(DummyClassifier(), path)

    with pytest.raises(TypeError, match="expected a scikit-learn Pipeline"):
        load_model(path, FEATURE_COLUMNS.values())


def test_load_model_rejects_a_model_trained_on_other_features(model_file: Path) -> None:
    with pytest.raises(ValueError, match="do not match the API features"):
        load_model(model_file, ["Glucose", "BMI"])


# ---------- Request schema ----------
def test_patient_is_converted_to_the_training_columns() -> None:
    frame = PatientFeatures.model_validate(EXAMPLE_PATIENT).to_frame()

    assert list(frame.columns) == list(FEATURE_COLUMNS.values())
    assert frame.loc[0, "Glucose"] == 148
    assert frame.loc[0, "BloodPressure"] == 72


def test_unknown_measurements_become_missing_values() -> None:
    patient = PatientFeatures(pregnancies=1, age=30)

    frame = patient.to_frame()

    assert np.isnan(frame.loc[0, ["Glucose", "BloodPressure", "BMI"]].astype(float)).all()


@pytest.mark.parametrize(
    "change",
    [
        {"glucose": 0},  # impossible zero: unknown values must be sent as null
        {"age": 10},  # below the plausible bounds
        {"bmi": 120},  # above the plausible bounds
        {"weight": 70},  # unknown field
    ],
)
def test_implausible_or_unknown_values_are_rejected(change: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        PatientFeatures.model_validate({**EXAMPLE_PATIENT, **change})

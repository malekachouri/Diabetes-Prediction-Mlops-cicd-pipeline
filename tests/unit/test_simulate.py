"""Unit tests of the traffic simulator (no API needed)."""

import json

import numpy as np
import pandas as pd
import pytest

from src.data.bounds import COLUMN_BOUNDS
from src.monitoring import simulate
from src.monitoring.simulate import DRIFT_SHIFT, sample_patients

TEST_SET = pd.DataFrame(
    {
        "Pregnancies": [1, 3, 0],
        "Glucose": [100.0, np.nan, 390.0],
        "BloodPressure": [70.0, 80.0, 60.0],
        "BMI": [25.0, 30.0, 40.0],
        "Age": [30, 50, 115],
        "Outcome": [0, 1, 1],
    }
)


def test_patients_use_the_api_fields() -> None:
    patients = sample_patients(TEST_SET, size=20, drift=False)

    assert len(patients) == 20
    assert set(patients[0]) == {"pregnancies", "glucose", "blood_pressure", "bmi", "age"}


def test_missing_values_are_sent_as_null() -> None:
    patients = sample_patients(TEST_SET, size=50, drift=False)

    assert any(patient["glucose"] is None for patient in patients)


def test_drift_shifts_the_population_within_the_bounds() -> None:
    normal = pd.DataFrame(sample_patients(TEST_SET, size=200, drift=False, seed=1))
    drifted = pd.DataFrame(sample_patients(TEST_SET, size=200, drift=True, seed=1))

    assert drifted["age"].mean() > normal["age"].mean() + DRIFT_SHIFT["Age"] / 2
    assert drifted["age"].max() <= COLUMN_BOUNDS["Age"][1]
    assert drifted["glucose"].max() <= COLUMN_BOUNDS["Glucose"][1]


def test_send_posts_the_patient_as_json(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def fake_urlopen(request, timeout: float) -> FakeResponse:  # noqa: ANN001
        sent.append((request.full_url, json.loads(request.data)))
        return FakeResponse()

    monkeypatch.setattr(simulate.urllib.request, "urlopen", fake_urlopen)

    simulate.send("http://api/", {"age": 50})

    assert sent == [("http://api/predict", {"age": 50})]


def test_main_sends_the_requested_number_of_patients(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = []
    monkeypatch.setattr(simulate.pd, "read_csv", lambda _: TEST_SET)
    monkeypatch.setattr(simulate, "send", lambda url, patient: sent.append(url))

    simulate.main(["--url", "http://api", "--size", "7", "--drift"])

    assert sent == ["http://api"] * 7

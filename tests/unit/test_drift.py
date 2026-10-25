"""Unit tests of the drift detection (synthetic data, no real data needed)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.monitoring import drift
from src.monitoring.drift import DriftResult, detect_drift, load_current

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]


def population(
    size: int, seed: int, glucose_shift: float = 0.0, age_shift: float = 0.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, size).astype(float),
            "Glucose": rng.normal(120 + glucose_shift, 30, size),
            "BloodPressure": rng.normal(70, 12, size),
            "BMI": rng.normal(32, 7, size),
            "Age": rng.integers(21, 70, size) + age_shift,
        }
    )


def write_predictions(path: Path, frame: pd.DataFrame) -> None:
    lines = [json.dumps({"event": "other"})]  # non-prediction lines are ignored
    for row in frame.to_dict(orient="records"):
        lines.append(json.dumps({"event": "prediction", "features": row}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------- Drift detection ----------
def test_same_population_has_no_drift() -> None:
    result = detect_drift(population(600, seed=0), population(200, seed=1))

    assert result.dataset_drift is False
    assert result.drifted_features == []


def test_shifted_population_is_detected() -> None:
    current = population(200, seed=1, glucose_shift=40, age_shift=15)

    result = detect_drift(population(600, seed=0), current)

    assert {"Glucose", "Age"} <= set(result.drifted_features)
    assert result.drifted_share >= 0.4


def test_dataset_drift_needs_half_of_the_features() -> None:
    current = population(200, seed=1, glucose_shift=40)  # 1 feature out of 5

    result = detect_drift(population(600, seed=0), current)

    assert result.drifted_features == ["Glucose"]
    assert result.dataset_drift is False


def test_html_report_is_written(tmp_path: Path) -> None:
    detect_drift(population(600, seed=0), population(100, seed=1), tmp_path / "report.html")

    assert (tmp_path / "report.html").stat().st_size > 0


# ---------- Production data ----------
def test_load_current_reads_the_logged_predictions(tmp_path: Path) -> None:
    frame = population(60, seed=2)
    frame.loc[0, "Glucose"] = None
    write_predictions(tmp_path / "predictions.jsonl", frame)

    current = load_current(tmp_path / "predictions.jsonl", FEATURES)

    assert list(current.columns) == FEATURES
    assert len(current) == 60
    assert np.isnan(current.loc[0, "Glucose"])


def test_load_current_needs_enough_predictions(tmp_path: Path) -> None:
    write_predictions(tmp_path / "predictions.jsonl", population(10, seed=2))

    with pytest.raises(ValueError, match="at least 50"):
        load_current(tmp_path / "predictions.jsonl", FEATURES)


def test_load_current_fails_without_production_data(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="make drift-collect"):
        load_current(tmp_path / "missing.jsonl", FEATURES)


# ---------- End to end ----------
def test_run_writes_the_report_and_the_summary(tmp_path: Path) -> None:
    params = yaml.safe_load(Path("params.yaml").read_text(encoding="utf-8"))
    params["data"]["processed_dir"] = str(tmp_path)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")
    population(300, seed=0).assign(Outcome=0).to_csv(tmp_path / "train.csv", index=False)
    write_predictions(tmp_path / "predictions.jsonl", population(100, seed=1, glucose_shift=40))

    result = drift.run(tmp_path / "params.yaml", tmp_path / "predictions.jsonl", tmp_path / "out")

    summary = json.loads((tmp_path / "out" / "drift_summary.json").read_text(encoding="utf-8"))
    assert summary["drifted_features"] == result.drifted_features == ["Glucose"]
    assert (tmp_path / "out" / "drift_report.html").exists()


@pytest.mark.parametrize(
    ("dataset_drift", "flag", "code"), [(True, True, 1), (True, False, 0), (False, True, 0)]
)
def test_main_exit_code(
    monkeypatch: pytest.MonkeyPatch, dataset_drift: bool, flag: bool, code: int
) -> None:
    result = DriftResult(dataset_drift, ["Glucose"], 0.2, 600, 100)
    monkeypatch.setattr(drift, "run", lambda **_: result)

    assert drift.main(["--fail-on-drift"] if flag else []) == code

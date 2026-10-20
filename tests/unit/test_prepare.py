"""Unit tests for the data preparation stage."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import DataConfig
from src.data.prepare import TEST_FILENAME, TRAIN_FILENAME, clean, load_raw, main, run, split

FEATURES = ["Pregnancies", "Glucose", "BMI"]
TARGET = "Outcome"


def make_config(tmp_path: Path, test_size: float = 0.25) -> DataConfig:
    return DataConfig(
        raw_path=tmp_path / "raw.csv",
        processed_dir=tmp_path / "processed",
        target=TARGET,
        features=FEATURES,
        zero_as_missing=["Glucose", "BMI"],
        test_size=test_size,
        random_state=42,
    )


def make_raw(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n_rows),
            "Glucose": rng.integers(80, 200, n_rows),
            "BMI": rng.uniform(18, 45, n_rows).round(1),
            "Insulin": rng.integers(0, 300, n_rows),
            TARGET: [0, 1] * (n_rows // 2),
        }
    )


def test_load_raw_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "raw.csv"
    make_raw().drop(columns=["BMI"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="BMI"):
        load_raw(path, [*FEATURES, TARGET])


def test_clean_keeps_only_model_columns(tmp_path: Path) -> None:
    cleaned = clean(make_raw(), make_config(tmp_path))
    assert list(cleaned.columns) == [*FEATURES, TARGET]


def test_clean_replaces_impossible_zeros_only(tmp_path: Path) -> None:
    raw = make_raw()
    raw.loc[0, ["Pregnancies", "Glucose", "BMI"]] = 0

    cleaned = clean(raw, make_config(tmp_path))

    assert cleaned.loc[0, "Pregnancies"] == 0  # zero pregnancies is a valid value
    assert np.isnan(cleaned.loc[0, "Glucose"])
    assert np.isnan(cleaned.loc[0, "BMI"])
    assert raw.loc[0, "Glucose"] == 0  # the input DataFrame is not modified


def test_split_is_stratified_disjoint_and_reproducible(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    df = clean(make_raw(), config)

    train, test = split(df, config)
    train_again, test_again = split(df, config)

    assert len(test) == 10
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(test.index)
    assert test[TARGET].mean() == pytest.approx(df[TARGET].mean())
    pd.testing.assert_frame_equal(test, test_again)
    pd.testing.assert_frame_equal(train, train_again)


def test_run_writes_train_and_test_files(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    make_raw().to_csv(config.raw_path, index=False)

    train_path, test_path = run(config)

    assert train_path == config.processed_dir / TRAIN_FILENAME
    assert test_path == config.processed_dir / TEST_FILENAME
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    assert len(train) + len(test) == 40
    assert list(train.columns) == [*FEATURES, TARGET]


def test_main_reads_params_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    make_raw().to_csv(config.raw_path, index=False)
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": str(config.raw_path),
                    "processed_dir": str(config.processed_dir),
                    "target": TARGET,
                    "features": FEATURES,
                    "zero_as_missing": config.zero_as_missing,
                    "test_size": config.test_size,
                    "random_state": config.random_state,
                },
                "train": {
                    "model_path": "models/model.joblib",
                    "n_estimators": 10,
                    "max_depth": 3,
                    "random_state": 42,
                },
                "evaluate": {"metrics_path": "reports/metrics.json"},
            }
        ),
        encoding="utf-8",
    )

    main(params_path)

    assert (config.processed_dir / TRAIN_FILENAME).exists()
    assert (config.processed_dir / TEST_FILENAME).exists()

"""Unit tests for the data validation schemas."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.config import DataConfig
from src.data.schemas import MIN_ROWS, prepared_schema, raw_schema, validate

FEATURES = ["Pregnancies", "Glucose", "BMI", "Age"]
TARGET = "Outcome"


def make_config() -> DataConfig:
    return DataConfig(
        raw_path=Path("raw.csv"),
        processed_dir=Path("processed"),
        target=TARGET,
        features=FEATURES,
        zero_as_missing=["Glucose", "BMI"],
        test_size=0.2,
        random_state=42,
    )


def make_valid_raw(n_rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "Pregnancies": rng.integers(0, 10, n_rows),
            "Glucose": rng.integers(60, 200, n_rows),
            "BMI": rng.uniform(18, 45, n_rows).round(1),
            "Age": rng.integers(21, 80, n_rows),
            TARGET: [0, 1] * (n_rows // 2),
        }
    )


def failing_checks(error: SchemaErrors) -> str:
    return error.failure_cases[["column", "check"]].astype(str).to_string()


def test_valid_raw_data_passes() -> None:
    validate(make_valid_raw(), raw_schema(make_config()))


def test_raw_data_accepts_zero_as_missing_value() -> None:
    raw = make_valid_raw()
    raw.loc[0, ["Glucose", "BMI"]] = 0

    validate(raw, raw_schema(make_config()))


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("Glucose", 25),  # neither 0 nor a plausible glucose level
        ("BMI", 95.0),
        ("Age", 12),  # the dataset only contains adults
        ("Pregnancies", -1),
    ],
)
def test_raw_data_rejects_implausible_values(column: str, value: float) -> None:
    raw = make_valid_raw()
    raw.loc[0, column] = value

    with pytest.raises(SchemaErrors) as error:
        validate(raw, raw_schema(make_config()))

    assert column in failing_checks(error.value)


def test_raw_data_rejects_unknown_target_values() -> None:
    raw = make_valid_raw()
    raw.loc[0, TARGET] = 2

    with pytest.raises(SchemaErrors):
        validate(raw, raw_schema(make_config()))


def test_raw_data_rejects_missing_values() -> None:
    raw = make_valid_raw().astype({"Age": float})
    raw.loc[0, "Age"] = np.nan

    with pytest.raises(SchemaErrors):
        validate(raw, raw_schema(make_config()))


def test_raw_data_rejects_too_few_rows() -> None:
    with pytest.raises(SchemaErrors) as error:
        validate(make_valid_raw(n_rows=MIN_ROWS - 2), raw_schema(make_config()))

    assert "fewer than" in failing_checks(error.value)


def test_raw_data_rejects_a_drifted_class_balance() -> None:
    raw = make_valid_raw()
    raw[TARGET] = 0
    raw.loc[:2, TARGET] = 1  # 3 diabetic patients out of 40

    with pytest.raises(SchemaErrors) as error:
        validate(raw, raw_schema(make_config()))

    assert "share of positive" in failing_checks(error.value)


def test_errors_are_all_reported_at_once() -> None:
    raw = make_valid_raw()
    raw.loc[0, "BMI"] = 95.0
    raw.loc[1, "Age"] = 12

    with pytest.raises(SchemaErrors) as error:
        validate(raw, raw_schema(make_config()))

    report = failing_checks(error.value)
    assert "BMI" in report
    assert "Age" in report


def test_prepared_data_accepts_missing_values_in_zero_as_missing_columns() -> None:
    prepared = make_valid_raw().astype({"Glucose": float})
    prepared.loc[0, "Glucose"] = np.nan

    validate(prepared, prepared_schema(make_config()))


def test_prepared_data_rejects_remaining_zeros() -> None:
    prepared = make_valid_raw()
    prepared.loc[0, "Glucose"] = 0  # should have been replaced by NaN

    with pytest.raises(SchemaErrors):
        validate(prepared, prepared_schema(make_config()))


def test_prepared_data_rejects_unexpected_columns() -> None:
    prepared = make_valid_raw()
    prepared["Insulin"] = 100

    with pytest.raises(SchemaErrors):
        validate(prepared, prepared_schema(make_config()))


def test_schema_requires_bounds_for_every_feature() -> None:
    config = replace(make_config(), features=[*FEATURES, "HeartRate"])

    with pytest.raises(KeyError, match="HeartRate"):
        raw_schema(config)

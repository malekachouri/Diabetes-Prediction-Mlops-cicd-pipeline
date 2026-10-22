"""Data validation schemas (pandera) for the raw and the prepared datasets.

The rules come from the dataset card (docs/data/diabetes-dataset.md): each
column must stay within physiologically plausible bounds, and the impossible
zeros of the raw data must be flagged as missing values after preparation.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from src.config import DataConfig

# Plausible (min, max) bounds of each column, from domain knowledge.
COLUMN_BOUNDS: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 400),
    "BloodPressure": (20, 200),
    "SkinThickness": (5, 100),
    "Insulin": (10, 1000),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (18, 120),
}
# A dataset below this size cannot give a meaningful stratified split.
MIN_ROWS = 20
# Expected share of diabetic patients; outside this range the data has drifted.
POSITIVE_SHARE_RANGE = (0.2, 0.6)


def _bounds(column: str) -> tuple[float, float]:
    if column not in COLUMN_BOUNDS:
        raise KeyError(f"no plausible bounds defined for column '{column}'")
    return COLUMN_BOUNDS[column]


def _target_column(config: DataConfig) -> pa.Column:
    return pa.Column(int, pa.Check.isin([0, 1]), nullable=False, name=config.target)


def _positive_share_check(config: DataConfig) -> pa.Check:
    low, high = POSITIVE_SHARE_RANGE
    return pa.Check(
        lambda df: low <= df[config.target].mean() <= high,
        error=f"share of positive '{config.target}' outside {POSITIVE_SHARE_RANGE}",
    )


def raw_schema(config: DataConfig) -> pa.DataFrameSchema:
    """Schema of the raw data: no missing value, 0 allowed where it encodes 'missing'."""
    columns: dict[str, pa.Column] = {}
    for name in config.features:
        low, high = _bounds(name)
        if name in config.zero_as_missing:
            check = pa.Check(
                lambda s, low=low, high=high: (s == 0) | s.between(low, high),
                error=f"must be 0 (missing) or within [{low}, {high}]",
            )
        else:
            check = pa.Check.in_range(low, high)
        columns[name] = pa.Column(float, check, nullable=False, coerce=True)
    columns[config.target] = _target_column(config)
    size_check = pa.Check(lambda df: len(df) >= MIN_ROWS, error=f"fewer than {MIN_ROWS} rows")
    return pa.DataFrameSchema(columns, checks=[size_check, _positive_share_check(config)])


def prepared_schema(config: DataConfig) -> pa.DataFrameSchema:
    """Schema of the prepared data: exact columns, impossible zeros replaced by NaN."""
    columns: dict[str, pa.Column] = {}
    for name in config.features:
        low, high = _bounds(name)
        columns[name] = pa.Column(
            float,
            pa.Check.in_range(low, high),
            nullable=name in config.zero_as_missing,
            coerce=True,
        )
    columns[config.target] = _target_column(config)
    return pa.DataFrameSchema(
        columns, checks=[_positive_share_check(config)], strict=True, ordered=True
    )


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate a DataFrame and report every failing check at once (lazy mode)."""
    return schema.validate(df, lazy=True)

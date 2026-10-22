"""Prepare the raw dataset: select columns, flag missing values, split train/test.

Run as a pipeline stage with ``python -m src.data.prepare``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DEFAULT_PARAMS_PATH, DataConfig, load_config
from src.data.schemas import prepared_schema, raw_schema, validate

logger = logging.getLogger(__name__)

TRAIN_FILENAME = "train.csv"
TEST_FILENAME = "test.csv"


def load_raw(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Load the raw CSV and check that every required column is present."""
    df = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"raw data is missing columns: {missing}")
    return df


def clean(df: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Keep the model columns and replace impossible zeros with NaN.

    Missing values are only flagged here, not imputed: imputation is learned
    on the training set inside the model pipeline to avoid data leakage.
    """
    cleaned = df[[*config.features, config.target]].copy()
    cleaned[config.zero_as_missing] = cleaned[config.zero_as_missing].replace(0, np.nan)
    return cleaned


def split(df: pd.DataFrame, config: DataConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test sets, stratified on the target to keep class balance."""
    train, test = train_test_split(
        df,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=df[config.target],
    )
    return train, test


def run(config: DataConfig) -> tuple[Path, Path]:
    """Execute the full preparation stage and return the written file paths."""
    raw = validate(load_raw(config.raw_path, [*config.features, config.target]), raw_schema(config))
    train, test = split(clean(raw, config), config)
    schema = prepared_schema(config)
    train, test = validate(train, schema), validate(test, schema)

    config.processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = config.processed_dir / TRAIN_FILENAME
    test_path = config.processed_dir / TEST_FILENAME
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logger.info("Raw rows: %d -> train: %d, test: %d", len(raw), len(train), len(test))
    logger.info("Missing values in train:\n%s", train.isna().sum().to_string())
    return train_path, test_path


def main(params_path: Path = DEFAULT_PARAMS_PATH) -> None:
    """Entry point of the ``prepare`` pipeline stage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run(load_config(params_path).data)


if __name__ == "__main__":
    main()

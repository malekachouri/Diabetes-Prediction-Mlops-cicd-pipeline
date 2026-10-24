"""Fixtures for the model tests: the trained model and the held-out test set.

These tests need the pipeline outputs (`make data-pull` then `make repro`);
they are skipped when the outputs are not available.
"""

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import Config, load_config
from src.data.prepare import TEST_FILENAME


@pytest.fixture(scope="session")
def config() -> Config:
    return load_config()


@pytest.fixture(scope="session")
def model(config: Config) -> Pipeline:
    if not config.train.model_path.exists():
        pytest.skip("trained model not available: run `make data-pull` and `make repro`")
    return joblib.load(config.train.model_path)


@pytest.fixture(scope="session")
def test_features(config: Config) -> pd.DataFrame:
    path = config.data.processed_dir / TEST_FILENAME
    if not path.exists():
        pytest.skip("test set not available: run `make data-pull` and `make repro`")
    return pd.read_csv(path)[config.data.features]

"""Validation of the real versioned dataset (requires `make data-pull`)."""

import pandas as pd
import pytest

from src.config import load_config
from src.data.prepare import TEST_FILENAME, TRAIN_FILENAME, clean
from src.data.schemas import prepared_schema, raw_schema, validate

CONFIG = load_config().data

pytestmark = pytest.mark.skipif(
    not CONFIG.raw_path.exists(), reason="real data not available: run `make data-pull`"
)


def test_real_raw_data_matches_the_raw_schema() -> None:
    validate(pd.read_csv(CONFIG.raw_path), raw_schema(CONFIG))


def test_real_data_after_cleaning_matches_the_prepared_schema() -> None:
    validate(clean(pd.read_csv(CONFIG.raw_path), CONFIG), prepared_schema(CONFIG))


@pytest.mark.parametrize("filename", [TRAIN_FILENAME, TEST_FILENAME])
def test_prepared_files_match_the_prepared_schema(filename: str) -> None:
    path = CONFIG.processed_dir / filename
    if not path.exists():
        pytest.skip("prepared data not available: run `make repro`")
    validate(pd.read_csv(path), prepared_schema(CONFIG))

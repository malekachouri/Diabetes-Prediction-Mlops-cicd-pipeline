"""Send realistic prediction requests to the API, with or without drift.

Patients are sampled from the test set. With ``--drift``, the population is
shifted (older patients with higher glucose and BMI), as if the API started
receiving patients from another population.

Run with ``python -m src.monitoring.simulate --url http://localhost:30080``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import urllib.request

import numpy as np
import pandas as pd

from src.config import DEFAULT_PARAMS_PATH, load_config
from src.data.bounds import COLUMN_BOUNDS
from src.data.prepare import TEST_FILENAME
from src.serving.schemas import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

# Population shift applied with --drift (added to each sampled patient)
DRIFT_SHIFT = {"Glucose": 35.0, "BMI": 6.0, "Age": 15.0}


def sample_patients(
    test_set: pd.DataFrame, size: int, drift: bool, seed: int = 0
) -> list[dict[str, float | None]]:
    """Sample API requests from the test set, optionally shifted, within the plausible bounds."""
    rng = np.random.default_rng(seed)
    rows = test_set.iloc[rng.integers(0, len(test_set), size)].copy()
    if drift:
        for column, shift in DRIFT_SHIFT.items():
            low, high = COLUMN_BOUNDS[column]
            rows[column] = (rows[column] + shift).clip(low, high)
    patients = []
    for _, row in rows.iterrows():
        patient: dict[str, float | None] = {}
        for field, column in FEATURE_COLUMNS.items():
            value = row[column]
            patient[field] = None if math.isnan(value) else float(value)
        patients.append(patient)
    return patients


def send(url: str, patient: dict[str, float | None]) -> None:
    """POST one patient to the /predict endpoint."""
    request = urllib.request.Request(  # noqa: S310 - URL given by the developer
        f"{url.rstrip('/')}/predict",
        data=json.dumps(patient).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5):  # noqa: S310  # nosec B310
        pass


def main(argv: list[str] | None = None) -> None:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:30080")
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--drift", action="store_true", help="shift the patient population")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = load_config(DEFAULT_PARAMS_PATH).data
    test_set = pd.read_csv(config.processed_dir / TEST_FILENAME)
    for patient in sample_patients(test_set, args.size, args.drift, args.seed):
        send(args.url, patient)
    logger.info(
        "Sent %d %s requests to %s", args.size, "drifted" if args.drift else "normal", args.url
    )


if __name__ == "__main__":
    main()

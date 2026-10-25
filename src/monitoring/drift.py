"""Detect data drift between the training data and the production requests.

The production requests are the JSON lines logged by the API (one per
prediction). Evidently compares the distribution of every feature with the
training set; the result is an HTML report and a JSON summary.

Run with ``python -m src.monitoring.drift`` (``make drift-report``).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

from src.config import DEFAULT_PARAMS_PATH, load_config

logger = logging.getLogger(__name__)

# Training set written by the prepare stage. Not imported from src.data.prepare:
# that module needs pandera, which is not installed in the drift check image
REFERENCE_FILENAME = "train.csv"
DEFAULT_CURRENT_PATH = Path("data/production/predictions.jsonl")
DEFAULT_REPORT_DIR = Path("reports/drift")
# Below this number of requests, the statistical tests are not meaningful
MIN_CURRENT_ROWS = 50
# The dataset has drifted when at least this share of the features has drifted
DRIFT_SHARE = 0.5


@dataclass(frozen=True)
class DriftResult:
    """Summary of one drift analysis."""

    dataset_drift: bool
    drifted_features: list[str]
    drifted_share: float
    reference_rows: int
    current_rows: int


def load_reference(processed_dir: Path, features: list[str]) -> pd.DataFrame:
    """Load the features of the training set, the reference distribution."""
    return pd.read_csv(processed_dir / REFERENCE_FILENAME)[features]


def load_current(path: Path, features: list[str]) -> pd.DataFrame:
    """Load the features of the logged predictions (unknown values become NaN)."""
    if not path.is_file():
        raise FileNotFoundError(f"no production data: {path} (run `make drift-collect`)")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("event") == "prediction":
            rows.append(record["features"])
    current = pd.DataFrame(rows, columns=features, dtype=float)
    if len(current) < MIN_CURRENT_ROWS:
        raise ValueError(
            f"only {len(current)} predictions in {path}, at least {MIN_CURRENT_ROWS} are needed"
        )
    return current


def detect_drift(
    reference: pd.DataFrame, current: pd.DataFrame, report_html: Path | None = None
) -> DriftResult:
    """Compare each feature of ``current`` with ``reference`` and summarize the drift."""
    definition = DataDefinition(numerical_columns=list(reference.columns))
    report = Report([DataDriftPreset(drift_share=DRIFT_SHARE)], include_tests=True)
    snapshot = report.run(
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
        current_data=Dataset.from_pandas(current, data_definition=definition),
    )
    if report_html is not None:
        report_html.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(report_html))

    # One "drift" test per feature: FAIL means the distribution has changed
    drifted = sorted(
        test["metric_config"]["params"]["column"]
        for test in snapshot.dict()["tests"]
        if test["id"] == "drift" and str(test["status"].value) == "FAIL"
    )
    share = len(drifted) / len(reference.columns)
    return DriftResult(
        dataset_drift=share >= DRIFT_SHARE,
        drifted_features=drifted,
        drifted_share=round(share, 4),
        reference_rows=len(reference),
        current_rows=len(current),
    )


def run(
    params_path: Path = DEFAULT_PARAMS_PATH,
    current_path: Path = DEFAULT_CURRENT_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> DriftResult:
    """Analyze the drift and write ``drift_report.html`` and ``drift_summary.json``."""
    config = load_config(params_path).data
    reference = load_reference(config.processed_dir, config.features)
    current = load_current(current_path, config.features)

    result = detect_drift(reference, current, report_dir / "drift_report.html")
    (report_dir / "drift_summary.json").write_text(
        json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
    )
    logger.info(
        "Drift on %d/%d features %s -> dataset drift: %s",
        len(result.drifted_features),
        len(config.features),
        result.drifted_features,
        result.dataset_drift,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; returns 1 on dataset drift with --fail-on-drift."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT_PATH)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fail-on-drift", action="store_true", help="exit 1 on dataset drift")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    result = run(current_path=args.current, report_dir=args.report_dir)
    return 1 if args.fail_on_drift and result.dataset_drift else 0


if __name__ == "__main__":
    raise SystemExit(main())

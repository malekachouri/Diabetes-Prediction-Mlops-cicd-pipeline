"""Decide whether a retrained candidate model should replace the champion.

Used by the retraining workflow (.github/workflows/retrain.yml) after
``dvc exp run``: the candidate metric comes from ``reports/metrics.json``, the
champion metric from the MLflow Model Registry. A pull request is opened only
when the candidate is strictly better, with the same rule as the promotion in
``src.models.register``.

Run with ``python -m src.models.retrain`` (needs the MLflow settings).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from src.config import DEFAULT_PARAMS_PATH, load_config
from src.models.register import SELECTION_METRIC, champion_metric
from src.tracking import is_tracking_enabled

logger = logging.getLogger(__name__)


def load_candidate_metric(metrics_path: Path) -> float:
    """Return the selection metric of the candidate written by the evaluate stage."""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return float(metrics[SELECTION_METRIC])


def should_open_pull_request(candidate: float, champion: float | None) -> bool:
    """A candidate replaces the champion only if it is strictly better."""
    return champion is None or candidate > champion


def summarize(candidate: float, champion: float | None, better: bool) -> str:
    """Markdown summary of the decision, for the workflow run page."""
    champion_text = "none" if champion is None else f"{champion:.4f}"
    decision = (
        "**Better than the champion**: a pull request is opened."
        if better
        else "**Not better than the champion**: the champion is kept, nothing is opened."
    )
    return (
        "## Retraining result\n\n"
        f"| Model | `{SELECTION_METRIC}` |\n|---|---|\n"
        f"| Candidate | {candidate:.4f} |\n"
        f"| Champion | {champion_text} |\n\n"
        f"{decision}\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Compare the candidate with the champion and write the decision for the workflow."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, help="file receiving better=true|false")
    parser.add_argument("--summary", type=Path, help="file receiving the Markdown summary")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not is_tracking_enabled():
        sys.exit("MLFLOW_TRACKING_URI is not set: the champion cannot be read")

    from mlflow.tracking import MlflowClient

    candidate = load_candidate_metric(load_config(DEFAULT_PARAMS_PATH).evaluate.metrics_path)
    champion = champion_metric(MlflowClient())
    better = should_open_pull_request(candidate, champion)
    logger.info("Candidate %.4f, champion %s -> better: %s", candidate, champion, better)

    if args.output:
        with args.output.open("a", encoding="utf-8") as output:
            output.write(f"better={str(better).lower()}\ncandidate={candidate:.4f}\n")
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as summary:
            summary.write(summarize(candidate, champion, better))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

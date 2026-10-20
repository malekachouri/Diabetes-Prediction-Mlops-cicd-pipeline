"""Evaluate the trained model on the held-out test set and save the metrics.

Run as a pipeline stage with ``python -m src.models.evaluate``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_PARAMS_PATH, Config, load_config
from src.data.prepare import TEST_FILENAME
from src.models.train import load_split
from src.tracking import is_tracking_enabled, log_run

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float | int]:
    """Compute classification metrics for the positive class (diabetic = 1).

    Accuracy alone is misleading on this imbalanced dataset (65/35), so recall,
    precision, F1 and ROC-AUC are reported as well, with the confusion matrix.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def evaluate(config: Config) -> dict[str, float | int]:
    """Score the saved model on the test set and write the metrics as JSON."""
    model = joblib.load(config.train.model_path)
    X_test, y_test = load_split(
        config.data.processed_dir / TEST_FILENAME, config.data.features, config.data.target
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test.to_numpy(), y_pred, y_proba)

    config.evaluate.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.evaluate.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Metrics on %d test rows: %s", len(X_test), metrics)

    if is_tracking_enabled():
        run_id = log_run(config, metrics, model, X_test)
        logger.info("Run logged to MLflow: %s", run_id)
    return metrics


def main(params_path: Path = DEFAULT_PARAMS_PATH) -> None:
    """Entry point of the ``evaluate`` pipeline stage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    evaluate(load_config(params_path))


if __name__ == "__main__":
    main()

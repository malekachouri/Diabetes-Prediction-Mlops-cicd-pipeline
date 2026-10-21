"""Log pipeline runs to MLflow (parameters, metrics, data version and model).

Tracking is optional: it is enabled only when ``MLFLOW_TRACKING_URI`` is set,
so the pipeline and the tests also run without any MLflow server.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.config import Config

DEFAULT_EXPERIMENT_NAME = "diabetes-prediction"
MODEL_ARTIFACT_PATH = "model"
# Types that skops must trust to deserialize the model: the node storage of the
# random forest trees and numpy dtypes. Safe here: we only load models we trained.
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree", "numpy.dtype"]


def is_tracking_enabled() -> bool:
    """Return True when an MLflow tracking server is configured."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


def flatten_params(config: Config) -> dict[str, str]:
    """Flatten the nested configuration into ``section.key`` MLflow parameters."""
    params: dict[str, str] = {}
    for section, values in asdict(config).items():
        for key, value in values.items():
            params[f"{section}.{key}"] = str(value)
    return params


def read_dvc_md5(data_path: Path) -> str:
    """Return the MD5 recorded in the DVC pointer file of a dataset."""
    pointer = data_path.with_name(data_path.name + ".dvc")
    content: dict[str, Any] = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    return str(content["outs"][0]["md5"])


def log_run(
    config: Config,
    metrics: dict[str, float | int],
    model: Pipeline,
    X_sample: pd.DataFrame,
) -> str:
    """Log one evaluated pipeline run to MLflow and return its run id."""
    import mlflow  # imported lazily: only needed when tracking is enabled
    from mlflow.models import infer_signature

    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME))
    with mlflow.start_run() as run:
        mlflow.log_params(flatten_params(config))
        mlflow.log_metrics({name: float(value) for name, value in metrics.items()})
        mlflow.set_tag("data.raw_md5", read_dvc_md5(config.data.raw_path))

        signature = infer_signature(X_sample, model.predict(X_sample))
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_dir = Path(tmp_dir) / MODEL_ARTIFACT_PATH
            mlflow.sklearn.save_model(
                model, model_dir, signature=signature, skops_trusted_types=SKOPS_TRUSTED_TYPES
            )
            mlflow.log_artifacts(str(model_dir), artifact_path=MODEL_ARTIFACT_PATH)
        return str(run.info.run_id)

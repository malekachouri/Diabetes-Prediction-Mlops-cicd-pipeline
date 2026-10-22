"""Register the evaluated model in the MLflow Model Registry and promote it.

The model of the current configuration is registered as a new version of
``diabetes-classifier``. It becomes the ``champion`` only if its
cross-validated recall beats the current champion (quality gate).

Run manually (it changes the shared registry) with ``python -m src.models.register``.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import DEFAULT_PARAMS_PATH, Config, load_config
from src.tracking import (
    DEFAULT_EXPERIMENT_NAME,
    MODEL_ARTIFACT_PATH,
    is_tracking_enabled,
    read_dvc_md5,
)

if TYPE_CHECKING:
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "diabetes-classifier"
CHAMPION_ALIAS = "champion"
SELECTION_METRIC = "cv_recall_mean"


def build_run_filter(config: Config) -> str:
    """Build an MLflow search filter matching the runs of the current configuration.

    The filter uses the training parameters and the DVC hash of the raw data:
    the Git commit alone does not identify a run (``dvc exp run`` changes the
    parameters without committing).
    """
    clauses = [f"params.`train.{key}` = '{value}'" for key, value in asdict(config.train).items()]
    clauses.append(f"tags.`data.raw_md5` = '{read_dvc_md5(config.data.raw_path)}'")
    clauses.append("attributes.status = 'FINISHED'")
    return " and ".join(clauses)


def find_latest_run(client: MlflowClient, config: Config) -> Run:
    """Return the most recent finished run logged with the current configuration."""
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", DEFAULT_EXPERIMENT_NAME)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise LookupError(f"MLflow experiment '{experiment_name}' does not exist")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=build_run_filter(config),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise LookupError("no finished MLflow run matches the current configuration")
    return runs[0]


def champion_metric(client: MlflowClient) -> float | None:
    """Return the selection metric of the current champion, or None if there is none."""
    from mlflow.exceptions import MlflowException

    try:
        version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS)
    except MlflowException:
        return None
    return client.get_run(str(version.run_id)).data.metrics.get(SELECTION_METRIC)


def register_and_promote(config: Config) -> tuple[str, bool]:
    """Register the current model and promote it if it beats the champion.

    Returns the new model version and whether it became the champion.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = find_latest_run(client, config)
    candidate = run.data.metrics[SELECTION_METRIC]
    current = champion_metric(client)

    existing = client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}' and run_id = '{run.info.run_id}'"
    )
    if existing:  # this run is already a model version: do not duplicate it
        version = existing[0].version
    else:
        version = mlflow.register_model(
            f"runs:/{run.info.run_id}/{MODEL_ARTIFACT_PATH}", REGISTERED_MODEL_NAME
        ).version
    logger.info(
        "Candidate: %s version %s (%s=%.4f)",
        REGISTERED_MODEL_NAME,
        version,
        SELECTION_METRIC,
        candidate,
    )

    if current is not None and candidate <= current:
        logger.info(
            "Not promoted: %s=%.4f does not beat the champion (%.4f)",
            SELECTION_METRIC,
            candidate,
            current,
        )
        return str(version), False

    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, str(version))
    logger.info("Version %s is the new %s", version, CHAMPION_ALIAS)
    return str(version), True


def main(params_path: Path = DEFAULT_PARAMS_PATH) -> None:
    """Entry point: register and promote the model of the current configuration."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if not is_tracking_enabled():
        sys.exit("MLFLOW_TRACKING_URI is not set: load .env first")
    register_and_promote(load_config(params_path))


if __name__ == "__main__":
    main()

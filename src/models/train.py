"""Train the model on the prepared training set and save it to disk.

Run as a pipeline stage with ``python -m src.models.train``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.config import DEFAULT_PARAMS_PATH, Config, TrainConfig, load_config
from src.data.prepare import TRAIN_FILENAME

logger = logging.getLogger(__name__)


def build_pipeline(config: TrainConfig) -> Pipeline:
    """Create the untrained model pipeline: median imputation + random forest.

    The imputer is part of the pipeline so that the medians are learned on the
    training data only and reapplied unchanged at prediction time.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=config.n_estimators,
                    max_depth=config.max_depth,
                    random_state=config.random_state,
                    n_jobs=1,  # single thread: bit-for-bit reproducible probabilities
                ),
            ),
        ]
    )


def load_split(path: Path, features: list[str], target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a prepared CSV and return the features matrix and the target vector."""
    df = pd.read_csv(path)
    return df[features], df[target]


def train(config: Config) -> Pipeline:
    """Fit the pipeline on the training set and save it to ``config.train.model_path``."""
    X_train, y_train = load_split(
        config.data.processed_dir / TRAIN_FILENAME, config.data.features, config.data.target
    )

    model = build_pipeline(config.train)
    model.fit(X_train, y_train)

    config.train.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.train.model_path)
    logger.info("Model trained on %d rows and saved to %s", len(X_train), config.train.model_path)
    return model


def main(params_path: Path = DEFAULT_PARAMS_PATH) -> None:
    """Entry point of the ``train`` pipeline stage."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    train(load_config(params_path))


if __name__ == "__main__":
    main()

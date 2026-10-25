"""Load the trained model served by the API.

The model file and its version are read from environment variables, so the
same code serves a local model or the model baked into a Docker image.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

DEFAULT_MODEL_PATH = Path("models/model.joblib")
DEFAULT_MODEL_VERSION = "dev"


@dataclass(frozen=True)
class ServingSettings:
    """Location and version of the model to serve."""

    model_path: Path
    model_version: str

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Read ``MODEL_PATH`` and ``MODEL_VERSION``, with local defaults."""
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", str(DEFAULT_MODEL_PATH))),
            model_version=os.environ.get("MODEL_VERSION", DEFAULT_MODEL_VERSION),
        )


def load_model(path: Path, expected_features: Iterable[str]) -> Pipeline:
    """Load the model and check that it was trained on the features of the API.

    Failing at startup is better than serving a model that expects other
    columns than the ones the API receives (training/serving skew).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"model file not found: {path} (run `make data-pull` or set MODEL_PATH)"
        )
    # joblib uses pickle: only load model files produced by our own pipeline
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"expected a scikit-learn Pipeline, got {type(model).__name__}")

    trained_on = set(getattr(model, "feature_names_in_", []))
    expected = set(expected_features)
    if trained_on != expected:
        raise ValueError(
            f"model features {sorted(trained_on)} do not match the API features {sorted(expected)}"
        )
    return model

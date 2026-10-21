"""Load and validate the pipeline parameters from ``params.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PARAMS_PATH = Path("params.yaml")
ALLOWED_CLASS_WEIGHTS = (None, "balanced", "balanced_subsample")


@dataclass(frozen=True)
class DataConfig:
    """Parameters of the data preparation stage."""

    raw_path: Path
    processed_dir: Path
    target: str
    features: list[str]
    zero_as_missing: list[str]
    test_size: float
    random_state: int

    def __post_init__(self) -> None:
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not self.features:
            raise ValueError("features must not be empty")
        if self.target in self.features:
            raise ValueError(f"target '{self.target}' must not be listed in features")
        unknown = sorted(set(self.zero_as_missing) - set(self.features))
        if unknown:
            raise ValueError(f"zero_as_missing columns are not features: {unknown}")


@dataclass(frozen=True)
class TrainConfig:
    """Parameters of the training stage."""

    model_path: Path
    n_estimators: int
    max_depth: int | None
    random_state: int
    min_samples_leaf: int = 1
    class_weight: str | None = None

    def __post_init__(self) -> None:
        if self.n_estimators < 1:
            raise ValueError(f"n_estimators must be >= 1, got {self.n_estimators}")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError(f"max_depth must be >= 1 or null, got {self.max_depth}")
        if self.min_samples_leaf < 1:
            raise ValueError(f"min_samples_leaf must be >= 1, got {self.min_samples_leaf}")
        if self.class_weight not in ALLOWED_CLASS_WEIGHTS:
            raise ValueError(
                f"class_weight must be one of {ALLOWED_CLASS_WEIGHTS}, got {self.class_weight}"
            )


@dataclass(frozen=True)
class EvaluateConfig:
    """Parameters of the evaluation stage."""

    metrics_path: Path


@dataclass(frozen=True)
class Config:
    """All pipeline parameters."""

    data: DataConfig
    train: TrainConfig
    evaluate: EvaluateConfig


def load_config(path: Path | str = DEFAULT_PARAMS_PATH) -> Config:
    """Read a params file and return a validated, typed configuration."""
    with Path(path).open(encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    data = raw["data"]
    train = raw["train"]
    evaluate = raw["evaluate"]

    return Config(
        data=DataConfig(
            raw_path=Path(data["raw_path"]),
            processed_dir=Path(data["processed_dir"]),
            target=data["target"],
            features=list(data["features"]),
            zero_as_missing=list(data["zero_as_missing"]),
            test_size=float(data["test_size"]),
            random_state=int(data["random_state"]),
        ),
        train=TrainConfig(
            model_path=Path(train["model_path"]),
            n_estimators=int(train["n_estimators"]),
            max_depth=train["max_depth"],
            random_state=int(train["random_state"]),
            min_samples_leaf=int(train.get("min_samples_leaf", 1)),
            class_weight=train.get("class_weight"),
        ),
        evaluate=EvaluateConfig(
            metrics_path=Path(evaluate["metrics_path"]),
        ),
    )

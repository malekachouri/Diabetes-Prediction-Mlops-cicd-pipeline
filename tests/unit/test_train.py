"""Unit tests for the training stage."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from src.config import Config, DataConfig, EvaluateConfig, TrainConfig
from src.data.prepare import TRAIN_FILENAME
from src.models.train import build_pipeline, load_split, main, train

FEATURES = ["Glucose", "BMI", "Age"]
TARGET = "Outcome"


def make_config(tmp_path: Path) -> Config:
    return Config(
        data=DataConfig(
            raw_path=tmp_path / "raw.csv",
            processed_dir=tmp_path / "processed",
            target=TARGET,
            features=FEATURES,
            zero_as_missing=["Glucose", "BMI"],
            test_size=0.25,
            random_state=42,
        ),
        train=TrainConfig(
            model_path=tmp_path / "models" / "model.joblib",
            n_estimators=10,
            max_depth=3,
            random_state=42,
        ),
        evaluate=EvaluateConfig(metrics_path=tmp_path / "reports" / "metrics.json"),
    )


def make_train_set(n_rows: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "Glucose": rng.integers(80, 200, n_rows).astype(float),
            "BMI": rng.uniform(18, 45, n_rows).round(1),
            "Age": rng.integers(21, 80, n_rows),
            TARGET: [0, 1] * (n_rows // 2),
        }
    )
    df.loc[0, "Glucose"] = np.nan  # prepared data contains missing values
    df.loc[1, "BMI"] = np.nan
    return df


def write_train_set(config: Config) -> None:
    config.data.processed_dir.mkdir(parents=True)
    make_train_set().to_csv(config.data.processed_dir / TRAIN_FILENAME, index=False)


def test_build_pipeline_uses_train_params(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    pipeline = build_pipeline(config.train)

    assert list(pipeline.named_steps) == ["imputer", "classifier"]
    classifier = pipeline.named_steps["classifier"]
    assert classifier.n_estimators == 10
    assert classifier.max_depth == 3
    assert classifier.random_state == 42
    assert classifier.min_samples_leaf == 1
    assert classifier.class_weight is None


def test_build_pipeline_passes_class_weight(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    train_config = TrainConfig(
        model_path=config.train.model_path,
        n_estimators=10,
        max_depth=3,
        random_state=42,
        min_samples_leaf=5,
        class_weight="balanced",
    )

    classifier = build_pipeline(train_config).named_steps["classifier"]

    assert classifier.min_samples_leaf == 5
    assert classifier.class_weight == "balanced"


def test_load_split_separates_features_and_target(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_train_set(config)

    X, y = load_split(config.data.processed_dir / TRAIN_FILENAME, FEATURES, TARGET)

    assert list(X.columns) == FEATURES
    assert y.name == TARGET
    assert len(X) == len(y) == 60


def test_train_handles_missing_values_and_saves_model(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_train_set(config)

    model = train(config)

    assert config.train.model_path.exists()
    loaded = joblib.load(config.train.model_path)
    assert isinstance(loaded, Pipeline)
    X, _ = load_split(config.data.processed_dir / TRAIN_FILENAME, FEATURES, TARGET)
    predictions = loaded.predict(X)
    assert set(predictions) <= {0, 1}
    np.testing.assert_array_equal(predictions, model.predict(X))


def test_train_is_reproducible(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_train_set(config)
    X, _ = load_split(config.data.processed_dir / TRAIN_FILENAME, FEATURES, TARGET)

    first = train(config).predict_proba(X)
    second = train(config).predict_proba(X)

    np.testing.assert_array_equal(first, second)


def test_main_reads_params_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    write_train_set(config)
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_path": str(config.data.raw_path),
                    "processed_dir": str(config.data.processed_dir),
                    "target": TARGET,
                    "features": FEATURES,
                    "zero_as_missing": config.data.zero_as_missing,
                    "test_size": config.data.test_size,
                    "random_state": config.data.random_state,
                },
                "train": {
                    "model_path": str(config.train.model_path),
                    "n_estimators": config.train.n_estimators,
                    "max_depth": config.train.max_depth,
                    "random_state": config.train.random_state,
                },
                "evaluate": {"metrics_path": str(config.evaluate.metrics_path)},
            }
        ),
        encoding="utf-8",
    )

    main(params_path)

    assert config.train.model_path.exists()

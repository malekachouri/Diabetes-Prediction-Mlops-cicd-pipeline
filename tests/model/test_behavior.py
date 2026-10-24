"""Behavioral tests of the trained model: expected reactions to input changes."""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

HIGH_RISK = {"Pregnancies": 2, "Glucose": 190, "BloodPressure": 80, "BMI": 40.0, "Age": 55}
LOW_RISK = {"Pregnancies": 0, "Glucose": 85, "BloodPressure": 70, "BMI": 22.0, "Age": 25}


def diabetes_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


# ---------- Minimum functionality: obvious cases ----------
def test_obvious_high_risk_patient_is_flagged(model: Pipeline) -> None:
    assert model.predict(pd.DataFrame([HIGH_RISK]))[0] == 1


def test_obvious_low_risk_patient_is_not_flagged(model: Pipeline) -> None:
    assert model.predict(pd.DataFrame([LOW_RISK]))[0] == 0


def test_missing_measurements_are_handled(model: Pipeline) -> None:
    patient = {**LOW_RISK, "Glucose": np.nan, "BloodPressure": np.nan, "BMI": np.nan}

    probability = diabetes_probability(model, pd.DataFrame([patient]))[0]

    assert 0.0 <= probability <= 1.0


# ---------- Directional expectations: known risk factors ----------
@pytest.mark.parametrize(("feature", "increase"), [("Glucose", 40), ("BMI", 8)])
def test_risk_factor_increases_the_predicted_risk(
    model: Pipeline, test_features: pd.DataFrame, feature: str, increase: float
) -> None:
    worse = test_features.copy()
    worse[feature] = worse[feature] + increase

    before = diabetes_probability(model, test_features)
    after = diabetes_probability(model, worse)

    assert after.mean() > before.mean()
    # a random forest is not monotonic: allow a few individual exceptions
    assert (after < before).mean() <= 0.15


# ---------- Invariance: negligible changes must not flip predictions ----------
@pytest.mark.parametrize(("feature", "delta"), [("BMI", 0.1), ("BloodPressure", 1)])
def test_negligible_change_does_not_flip_predictions(
    model: Pipeline, test_features: pd.DataFrame, feature: str, delta: float
) -> None:
    nudged = test_features.copy()
    nudged[feature] = nudged[feature] + delta

    flipped = (model.predict(nudged) != model.predict(test_features)).mean()

    assert flipped <= 0.05


def test_predictions_are_deterministic(model: Pipeline, test_features: pd.DataFrame) -> None:
    np.testing.assert_array_equal(
        diabetes_probability(model, test_features), diabetes_probability(model, test_features)
    )

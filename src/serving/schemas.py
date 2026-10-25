"""Request and response models of the prediction API (Pydantic).

The bounds of every field come from ``src.data.bounds``: the API rejects the
same implausible values as the data validation of the training pipeline.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from src.data.bounds import COLUMN_BOUNDS

# API field name -> column name the model was trained on
FEATURE_COLUMNS = {
    "pregnancies": "Pregnancies",
    "glucose": "Glucose",
    "blood_pressure": "BloodPressure",
    "bmi": "BMI",
    "age": "Age",
}
EXAMPLE_PATIENT: dict[str, JsonValue] = {
    "pregnancies": 2,
    "glucose": 148,
    "blood_pressure": 72,
    "bmi": 33.6,
    "age": 50,
}


def _low(column: str) -> float:
    return COLUMN_BOUNDS[column][0]


def _high(column: str) -> float:
    return COLUMN_BOUNDS[column][1]


class PatientFeatures(BaseModel):
    """Health indicators of one patient. Unknown measurements are sent as null."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    pregnancies: int = Field(
        description="Number of pregnancies", ge=_low("Pregnancies"), le=_high("Pregnancies")
    )
    glucose: float | None = Field(
        default=None,
        description="Plasma glucose (mg/dL), null if unknown",
        ge=_low("Glucose"),
        le=_high("Glucose"),
    )
    blood_pressure: float | None = Field(
        default=None,
        description="Diastolic blood pressure (mm Hg), null if unknown",
        ge=_low("BloodPressure"),
        le=_high("BloodPressure"),
    )
    bmi: float | None = Field(
        default=None,
        description="Body mass index (kg/m²), null if unknown",
        ge=_low("BMI"),
        le=_high("BMI"),
    )
    age: int = Field(description="Age (years)", ge=_low("Age"), le=_high("Age"))

    def to_frame(self) -> pd.DataFrame:
        """Return a one-row DataFrame with the training column names (null -> NaN)."""
        values = self.model_dump()
        row = {column: values[field] for field, column in FEATURE_COLUMNS.items()}
        return pd.DataFrame([row], dtype=float)


class Prediction(BaseModel):
    """Prediction returned for one patient."""

    diabetic: bool = Field(description="True when the model predicts diabetes")
    probability: float = Field(ge=0, le=1, description="Predicted probability of diabetes")
    model_version: str = Field(description="Version of the model that made the prediction")


class Health(BaseModel):
    """Health check response."""

    status: str
    model_version: str

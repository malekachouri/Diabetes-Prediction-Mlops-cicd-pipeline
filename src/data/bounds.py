"""Plausible bounds of each column, shared by data validation and the API.

This module has no heavy dependency (no pandera), so the serving code can
import it in a production image that only installs the main dependencies.
"""

from __future__ import annotations

# Plausible (min, max) bounds of each column, from domain knowledge.
COLUMN_BOUNDS: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 400),
    "BloodPressure": (20, 200),
    "SkinThickness": (5, 100),
    "Insulin": (10, 1000),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0, 3),
    "Age": (18, 120),
}

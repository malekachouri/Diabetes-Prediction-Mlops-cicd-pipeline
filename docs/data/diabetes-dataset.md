# Dataset Card: Pima Indians Diabetes

## Overview

| | |
|---|---|
| **File** | `data/raw/diabetes.csv` (tracked by DVC) |
| **Original source** | National Institute of Diabetes and Digestive and Kidney Diseases (via UCI / Kaggle) |
| **Download URL** | https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv |
| **MD5** | `f2906818eda8fcfc8f8416557ab1e6df` |
| **Rows** | 768 patients |
| **Columns** | 9 (8 features + 1 target) |
| **Population** | Females, at least 21 years old, of Pima Indian heritage |

## Schema

| Column | Type | Description | Used by the model |
|---|---|---|---|
| `Pregnancies` | int | Number of pregnancies | Yes |
| `Glucose` | int | Plasma glucose concentration (2h oral glucose tolerance test) | Yes |
| `BloodPressure` | int | Diastolic blood pressure (mm Hg) | Yes |
| `SkinThickness` | int | Triceps skin fold thickness (mm) | No |
| `Insulin` | int | 2-hour serum insulin (mu U/ml) | No |
| `BMI` | float | Body mass index (kg/m²) | Yes |
| `DiabetesPedigreeFunction` | float | Diabetes likelihood based on family history | No |
| `Age` | int | Age in years | Yes |
| `Outcome` | int | **Target**: 1 = diabetic, 0 = not diabetic | Target |

## Class balance

| Outcome | Count | Share |
|---|---|---|
| 0 (not diabetic) | 500 | 65.1% |
| 1 (diabetic) | 268 | 34.9% |

The dataset is **imbalanced**: accuracy alone is a misleading metric.
Use recall, precision, F1 and ROC-AUC instead.

## Known data quality issues

Several columns use `0` to encode **missing values**, which is
physiologically impossible:

| Column | Rows with `0` |
|---|---|
| `Glucose` | 5 |
| `BloodPressure` | 35 |
| `BMI` | 11 |
| `SkinThickness` | 227 |
| `Insulin` | 374 |

These must be handled during data cleaning (step 3) and enforced by data
validation tests (step 5).

## Limitations

- Small dataset (768 rows): high variance between train/test splits.
- Single population (Pima Indian women): the model must not be assumed to
  generalize to other populations.
- This project is for educational purposes only and must not be used for
  medical decisions.

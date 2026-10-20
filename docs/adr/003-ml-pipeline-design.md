# ADR-003: ML pipeline design

- **Status:** Accepted
- **Date:** 2026-10-20

## Context

The starting point was a single `train.py` script with hard-coded values:
impossible to test, to re-run partially, or to trace back to the exact code,
data and parameters that produced a model. The dataset also has specific
issues documented in the [dataset card](../data/diabetes-dataset.md):
impossible zero values and a 65/35 class imbalance.

## Decision

**Structure**
- Three stages, one module each: `prepare` (`src/data/prepare.py`), `train`
  (`src/models/train.py`) and `evaluate` (`src/models/evaluate.py`).
- Every stage exposes small pure functions (unit-tested) and a `main()`
  entry point run with `python -m`.
- All parameters live in `params.yaml`, loaded into typed, immutable and
  validated dataclasses (`src/config.py`) that fail fast on invalid values.
- Stages are chained with DVC (`dvc.yaml`); `dvc.lock` records the hashes of
  every input and output of the last run.

**Data and modelling**
- Impossible zeros (`Glucose`, `BloodPressure`, `BMI`) are flagged as missing
  in `prepare`, but **imputed inside the model pipeline**: medians are learned
  on the training set only, which avoids data leakage and guarantees the same
  preprocessing at inference time.
- Train/test split is stratified on the target to keep the class balance.
- Baseline model: random forest (robust, no feature scaling needed).
- `n_jobs=1`: parallel trees made `predict_proba` non-deterministic at the
  last floating-point digit, which caused a flaky reproducibility test.

**Evaluation**
- Recall on the diabetic class is the primary metric: for screening, a missed
  diabetic patient (false negative) costs more than a false alarm.
- Precision, F1, ROC-AUC, accuracy and the confusion matrix are also reported.
- `reports/metrics.json` is tracked in Git to compare runs with
  `dvc metrics diff`.

## Alternatives considered

- **Single script:** simplest, but untestable and always re-runs everything.
- **Imputation in `prepare`:** simpler stage, but the medians would be
  computed on the full dataset, leaking test information into training.
- **Airflow / Prefect / Kubeflow:** powerful orchestrators, but they require
  running a server; DVC pipelines are enough for a single-machine workflow and
  integrate with the data versioning already in place (ADR-002).
- **Accuracy as the main metric:** a model predicting "not diabetic" for every
  patient would reach 65% accuracy while being useless.

## Consequences

**Positive**
- `dvc repro` re-runs only what changed and restores cached results.
- Same code + data + parameters give bit-for-bit identical metrics on any
  machine.
- Each stage is covered by unit tests on small synthetic datasets, runnable
  in CI without the real data.

**Negative**
- Every new dependency of a stage (including imports) must be declared in
  `dvc.yaml`, otherwise DVC may skip a stage that should re-run.
- The baseline recall (0.59) is too low for screening; this is addressed by
  the experiments of step 4.

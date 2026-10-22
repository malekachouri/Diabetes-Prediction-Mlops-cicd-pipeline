# ADR-004: Experiment tracking and model registry with MLflow

- **Status:** Accepted
- **Date:** 2026-10-22

## Context

DVC makes every pipeline run reproducible and compares the workspace with the
last commit (`dvc metrics diff`). Improving the baseline recall (0.59) required
comparing many configurations, and answering questions DVC alone does not:
which experiment gave the best result and with which parameters, where is the
model of that experiment, and which model is approved for production.

## Decision

**Tracking**
- MLflow, using the free server hosted by DagsHub (same account as the DVC
  remote, ADR-002).
- Tracking is optional: it is enabled only when `MLFLOW_TRACKING_URI` is set.
  The pipeline and the tests run without any server; an autouse fixture
  removes the `MLFLOW_*` variables during the tests.
- Each `evaluate` run logs every parameter of `params.yaml`, the metrics, the
  DVC hash of the raw data, the Git commit (automatic tag) and the model.
- The model is saved with skops (secure format) with explicitly trusted types.

**Model selection**
- Experiments are run with `dvc exp run -S ...` (no commit per experiment).
- They are compared on the 5-fold cross-validated recall of the training set
  (`cv_recall_mean`), ties broken by the lowest standard deviation and the
  simplest model. The test set is only used to confirm the final choice.

**Registry**
- Registered model `diabetes-classifier`; the production model is the version
  carrying the `champion` alias.
- `make register` finds the run of the current configuration (training
  parameters + data hash, not the commit), registers it once, and moves the
  alias only if its `cv_recall_mean` beats the current champion.

## Alternatives considered

- **DVC experiments only:** good local comparison, but no shared UI, no model
  registry, and models are not attached to experiments.
- **Weights & Biases:** excellent UI, but proprietary and limited free tier.
- **Self-hosted MLflow server:** full control, but requires a machine, a
  database and storage to maintain.
- **Selecting on the test set:** simpler, but leaks the test set and gives an
  optimistic final score.
- **Model stages (Staging / Production):** deprecated in recent MLflow versions
  in favour of aliases.

## Consequences

**Positive**
- Every model in production can be traced back to its code, data and parameters.
- The API (step 7) loads `models:/diabetes-classifier@champion`: changing the
  production model means moving an alias, not changing code.
- The quality gate prevents a worse model from replacing the champion, and will
  be reused by the continuous deployment (step 8).

**Negative**
- The Git commit tag is the same for all experiments of a grid: parameters must
  be logged to identify a run.
- `dvc exp run -S` rewrites `params.yaml` (comments, indentation): the winning
  values are applied by hand.
- Depends on DagsHub's free tier; the MLflow client and server versions must
  stay compatible.

# ADR-010: Continuous training with a human in the loop

- **Status:** Accepted
- **Date:** 2026-10-26

## Context

Step 9 detects when the production requests drift away from the training
data, but nothing reacts to it. Retraining is only useful with new labeled
data: the API never receives the confirmed diagnosis of a patient, and the
768 patients of the Pima dataset are already used. Retraining on the same data
gives the same model. The model screens patients for diabetes: a worse model
must never replace a better one, and changes to it must stay reviewable.

## Decision

**No invented data**
- The project does not generate synthetic patients to demonstrate the loop:
  the deployed model is trained on real, labeled data only.

**Drift opens an issue**
- On dataset drift, the `drift-monitor` CronJob opens a GitHub issue labelled
  `data-drift` with the drifted features, the number of predictions, the model
  versions and the next steps. Only one such issue is open at a time.
- The GitHub token is optional, fine-grained (Issues only, this repository
  only) and stored as a Kubernetes Secret created by hand, never in Git.

**Retrain workflow** (`.github/workflows/retrain.yml`, started by hand)
- `dvc exp run` trains a candidate, with optional parameter changes (unknown
  keys are rejected) and logs it to MLflow.
- The candidate must pass the data and model tests (quality gate).
- `src/models/retrain.py` compares its cross-validated recall with the
  champion of the MLflow Model Registry: a pull request is opened only when
  the candidate is strictly better, with the same rule as the promotion.
- The candidate branch is pushed with the deploy key, so that the CI runs on
  it and provides the required checks; the merge stays a human decision.

**Registry aligned with production**
- After each merge on `main`, the `register` job of the CI registers the
  deployed model and promotes it to champion if it is better.

## Alternatives considered

- **Automatic retraining on every drift alert:** without new labels it
  retrains on the same data and produces the same model; with labels, a
  medical model should still be reviewed before deployment.
- **Synthetic labeled data for the demonstration:** shows a complete
  automatic loop, but the deployed model would be partly trained on invented
  patients.
- **Scheduled retraining (e.g. weekly):** useful when labeled data arrives
  continuously; here the data changes rarely, so a schedule would mostly
  retrain the same model.
- **Deploying the candidate directly, without a pull request:** faster, but
  it removes the review and the full CI run on the new model.

## Consequences

**Positive**
- The loop is closed: monitoring → issue → new data → retraining → pull
  request → CI/CD → registry, with a human only where a human is needed.
- A retrained model can only reach production if it passes the quality gate,
  beats the champion and passes the CI.
- Real runs confirmed the safeguards: the same data kept the champion
  (0.7663 = 0.7663) and a worse candidate was rejected (0.7570 < 0.7663).

**Negative**
- Retraining is not fully automatic: it waits for new labeled data and for a
  human to start the workflow and merge the pull request.
- The issue is authored by the owner of the token; a GitHub App would show a
  bot account and would not depend on a personal token that expires.
- The champion is chosen on one metric (cross-validated recall); a richer
  comparison (precision, calibration, fairness by age group) is future work.

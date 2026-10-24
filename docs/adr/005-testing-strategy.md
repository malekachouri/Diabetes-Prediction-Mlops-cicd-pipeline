# ADR-005: Testing strategy for code, data and model

- **Status:** Accepted
- **Date:** 2026-10-24

## Context

In an ML project, most failures do not come from a code bug: corrupted or
drifted data, or a degraded model, still produce a pipeline that "runs".
Unit tests alone cannot detect them. The tests must also run in a CI that may
not have access to the real data (DagsHub credentials).

## Decision

Three test layers, one folder each:

| Layer | Folder | Data | Purpose |
|---|---|---|---|
| Unit | `tests/unit/` | small synthetic datasets | each module behaves as specified |
| Data | `tests/data/` | synthetic + real data | pandera schemas accept valid data and reject invalid data |
| Model | `tests/model/` | trained model + test set | quality gate and expected behavior |

**Data validation** (pandera, `src/data/schemas.py`)
- Runs inside the `prepare` stage: invalid data stops the pipeline (fail fast)
  and every error is reported at once.
- Rules come from the dataset card: plausible physiological bounds, zeros
  only where they encode "missing", no remaining zeros after preparation,
  exact columns, minimum size, share of diabetic patients in [0.2, 0.6].

**Model tests**
- Quality gate on the committed metrics: recall >= 0.75, cross-validated
  recall >= 0.70, precision >= 0.50, ROC-AUC >= 0.75, cv recall std <= 0.10.
  Thresholds sit below the current model (noise tolerance) and above the
  baseline it replaced.
- Behavioral tests: obvious high/low risk patients, missing measurements,
  directional expectations (glucose, BMI), invariance to negligible changes.
- The gate was proven to reject the baseline model before being merged.

**Execution**
- Tests needing the real data or the trained model are skipped (not failed)
  when those files are absent.
- `make check` fails below 80% coverage of `src/`; `make test-unit`,
  `make test-data` and `make test-model` run one layer at a time.

## Alternatives considered

- **Great Expectations:** powerful, but heavier (contexts, stores, HTML docs)
  than needed for one dataset; pandera schemas are plain Python and testable.
- **Thresholds on the test set only:** a single split is noisy on 154 rows;
  the cross-validated recall is checked as well.
- **Strict monotonicity tests:** a random forest is not monotonic per patient;
  the directional tests check the mean risk and bound the exceptions (15%).

## Consequences

**Positive**
- A corrupted dataset, a drifted class balance or a degraded model fails the
  pipeline or the tests instead of silently reaching production.
- The same tests run locally and in the CI (step 6).

**Negative**
- The plausible bounds and thresholds must be maintained when the data or the
  business requirements change.
- In a CI without data, data and model tests are skipped: the CI must pull the
  data to enforce them (step 6).

# ADR-007: Model serving with FastAPI and Docker

- **Status:** Accepted
- **Date:** 2026-10-25

## Context

The trained model is a scikit-learn pipeline saved in `models/model.joblib`;
only Python code can use it. Applications need to send the health indicators
of a patient over HTTP and get a prediction. The served model must receive its
inputs in exactly the same form as during training, the service must run the
same way everywhere (laptop, CI, Kubernetes in step 8), and it must not carry
secrets or development tools.

## Decision

**API** ([`src/serving/`](../../src/serving/))
- FastAPI app with `GET /health`, `POST /predict` and the generated `/docs`.
- Requests are validated by Pydantic models: snake_case fields mapped to the
  training columns, unknown fields rejected, and the plausible bounds of
  [`src/data/bounds.py`](../../src/data/bounds.py), shared with the pandera
  schemas of the training pipeline.
- Unknown measurements are sent as `null` and become `NaN`, as the impossible
  zeros of the training data; a `0` returns `422`.
- The model is loaded once at startup (`lifespan`). The app refuses to start
  if the file is missing or if the model was trained on other features.
- The model location and version come from `MODEL_PATH` and `MODEL_VERSION`.

**Image** ([`docker/serve.Dockerfile`](../../docker/serve.Dockerfile))
- Multi-stage build: uv installs only the main dependencies
  (`uv sync --frozen --no-default-groups`) in a builder stage; the runtime
  stage copies the virtual environment, `src/` and the model, without uv.
- Base image and uv pinned to exact versions; non-root user; health check on
  `/health`; `.dockerignore` as an allowlist so that secrets and data never
  reach the build.
- The model is baked into the image, and `MODEL_VERSION` is the Git commit.

**CI**
- The `Docker image` job builds the image with the model that passed the data
  and model tests (shared as an artifact), then `docker/smoke-test.sh` starts
  a container and checks `/health`, `/predict`, a `422` and the non-root user.

## Alternatives considered

- **Load `models:/diabetes-classifier@champion` from MLflow at startup:** the
  image would need MLflow, network access and DagsHub credentials at runtime,
  the startup would depend on DagsHub, and the same image could serve
  different models over time. Baking the model keeps the image immutable and
  free of secrets; choosing which model is deployed belongs to step 8.
- **MLflow model serving or BentoML:** ready-made servers, but heavier and with
  less control over input validation and error messages.
- **Flask:** no built-in request validation or generated documentation.
- **A single-stage image with all dependency groups:** larger, and it would
  ship pytest, DVC and MLflow to production.

## Consequences

**Positive**
- The API rejects the same implausible values as the data validation, and a
  wrong model fails at startup instead of returning wrong predictions.
- What the CI tests is exactly what is delivered: same model file, same image.
- The image runs without credentials and reports which model version answers.

**Negative**
- The image is about 700 MB, mostly scipy, pandas and scikit-learn.
- A new model requires a new image build.
- `joblib` uses pickle: only model files produced by our own pipeline may be
  loaded.

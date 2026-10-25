# ==========================================================
#  Scheduled drift check image (Kubernetes CronJob)
#  Build from the repository root (the training data must exist:
#  make data-pull):
#    docker build -f docker/drift.Dockerfile -t diabetes-drift .
# ==========================================================

# ---------- Stage 1: install the dependencies with uv ----------
FROM python:3.11.17-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Main dependencies + the monitoring group (Evidently); no dev, data or tracking
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-default-groups --group monitoring --no-install-project

# ---------- Stage 2: runtime image ----------
FROM python:3.11.17-slim-trixie AS runtime

LABEL org.opencontainers.image.source="https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline" \
      org.opencontainers.image.description="Scheduled data drift check of the diabetes API (Evidently)" \
      org.opencontainers.image.licenses="MIT"

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
# Reference distribution: the training set of the deployed model (public dataset)
COPY params.yaml params.yaml
COPY data/processed/train.csv data/processed/train.csv
COPY src/ src/

# HOME and caches in /tmp: the root filesystem is read-only in the cluster
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp

USER app

CMD ["python", "-m", "src.monitoring.job"]

# ==========================================================
#  Prediction API image
#  Build from the repository root (the model must exist:
#  make data-pull):
#    docker build -f docker/serve.Dockerfile -t diabetes-api .
# ==========================================================

# ---------- Stage 1: install the dependencies with uv ----------
FROM python:3.11.17-slim-trixie AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /usr/local/bin/uv

# Compile .pyc files now (faster startup), copy instead of hard links (cache
# mount), and use the Python of the image instead of downloading one
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Only the main dependencies: no dev, data or tracking groups in production.
# Dependencies change rarely: this layer stays cached when only the code changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-default-groups --no-install-project

# ---------- Stage 2: runtime image (no uv, no build tools) ----------
FROM python:3.11.17-slim-trixie AS runtime

# Links the published package to its repository on GitHub
LABEL org.opencontainers.image.source="https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline" \
      org.opencontainers.image.description="Diabetes prediction API (FastAPI + scikit-learn)" \
      org.opencontainers.image.licenses="MIT"

# Run as an unprivileged user, never as root
RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ src/
COPY models/model.joblib model.joblib

# Version of the served model, set at build time (e.g. the Git commit)
ARG MODEL_VERSION=dev
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model.joblib \
    MODEL_VERSION=$MODEL_VERSION

USER app
EXPOSE 8000

# The container is "healthy" only when the API answers on /health
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]

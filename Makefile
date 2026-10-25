# ==========================================================
#  Diabetes Prediction - MLOps CI/CD Pipeline
#  Single command interface for developers AND the CI
# ==========================================================

# Use ">" instead of a tab at the start of recipe lines
# (avoids "missing separator" errors after copy-paste)
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
SHELL := /bin/bash

SRC   := src
TESTS := tests

# Reference branch of the metrics report (override: make report BASE=main)
BASE ?= origin/main

# Port of the local API (override: make serve PORT=8003)
PORT ?= 8002

# Export the variables of .env (MLflow settings) when the file exists
LOAD_ENV := if [ -f .env ]; then set -a; . ./.env; set +a; fi;

.PHONY: help install install-ci lint format typecheck security test check pre-commit \
        test-unit test-data test-model test-api data-pull data-push data-status repro \
        pipeline-check metrics report dag serve register clean

help: ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
>   awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

# ---------- Environment ----------
install: ## Install the dev environment + pre-commit hooks
> uv sync
> uv run pre-commit install

install-ci: ## Install the exact locked environment (used by the CI)
> uv sync --frozen

# ---------- Quality ----------
lint: ## Check style and errors (ruff)
> uv run ruff check .
> uv run ruff format --check .

format: ## Auto-fix style issues (ruff)
> uv run ruff check --fix .
> uv run ruff format .

typecheck: ## Check types (mypy)
> uv run mypy

security: ## Scan code for security issues (bandit)
> uv run bandit -q -c pyproject.toml -r $(SRC)

test: ## Run all tests with coverage (fails below 80%)
> uv run pytest

test-unit: ## Run the unit tests of the code (no data needed)
> uv run pytest --no-cov tests/unit

test-data: ## Run the data validation tests (real data tests need make data-pull)
> uv run pytest --no-cov tests/data

test-model: ## Run the model quality gate and behavioral tests (need make repro)
> uv run pytest --no-cov tests/model

test-api: ## Run the API integration tests (no data needed)
> uv run pytest --no-cov tests/integration

check: lint typecheck security test ## Run ALL checks (same as the CI)

pre-commit: ## Run all pre-commit hooks on all files
> uv run pre-commit run --all-files

# ---------- Data (DVC) ----------
data-pull: ## Download the data versioned in Git from the DVC remote
> uv run dvc pull

data-push: ## Upload local data versions to the DVC remote
> uv run dvc push

data-status: ## Show local and remote data status
> uv run dvc status
> uv run dvc status --cloud

# ---------- ML pipeline (DVC) ----------
repro: ## Run the ML pipeline (only the stages that changed), logged to MLflow if .env exists
> $(LOAD_ENV) uv run dvc repro

pipeline-check: ## Fail if code, params or data changed without `make repro`
> @status="$$(uv run dvc status --json)"; \
>   if [ "$$status" != "{}" ]; then \
>     uv run dvc status; \
>     echo "Pipeline not up to date: run 'make repro', then commit dvc.lock"; \
>     exit 1; \
>   fi; \
>   echo "Pipeline is up to date with dvc.lock"

metrics: ## Show the current metrics and the diff with the last commit
> uv run dvc metrics show
> uv run dvc metrics diff

report: ## Write the metrics and params diff against BASE to reports/report.md
> @mkdir -p reports
> @{ \
>   echo "## Model report"; \
>   echo ""; \
>   echo "Metrics of this branch compared with \`$(BASE)\`."; \
>   echo "Quality gate: recall >= 0.75, cv_recall_mean >= 0.70,"; \
>   echo "precision >= 0.50, roc_auc >= 0.75 (tests/model)."; \
>   echo ""; \
>   uv run dvc metrics diff $(BASE) --all --no-path --md; \
>   echo "### Parameters"; \
>   echo ""; \
>   if [ "$$(uv run dvc params diff $(BASE) --json)" = "{}" ]; then \
>     echo "No parameter changed."; \
>   else \
>     uv run dvc params diff $(BASE) --no-path --md; \
>   fi; \
> } > reports/report.md
> @cat reports/report.md

dag: ## Show the pipeline stages graph
> uv run dvc dag

register: ## Register the current model in MLflow and promote it if better (needs .env)
> $(LOAD_ENV) uv run python -m src.models.register

# ---------- Serving ----------
serve: ## Start the prediction API on http://127.0.0.1:PORT (default 8002, docs: /docs)
> uv run uvicorn src.serving.app:app --reload --host 127.0.0.1 --port $(PORT)

# ---------- Cleanup ----------
clean: ## Remove caches and temporary files
> find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
> rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml

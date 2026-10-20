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

.PHONY: help install install-ci lint format typecheck security test check pre-commit \
        data-pull data-push data-status clean

help: ## Show this help
> @grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
>   awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

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

test: ## Run tests with coverage (pytest)
> uv run pytest

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

# ---------- Cleanup ----------
clean: ## Remove caches and temporary files
> find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
> rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml

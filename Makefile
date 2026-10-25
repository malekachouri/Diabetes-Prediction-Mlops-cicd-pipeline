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

# Name of the API Docker image (override: make docker-build IMAGE=my-api)
IMAGE ?= diabetes-api

# Extra docker build flags (the CI adds its layer cache)
DOCKER_BUILD_FLAGS ?=

# Container registry of the published image (override: make docker-push REGISTRY=...)
REGISTRY ?= ghcr.io/malekachouri

# Local Kubernetes cluster: every kubectl command targets it explicitly
KIND_CLUSTER ?= diabetes
KUBE_CONTEXT := kind-$(KIND_CLUSTER)

# Pinned Argo CD release installed in the local cluster
ARGOCD_VERSION ?= v3.5.6

# Export the variables of .env (MLflow settings) when the file exists
LOAD_ENV := if [ -f .env ]; then set -a; . ./.env; set +a; fi;

.PHONY: help install install-ci lint format typecheck security test check pre-commit \
        test-unit test-data test-model test-api data-pull data-push data-status repro \
        pipeline-check metrics report dag serve docker-build docker-run docker-smoke \
        docker-push kind-up kind-down k8s-apply k8s-set-image argocd-install argocd-app \
        argocd-ui monitoring-app prometheus-ui grafana-ui register clean

help: ## Show this help
> @grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | \
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

# ---------- Docker ----------
docker-build: ## Build the API image with the local model (needs make data-pull)
> docker build $(DOCKER_BUILD_FLAGS) -f docker/serve.Dockerfile \
>   --build-arg MODEL_VERSION=$$(git rev-parse --short HEAD) -t $(IMAGE):latest .

docker-run: ## Run the API image on http://127.0.0.1:PORT (Ctrl+C to stop)
> docker run --rm -p $(PORT):8000 $(IMAGE):latest

docker-smoke: ## Smoke test the API image: /health, /predict, non-root user
> docker/smoke-test.sh $(IMAGE):latest

docker-push: ## Push the API image to REGISTRY, tagged with the Git commit and latest
> @tag="$$(git rev-parse --short HEAD)"; \
>   for version in "$$tag" latest; do \
>     docker tag $(IMAGE):latest $(REGISTRY)/$(IMAGE):$$version; \
>     docker push $(REGISTRY)/$(IMAGE):$$version; \
>   done; \
>   echo "Published $(REGISTRY)/$(IMAGE):$$tag"

# ---------- Kubernetes (local kind cluster) ----------
kind-up: ## Create the local kind cluster (infra/k8s/kind-config.yaml)
> kind create cluster --config infra/k8s/kind-config.yaml

kind-down: ## Delete the local kind cluster
> kind delete cluster --name $(KIND_CLUSTER)

k8s-apply: ## Deploy the local overlay to the kind cluster and wait for the rollout
> kubectl --context $(KUBE_CONTEXT) apply -k infra/k8s/overlays/local
> kubectl --context $(KUBE_CONTEXT) -n diabetes rollout status deployment/diabetes-api \
>   --timeout=180s

k8s-set-image: ## Set the deployed image tag in the local overlay (TAG=<commit>)
> @[[ "$(TAG)" =~ ^[0-9a-f]{7,40}$$ ]] || { echo "usage: make k8s-set-image TAG=<commit>"; exit 1; }
> sed -i 's/^\(    newTag: \).*/\1"$(TAG)"/' infra/k8s/overlays/local/kustomization.yaml
> @grep -n "newTag" infra/k8s/overlays/local/kustomization.yaml
> kubectl kustomize infra/k8s/overlays/local > /dev/null

argocd-install: ## Install Argo CD in the kind cluster and wait until it is ready
> kubectl --context $(KUBE_CONTEXT) create namespace argocd --dry-run=client -o yaml | \
>   kubectl --context $(KUBE_CONTEXT) apply -f -
> kubectl --context $(KUBE_CONTEXT) apply -n argocd --server-side --force-conflicts \
>   -f https://raw.githubusercontent.com/argoproj/argo-cd/$(ARGOCD_VERSION)/manifests/install.yaml
> kubectl --context $(KUBE_CONTEXT) -n argocd wait --for=condition=Available deployment --all \
>   --timeout=300s

argocd-app: ## Hand the API deployment over to Argo CD (infra/argocd/application.yaml)
> kubectl --context $(KUBE_CONTEXT) apply -f infra/argocd/application.yaml

argocd-ui: ## Argo CD UI on https://localhost:8443 (prints the admin password, Ctrl+C to stop)
> @printf 'user: admin\npassword: '; \
>   kubectl --context $(KUBE_CONTEXT) -n argocd get secret argocd-initial-admin-secret \
>     -o jsonpath='{.data.password}' | base64 -d; echo
> kubectl --context $(KUBE_CONTEXT) -n argocd port-forward svc/argocd-server 8443:443

# ---------- Monitoring (Prometheus + Grafana in the kind cluster) ----------
monitoring-app: ## Hand Prometheus, Grafana and the dashboards over to Argo CD
> kubectl --context $(KUBE_CONTEXT) apply -f infra/argocd/monitoring.yaml

prometheus-ui: ## Prometheus UI on http://localhost:9090 (Ctrl+C to stop)
> kubectl --context $(KUBE_CONTEXT) -n monitoring port-forward svc/prometheus-server 9090:80

grafana-ui: ## Grafana on http://localhost:3001 (prints the admin password, Ctrl+C to stop)
> @printf 'user: admin\npassword: '; \
>   kubectl --context $(KUBE_CONTEXT) -n monitoring get secret grafana \
>     -o jsonpath='{.data.admin-password}' | base64 -d; echo
> kubectl --context $(KUBE_CONTEXT) -n monitoring port-forward svc/grafana 3001:80

# ---------- Cleanup ----------
clean: ## Remove caches and temporary files
> find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} +
> rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml

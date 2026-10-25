# 🩺 Diabetes Prediction — End-to-End MLOps CI/CD Pipeline

> Predict a patient's diabetes risk from 5 health indicators, with a fully
> automated MLOps pipeline: from commit to production, using only free and
> open-source tools.

[![CI](https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000)

---

## 🎯 Purpose

| | |
|---|---|
| **Business problem** | Early screening of diabetes risk |
| **ML task** | Binary classification (diabetic / not diabetic) |
| **Input features** | Pregnancies, Glucose, BloodPressure, BMI, Age |
| **Dataset** | Pima Indians Diabetes Dataset |
| **Project goal** | Demonstrate a production-grade MLOps lifecycle: CI, CD, CT, monitoring |

The model itself is intentionally simple. **The focus of this project is the
pipeline around it**: reproducibility, automated testing, versioning,
deployment, monitoring and continuous retraining.

---

## 🏗️ Architecture

```
 Developer ──push/PR──▶ GitHub Actions
                         ├─ ① CI code      lint · types · security · tests
                         ├─ ② CI data      schema & quality validation
                         ├─ ③ Train        DVC pipeline + MLflow tracking
                         ├─ ④ Evaluate     quality gates + CML report on PR
                         ├─ ⑤ Register     MLflow Model Registry → Staging
                         ├─ ⑥ Build        Docker image + Trivy scan → GHCR
                         └─ ⑦ Deploy       ArgoCD (GitOps) → staging → prod
                                                   │
                                                   ▼
                                  FastAPI model server (Kubernetes)
                                                   │
                                                   ▼
                          Monitoring: Prometheus · Grafana · Evidently
                                                   │
                                  drift detected ──┴──▶ retrain (back to ③)
```

---

## 🧰 Tech Stack

| Layer | Tool |
|---|---|
| Environment & dependencies | [uv](https://github.com/astral-sh/uv) |
| Code quality | Ruff, Mypy, Bandit, pre-commit, Gitleaks |
| Testing | Pytest, pytest-cov |
| Data versioning | DVC + DagsHub |
| Experiment tracking & registry | MLflow |
| Data validation | Pandera / Great Expectations |
| CI/CD | GitHub Actions, CML |
| Serving | FastAPI |
| Containers | Docker, GHCR, Trivy |
| Deployment | GHCR, Kubernetes (kind), Kustomize, Argo CD |
| Infrastructure as Code | Terraform |
| Monitoring | Prometheus, Grafana, Evidently |

---

## 📁 Project Structure

```
.
├── .github/workflows/   # CI/CD pipelines
├── configs/             # hyperparameters and quality-gate thresholds
├── data/                # DVC-tracked data (raw/, processed/) - not in Git
├── docker/              # Dockerfile and smoke test of the API image
├── docs/adr/            # Architecture Decision Records
├── infra/
│   ├── argocd/          # Argo CD application (GitOps)
│   ├── k8s/             # Kustomize manifests (base/, overlays/local/), kind config
│   └── terraform/       # infrastructure as code
├── monitoring/          # Grafana dashboards, Evidently reports, alerts
├── notebooks/           # exploration only, never used in production
├── pipelines/           # DVC pipeline definition
├── src/
│   ├── data/            # ingestion, cleaning, validation
│   ├── features/        # feature engineering
│   ├── models/          # train, evaluate, predict
│   └── serving/         # inference API
└── tests/
    ├── unit/            # pure functions
    ├── data/            # schema and distribution tests
    ├── model/           # performance, invariance, bias tests
    └── integration/     # end-to-end API tests
```

---

## 🚀 Quick Start

**Prerequisites:** Git, [uv](https://docs.astral.sh/uv/getting-started/installation/), make

```bash
git clone https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline.git
cd Diabetes-Prediction-Mlops-cicd-pipeline
make install    # create the environment + install git hooks
make check      # run all quality checks (same as the CI)
```

Run `make` to list all available commands.

---

## 📦 Data

The dataset is versioned with [DVC](https://dvc.org/) and stored on
[DagsHub](https://dagshub.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline).
Git only tracks small pointer files (`*.dvc`); the data itself lives on the DVC remote.

| Dataset | Pointer | Documentation |
|---|---|---|
| Pima Indians Diabetes (768 rows) | `data/raw/diabetes.csv.dvc` | [Dataset card](docs/data/diabetes-dataset.md) |

**Get the data** (requires a free DagsHub account and an access token):

```bash
uv run dvc remote modify origin --local auth basic
uv run dvc remote modify origin --local user <dagshub-user>
uv run dvc remote modify origin --local password <dagshub-token>
make data-pull
```

Credentials are stored in `.dvc/config.local`, which is git-ignored.

---

## 🧪 ML Pipeline

The pipeline is defined in [`dvc.yaml`](dvc.yaml) and configured in
[`params.yaml`](params.yaml). DVC only re-runs the stages whose code, data or
parameters changed.

```
data/raw/diabetes.csv ──▶ prepare ──▶ train ──▶ evaluate ──▶ reports/metrics.json
                          (src/data/   (src/models/ (src/models/
                           prepare.py)  train.py)    evaluate.py)
```

| Stage | What it does | Output |
|---|---|---|
| `prepare` | Keep model columns, flag impossible zeros as missing, stratified 80/20 split | `data/processed/{train,test}.csv` |
| `train` | Median imputation + random forest in a single scikit-learn `Pipeline` | `models/model.joblib` |
| `evaluate` | Score the model on the held-out test set | `reports/metrics.json` |

```bash
make data-pull   # get the data (see the Data section)
make repro       # run the pipeline
make metrics     # show metrics and the diff with the last commit
make dag         # show the stages graph
```

**Current model vs baseline** (random forest, 200 trees, 154 test patients):

| Model | CV recall | Recall | Precision | F1 | ROC-AUC | Missed diabetics |
|---|---|---|---|---|---|---|
| Baseline (depth 6) | 0.5843 | 0.5926 | 0.6531 | 0.6214 | 0.8193 | 22 / 54 |
| **Current** (`class_weight: balanced`, depth 4) | **0.7663** | **0.8148** | 0.6111 | 0.6984 | 0.8202 | **10 / 54** |

Recall is the key metric for screening: a missed diabetic patient costs more
than a false alarm. Models are compared on the 5-fold cross-validated recall
of the training set; the test set is only used to confirm the final choice.

---

## 📈 Experiment Tracking & Model Registry

Every pipeline run is logged to [MLflow on DagsHub](https://dagshub.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline/experiments) when `.env` is configured (see [`.env.example`](.env.example)):
all parameters, metrics, the DVC data hash, the Git commit and the model.

```bash
cp .env.example .env          # then fill in your DagsHub user and token
make repro                    # runs the pipeline and logs the run to MLflow
uv run dvc exp run -S train.max_depth=8   # try a variant (also logged)
make register                 # register the model, promote it if it is better
```

The production model is `models:/diabetes-classifier@champion`. A new version
only becomes the champion if its cross-validated recall beats the current one.

---

## 🚀 Prediction API

The model is served by a [FastAPI](https://fastapi.tiangolo.com/) app
([`src/serving/`](src/serving/)).

| Endpoint | Description |
|---|---|
| `GET /health` | API status and version of the served model |
| `POST /predict` | Diabetes prediction for one patient |
| `GET /docs` | Interactive documentation (Swagger UI) |

```bash
make serve            # local API on http://127.0.0.1:8002 (change it with PORT=...)
curl -s -X POST localhost:8002/predict -H 'Content-Type: application/json' \
  -d '{"pregnancies": 2, "glucose": 148, "blood_pressure": 72, "bmi": 33.6, "age": 50}'
# {"diabetic":true,"probability":0.7431,"model_version":"dev"}
```

- Inputs are validated with the same plausible bounds as the training data
  ([`src/data/bounds.py`](src/data/bounds.py)): an impossible value such as
  `glucose: 0` returns `422`.
- Unknown measurements (glucose, blood pressure, BMI) are sent as `null` and
  imputed by the model pipeline, like the missing values of the training data.
- The API refuses to start if the model file is missing or was trained on
  other features.

**Docker image** ([`docker/serve.Dockerfile`](docker/serve.Dockerfile))

```bash
make docker-build     # image with the local model, tagged with the Git commit as model version
make docker-run       # run it on http://127.0.0.1:8002
make docker-smoke     # start a container and check /health, /predict, 422 and the user
```

- Multi-stage build with uv: only the main dependencies, no dev, data or tracking tools.
- Runs as a non-root user, with a Docker health check on `/health`.
- The model is baked into the image: one image is one model version, and no
  credentials are needed at runtime.

---

## ✅ Tests

| Layer | Folder | What it checks | Needs data |
|---|---|---|---|
| Unit | `tests/unit/` | config, prepare, train, evaluate, tracking, registry code | No |
| Data | `tests/data/` | pandera schemas: plausible ranges, impossible zeros, class balance | Rules: no · real data: yes |
| Model | `tests/model/` | quality gate (recall >= 0.75, ROC-AUC >= 0.75...) and behavior (obvious cases, risk factors, invariance) | Yes |
| API | `tests/integration/` | `/health`, `/predict`, input validation (422), no startup without a model | No |

```bash
make check        # lint + types + security + all tests (coverage >= 80%)
make test-unit    # fast, no data needed
make test-data    # data validation
make test-model   # model quality gate and behavioral tests
make test-api     # API integration tests
```

Tests that need the real data or the trained model are skipped when they are
not available (`make data-pull` and `make repro` provide them). The CI pulls
the data, so none of them is skipped there.

---

## 🔄 Continuous Integration

Every pull request and every push to `main` runs [`ci.yml`](.github/workflows/ci.yml)
on GitHub Actions. Each step calls the same `make` target as a developer, so a
CI failure is reproduced locally with the same command.

```
Pull request
   │
   ▼
① Lint, types, security and tests   make lint · typecheck · security · test   (no data)
   │ green
   ▼
② Data and model tests              make data-pull · pipeline-check · test    (DagsHub data)
   │ green, tested model shared as an artifact
   ▼
③ Docker image                      make docker-build · docker-smoke          (tested model)
   │ green
   ▼
④ Metrics report                    make report → CML comment on the PR       (PR only)
⑤ Deploy                            make k8s-set-image → commit on main       (main only)
```

| Job | Required to merge | What it guarantees |
|---|---|---|
| Lint, types, security and tests | Yes | Clean, typed, secure code; unit tests pass with coverage >= 80% |
| Data and model tests | Yes | `dvc.lock` matches the code and params; all 96 tests run (none skipped), including the model quality gate |
| Docker image | Yes | The image builds with the tested model, starts, answers `/health` and `/predict` and runs as non-root |
| Metrics report | No | Reviewers see the metrics and params diff against `main` in a PR comment |
| Deploy | — (main only) | The published image tag is committed to the `local` overlay for Argo CD |

**Security**
- Actions are pinned to a full commit SHA; Dependabot proposes updates weekly.
- The workflow token is read-only; only the report job can write PR comments
  and only the image job can publish packages.
- The DagsHub token is a repository secret (`DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`),
  passed through `env:` to a single step. Pull requests from forks skip the data jobs.

```bash
make pipeline-check   # fail if code, params or data changed without `make repro`
make report           # write the metrics and params diff against origin/main
```

---

## 🚢 Continuous Deployment (GitOps)

After every merge on `main`, the tested image is published and deployed
without any manual command:

```
merge on main
   ▼ CI: tests → Docker image built with the tested model and smoke tested
   ▼ publish ghcr.io/malekachouri/diabetes-api:<commit> (and :latest)
   ▼ Deploy job: commit "deploy: update api image to <commit>" on main
   ▼ Argo CD (inside the cluster) syncs infra/k8s/overlays/local from main
   ▼ rolling update: new pods ready → old pods removed
```

| Piece | Location | Role |
|---|---|---|
| Image registry | [GHCR](https://github.com/malekachouri/Diabetes-Prediction-Mlops-cicd-pipeline/pkgs/container/diabetes-api) | Public images, tagged with the commit they were built from |
| Manifests | [`infra/k8s/`](infra/k8s/) | Kustomize base (Deployment, Service) and `local` overlay (namespace, NodePort, image tag) |
| Cluster | [`infra/k8s/kind-config.yaml`](infra/k8s/kind-config.yaml) | Local Kubernetes cluster with kind |
| GitOps | [`infra/argocd/application.yaml`](infra/argocd/application.yaml) | Argo CD keeps the cluster equal to Git: auto-sync, prune, self-heal |

**Run it locally** (Docker, kind and kubectl required):

```bash
make kind-up          # create the kind cluster "diabetes"
make argocd-install   # install Argo CD (pinned version)
make argocd-app       # hand the API deployment over to Argo CD
make argocd-ui        # Argo CD UI on https://localhost:8443
curl -s localhost:30080/health
```

- Deploying a version is a commit that changes `newTag`; rolling back is a
  `git revert` of that commit.
- Rolling updates use `maxUnavailable: 0` and probes on `/health`: a version
  that cannot start never replaces the pods that are serving.
- Pods run as non-root, with a read-only root filesystem, no Linux
  capabilities and no service account token.
- The deploy job is the only one allowed to push to `main`, with a deploy key
  in the ruleset bypass list; its commit does not trigger the CI again.

---

## 🗺️ Roadmap

- [x] **Step 1** — Foundations: structure, environment, code quality, pre-commit
- [x] **Step 2** — Data versioning (DVC + DagsHub)
- [x] **Step 3** — Modular ML pipeline
- [x] **Step 4** — Experiment tracking & model registry (MLflow)
- [x] **Step 5** — ML testing & data validation
- [x] **Step 6** — Continuous Integration (GitHub Actions + CML)
- [x] **Step 7** — Model serving & containerization (FastAPI + Docker)
- [x] **Step 8** — Continuous Deployment & GitOps (GHCR + ArgoCD)
- [ ] **Step 9** — Monitoring (Prometheus + Grafana + Evidently)
- [ ] **Step 10** — Continuous Training & feedback loop

---

## 🤝 Conventions

- **Branches:** `feature/*`, `fix/*`, `chore/*` → Pull Request → `main`
- **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `chore:`, `build:`, `ci:`, `docs:`, `test:`)
- **Language:** all code, comments and documentation are written in English
- **Decisions:** every significant technical choice is recorded in [`docs/adr/`](docs/adr/)

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

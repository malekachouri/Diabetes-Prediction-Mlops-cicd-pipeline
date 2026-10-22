# 🩺 Diabetes Prediction — End-to-End MLOps CI/CD Pipeline

> Predict a patient's diabetes risk from 5 health indicators, with a fully
> automated MLOps pipeline: from commit to production, using only free and
> open-source tools.

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
| Deployment | Kubernetes (k3s), ArgoCD |
| Infrastructure as Code | Terraform |
| Monitoring | Prometheus, Grafana, Evidently |

---

## 📁 Project Structure

```
.
├── .github/workflows/   # CI/CD pipelines
├── configs/             # hyperparameters and quality-gate thresholds
├── data/                # DVC-tracked data (raw/, processed/) - not in Git
├── docker/              # Dockerfiles (train, serve)
├── docs/adr/            # Architecture Decision Records
├── infra/
│   ├── k8s/             # Kubernetes manifests (staging/, prod/)
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

## 🗺️ Roadmap

- [x] **Step 1** — Foundations: structure, environment, code quality, pre-commit
- [x] **Step 2** — Data versioning (DVC + DagsHub)
- [x] **Step 3** — Modular ML pipeline
- [x] **Step 4** — Experiment tracking & model registry (MLflow)
- [ ] **Step 5** — ML testing & data validation
- [ ] **Step 6** — Continuous Integration (GitHub Actions + CML)
- [ ] **Step 7** — Model serving & containerization (FastAPI + Docker)
- [ ] **Step 8** — Continuous Deployment & GitOps (GHCR + ArgoCD)
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

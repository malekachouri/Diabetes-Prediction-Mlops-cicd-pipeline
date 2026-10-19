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

## 🗺️ Roadmap

- [x] **Step 1** — Foundations: structure, environment, code quality, pre-commit
- [ ] **Step 2** — Data versioning (DVC + DagsHub)
- [ ] **Step 3** — Modular ML pipeline
- [ ] **Step 4** — Experiment tracking & model registry (MLflow)
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

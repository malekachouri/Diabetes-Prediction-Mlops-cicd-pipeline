# ADR-001: Development tooling choices

- **Status:** Accepted
- **Date:** 2026-10-19

## Context

This project must deliver an end-to-end MLOps pipeline using only free and
open-source tools. Before writing any ML code, we need a foundation that
guarantees:

- **Reproducibility:** the same code must produce the same environment, and
  therefore the same model, on every machine (laptop, CI, Docker).
- **Fast feedback:** errors should be caught on the developer's machine,
  before reaching the CI ("shift left").
- **Security:** no secrets, datasets or model files may be committed to Git.
- **Consistency:** the same rules must apply locally, in pre-commit and in CI.

## Decision

| Concern | Tool | Rationale |
|---|---|---|
| Python & dependency management | **uv** | Lock file with hashes, Python version pinning, dependency groups, 10-100x faster than pip |
| Linting & formatting | **Ruff** | Replaces flake8, black, isort and pyupgrade in a single fast tool |
| Static type checking | **Mypy** | Catches type errors (e.g. passing a `str` instead of a `DataFrame`) |
| Code security | **Bandit** + Ruff `S` rules | Detects insecure patterns (hardcoded passwords, unsafe `pickle`) |
| Secret detection | **Gitleaks** | Blocks commits that contain credentials |
| Testing | **Pytest** + pytest-cov | De facto standard, with coverage reporting |
| Git hooks | **pre-commit** | Runs the checks automatically on commit and push |
| Command interface | **Makefile** | Same commands for developers and the CI |
| Configuration | **pyproject.toml** | Single source of configuration for all tools |

## Alternatives considered

- **pip + requirements.txt:** no reliable lock file, no Python version
  management. Rejected for reproducibility reasons.
- **Poetry:** mature and reliable, but significantly slower than uv.
- **Conda:** useful for complex C/CUDA dependencies, which this project does
  not need. Too heavy.
- **black + flake8 + isort:** three tools to configure and keep in sync,
  where Ruff covers all of them.

## Consequences

**Positive**
- `uv sync --frozen` recreates the exact same environment everywhere.
- Most errors are caught locally in seconds instead of minutes in CI.
- The CI (step 6) only needs to call `make install-ci` and `make check`.

**Negative**
- Contributors must install `uv` and `make`.
- The Ruff version must be kept in sync between `uv.lock` and
  `.pre-commit-config.yaml`.

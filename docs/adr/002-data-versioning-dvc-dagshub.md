# ADR-002: Data versioning with DVC and DagsHub

- **Status:** Accepted
- **Date:** 2026-10-20

## Context

A model is the product of code, hyperparameters **and data**. Code and
hyperparameters are versioned in Git, but datasets are not: without data
versioning, we cannot answer "which exact data was this model trained on?",
reproduce a past training run, or explain a performance change.

Constraints:

- Git (and GitHub) is not designed for large or binary files: every version
  is stored in full, forever, and GitHub rejects files over 100 MB.
- The CI must be able to fetch the exact data version of any commit.
- The solution must be free and open source.
- Credentials must never be committed.

## Decision

- **DVC** versions the data. Git tracks small pointer files (`*.dvc`)
  containing the MD5 hash of each dataset; DVC stores the content in a
  content-addressable cache.
- **DagsHub** is the default DVC remote (free tier: 10 GB, no credit card).
- Credentials are stored per machine in `.dvc/config.local` (git-ignored),
  and as a GitHub Actions secret in the CI.
- DVC is installed in a dedicated `data` dependency group, so it is not
  shipped in production images.
- `dvc install` is **not** used, because it overwrites the pre-commit Git
  hooks. Developers run `uv run dvc checkout` / `make data-pull` explicitly.

## Alternatives considered

- **Git LFS:** simple, but limited free storage on GitHub (1 GB) and no
  ML-oriented features (pipelines, metrics, experiment comparison).
- **LakeFS:** powerful Git-like data lake, but requires running a server and
  an object store; too heavy for this project.
- **Remote on Google Drive:** free, but OAuth setup is fragile in CI.
- **Remote on S3 / GCS:** robust, but not free beyond the trial period.
- **Self-hosted MinIO:** free, but requires hosting and maintaining a server.

## Consequences

**Positive**
- Every commit references an exact, verifiable data version (MD5 hash).
- `git checkout <rev>` + `dvc checkout` restores code **and** data together.
- DagsHub also provides a hosted MLflow server, reused in step 4.

**Negative**
- Two commands per sync (`git pull` + `dvc pull`, `git push` + `dvc push`):
  forgetting the DVC one leaves data out of sync.
- Contributors need a DagsHub account and token to pull the data.
- Dependency on a third-party free tier (mitigated: DVC remotes can be
  switched to any other storage without changing the pointer files).

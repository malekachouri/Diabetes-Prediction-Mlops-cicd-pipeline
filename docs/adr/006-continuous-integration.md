# ADR-006: Continuous integration with GitHub Actions and CML

- **Status:** Accepted
- **Date:** 2026-10-25

## Context

The checks of steps 1 to 5 (lint, types, security, tests, data validation,
model quality gate) only ran on the developer's machine, through pre-commit
hooks that can be skipped or not installed. A pull request could be merged
with broken code, a degraded model, or a `dvc.lock` that no longer matches the
code. The CI must stay free, must not leak the DagsHub token, and reviewers
need to see the effect of a change on the model without running anything.

## Decision

One workflow, [`ci.yml`](../../.github/workflows/ci.yml), on every pull
request and every push to `main`, with three chained jobs:

| Job | Runs | Steps | Required |
|---|---|---|---|
| Lint, types, security and tests | always | `make install-ci`, `lint`, `typecheck`, `security`, `test` | Yes |
| Data and model tests | after job 1 | DagsHub credentials, `make data-pull`, `pipeline-check`, `test` | Yes |
| Metrics report | after job 2, pull requests only | `make report`, `cml comment update` | No |

**Same commands everywhere**
- Each workflow step calls a `make` target; `make install-ci` runs
  `uv sync --frozen`, so the CI uses exactly the versions of `uv.lock`.

**Data and model in the CI**
- `make pipeline-check` fails unless `dvc status` is empty: the pulled data,
  the code and the params must match `dvc.lock`. It also prevents a green run
  where data and model tests are silently skipped after an incomplete pull.

**Metrics report**
- `reports/metrics.json` is committed (`cache: false`), so `dvc metrics diff`
  compares the branch with the base branch from Git history, without
  retraining. CML updates a single comment per pull request.

**Security**
- Every action is pinned to a full commit SHA with the version in a comment;
  Dependabot opens a weekly pull request when a new release exists.
- `permissions: contents: read` by default; only the report job gets
  `pull-requests: write`.
- The DagsHub token is a repository secret, exposed through `env:` to the
  single step that writes `.dvc/config.local` on the ephemeral runner.
- Pull requests from forks do not receive secrets: the data and report jobs
  are skipped for them instead of failing.

**Branch protection**
- The `protect-main` ruleset requires the first two jobs to pass and the
  branch to be up to date with `main` before merging.

## Alternatives considered

- **GitLab CI / Jenkins:** the code is on GitHub and GitHub-hosted runners
  are free for public repositories; Jenkins needs a server to maintain.
- **Retrain the model in the CI:** slower and redundant; `pipeline-check`
  already proves that the committed outputs match the code, params and data.
- **Floating tags (`@v10`):** shorter, but a tag can be moved to other code or
  not exist at all (`astral-sh/setup-uv` no longer publishes major tags,
  which broke the first CI run).
- **A plain `gh pr comment` instead of CML:** possible, but CML finds and
  updates its previous comment and is the standard tool of the DVC ecosystem.

## Consequences

**Positive**
- Nothing reaches `main` without passing the code, data and model checks.
- A forgotten `make repro` or an unpushed DVC cache is caught by the CI.
- Reviewers see the metrics and params diff directly in the pull request.

**Negative**
- The CI depends on DagsHub availability and on a personal access token that
  must be rotated if it leaks or expires.
- Pinned SHAs are less readable; Dependabot pull requests must be reviewed
  and merged regularly.
- The three jobs add a few minutes to every pull request.

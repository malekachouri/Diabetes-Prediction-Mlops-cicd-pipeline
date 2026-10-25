# Architecture Decision Records

An ADR records **one** significant technical decision: the context, the
decision itself, and its consequences. ADRs are never deleted: if a decision
changes, a new ADR supersedes the old one.

| ADR | Title | Status |
|---|---|---|
| [001](001-tooling-choices.md) | Development tooling choices | Accepted |
| [002](002-data-versioning-dvc-dagshub.md) | Data versioning with DVC and DagsHub | Accepted |
| [003](003-ml-pipeline-design.md) | ML pipeline design | Accepted |
| [004](004-experiment-tracking-and-model-registry.md) | Experiment tracking and model registry with MLflow | Accepted |
| [005](005-testing-strategy.md) | Testing strategy for code, data and model | Accepted |
| [006](006-continuous-integration.md) | Continuous integration with GitHub Actions and CML | Accepted |
| [007](007-model-serving.md) | Model serving with FastAPI and Docker | Accepted |

## Template

```markdown
# ADR-XXX: Title

- **Status:** Proposed | Accepted | Superseded by ADR-YYY
- **Date:** YYYY-MM-DD

## Context
What problem are we solving? What constraints apply?

## Decision
What did we choose?

## Alternatives considered
What else was evaluated, and why was it rejected?

## Consequences
What becomes easier or harder because of this decision?
```

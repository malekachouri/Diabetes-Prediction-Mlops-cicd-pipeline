# ADR-008: Continuous deployment with GHCR, kind and Argo CD

- **Status:** Accepted
- **Date:** 2026-10-25

## Context

After step 7, every pull request builds and tests the API image, but the image
disappears at the end of the CI run and nothing is deployed. A merge on `main`
should put the tested version in service without any manual command, with a
history of what was deployed, a simple rollback, and no outage when a version
cannot start. Everything must stay free and run on a developer machine.

## Decision

**Publication**
- After a merge on `main`, the `Docker image` job pushes the image it has just
  smoke tested to GitHub Container Registry, tagged with the commit
  (`ghcr.io/malekachouri/diabetes-api:<commit>`) and `latest`. Pull requests
  never publish. The workflow `GITHUB_TOKEN` with `packages: write` on this job
  only is enough; the package is public.

**Manifests** ([`infra/k8s/`](../../infra/k8s/))
- Kustomize `base` (Deployment, Service) and a `local` overlay (namespace,
  NodePort 30080, image tag). The deployed version is the single `newTag` line
  of the overlay.
- Two replicas, rolling update with `maxUnavailable: 0`, startup, readiness and
  liveness probes on `/health`, resources from the measured usage (about
  145 MiB), non-root user, read-only root filesystem, no capabilities, no
  service account token.

**Cluster**
- A local kind cluster named `diabetes`, described in
  [`infra/k8s/kind-config.yaml`](../../infra/k8s/kind-config.yaml) with a pinned
  node image; every `kubectl` command of the Makefile passes
  `--context kind-diabetes`.

**GitOps** ([`infra/argocd/application.yaml`](../../infra/argocd/application.yaml))
- Argo CD (pinned version) runs in the cluster and syncs the `local` overlay
  of `main`, with automated sync, prune and self-heal.
- After publishing, the `Deploy` job commits the new tag to `main`
  (`make k8s-set-image`), and Argo CD rolls it out.
- The `Deploy` job pushes with a deploy key, the only actor in the bypass list
  of the `protect-main` ruleset. Its commit is excluded from the CI trigger
  (`paths-ignore`) to avoid an endless build-and-deploy loop; concurrent
  deployments are serialized and a rejected push is rebased and retried.

## Alternatives considered

- **`kubectl apply` from the CI (push model):** the CI would need credentials
  to the cluster, and the cluster state would not be described in Git. With
  Argo CD the cluster pulls its desired state and the CI needs no access to it.
- **Argo CD Image Updater:** watches the registry instead of Git; one more
  component, and without Git write-back the deployed version is not recorded
  in Git.
- **Deploying `latest`:** a moving tag cannot be rolled back and does not say
  which version runs; commit tags are immutable.
- **k3d or minikube:** k3d was the first plan; kind was chosen because it is
  the cluster used by the Kubernetes project itself and is already installed
  on the development machine.
- **A pull request for each tag update:** pull requests created with the
  workflow token do not trigger the CI, so required checks would never pass
  without a personal access token.

## Consequences

**Positive**
- A merge on `main` is deployed automatically; the Git history is the
  deployment history and a rollback is a `git revert`.
- Manual changes to the cluster are undone by Argo CD.
- A version that cannot start leaves the previous pods serving: during the
  first automated deployment the image pull hung for 13 minutes and the API
  kept answering with the previous version.

**Negative**
- A deploy key that can push to `main` is a sensitive secret; it is used by a
  single job and can be revoked from the repository settings.
- Without a shared build cache, every CI build produced a new dependency
  layer (about 117 MB compressed) that the cluster downloaded for each new
  version, and the pull sometimes stalled. The CI now reuses a BuildKit
  cache (GitHub Actions cache): the layer only changes with `uv.lock`. A
  stuck pull still needs a manual `crictl pull` and a pod restart.
- Argo CD polls Git every 3 minutes: a deployment starts up to 3 minutes
  after the tag commit, unless a refresh is requested.
- The local cluster is not production: one node, no ingress, no TLS.

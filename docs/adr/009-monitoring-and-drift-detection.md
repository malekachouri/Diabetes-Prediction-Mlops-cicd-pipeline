# ADR-009: Monitoring and data drift detection

- **Status:** Accepted
- **Date:** 2026-10-26

## Context

After step 8, every merge is deployed automatically, but nothing tells how the
service and the model behave in production. An ML model can answer `200` in a
few milliseconds and still be wrong: if the population of patients changes
(older, higher glucose), its training data no longer represents the requests,
without any error. The monitoring must run in the local kind cluster, stay
free, be declared in Git like the rest, and produce a signal that step 10 can
use to retrain the model.

## Decision

**Service and model metrics**
- The API exposes `/metrics` (`prometheus-client`): requests by endpoint and
  status, latency, predictions by class, probability histogram, measurements
  sent as `null`, served model version. Labels use the route template, never
  the raw URL, to keep the number of time series bounded.
- Prometheus (chart `prometheus-community/prometheus`, server and Pushgateway
  only) discovers the API pods through their `prometheus.io/*` annotations.
- Grafana (chart `grafana` from `grafana-community`) gets a provisioned
  Prometheus datasource and the "Diabetes API" dashboard, a JSON file in Git
  loaded by the dashboard sidecar.

**Data drift**
- The API logs every prediction as one JSON line on stdout, with the training
  column names.
- `src/monitoring/drift.py` compares the requests with the training set using
  Evidently (`DataDriftPreset`): a dataset drifts when at least half of the
  features drift, on at least 50 predictions.
- A CronJob (`drift-monitor`, every 15 minutes) runs the check in a dedicated
  image (`diabetes-drift`: main dependencies, Evidently and the training set).
  It reads the API logs through the Kubernetes API with a service account
  limited to listing pods and reading their logs, and pushes the result to the
  Pushgateway.
- Prometheus raises `DiabetesDataDrift` on drift and `DiabetesDriftCheckStale`
  when the check has not run for an hour; Grafana shows the drift per feature.

**Delivery**
- Prometheus, Grafana and the dashboards are Argo CD applications with pinned
  chart versions; the drift image is built, smoke tested and published by the
  same CI job as the API image, with the same tag.

## Alternatives considered

- **kube-prometheus-stack:** the reference stack, but its operator, CRDs and
  exporters are heavy for a local cluster; the plain chart with pod
  annotations is enough for one application.
- **Drift computed inside the API:** no extra component, but Evidently would
  be added to the production image and every pod would only see its own
  requests.
- **Logs shipped to Loki or a bucket:** durable and scalable, but more
  infrastructure; reading the pod logs is enough locally.
- **Drift check on the developer machine (`make drift-report`):** kept for
  investigation, but it only runs when someone thinks of it.

## Consequences

**Positive**
- Service health, model behavior and data drift are visible in one dashboard,
  and a drift raises an alert without anyone looking.
- The drift result is a Prometheus metric that step 10 can use as a trigger.
- The monitoring configuration is versioned and reviewed like the code.

**Negative**
- Pod logs are lost when the pods restart (every deployment): the drift check
  only sees the requests received since the last deployment.
- Metrics are not persisted (no volume) and there is no Alertmanager: alerts
  are visible in Prometheus and Grafana only.
- The drift image is large (about 1.5 GB, mostly Evidently) and contains the
  training set; a private dataset would be read from a volume or a bucket.
- Each feature test can flag a drift by chance (5% threshold); decisions rely
  on the share of drifted features and should be confirmed over time.

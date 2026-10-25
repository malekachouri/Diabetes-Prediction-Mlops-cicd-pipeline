"""Scheduled drift check, run by a Kubernetes CronJob inside the cluster.

1. Read the predictions logged by the API pods through the Kubernetes API
   (service account allowed to list pods and read their logs only).
2. Compare them with the training data (src.monitoring.drift).
3. Push the result to the Prometheus Pushgateway: Prometheus raises the alert
   and Grafana shows it.

Run with ``python -m src.monitoring.job`` (image docker/drift.Dockerfile).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import ssl
import tempfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from src.config import DEFAULT_PARAMS_PATH, load_config
from src.monitoring import drift
from src.monitoring.drift import MIN_CURRENT_ROWS, DriftResult

logger = logging.getLogger(__name__)

SERVICE_ACCOUNT_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")
PREDICTION_PREFIX = '{"event": "prediction"'
PUSH_JOB_NAME = "drift-monitor"

# Reads the body of a Kubernetes API path (injected in the tests)
ApiGet = Callable[[str], str]


def in_cluster_api_get(account_dir: Path = SERVICE_ACCOUNT_DIR) -> ApiGet:
    """Return a function calling the Kubernetes API with the pod's service account."""
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ["KUBERNETES_SERVICE_PORT"]
    token = (account_dir / "token").read_text(encoding="utf-8").strip()
    context = ssl.create_default_context(cafile=str(account_dir / "ca.crt"))

    def get(path: str) -> str:
        request = urllib.request.Request(  # noqa: S310 - in-cluster HTTPS URL
            f"https://{host}:{port}{path}", headers={"Authorization": f"Bearer {token}"}
        )
        with urllib.request.urlopen(request, context=context, timeout=30) as response:  # noqa: S310  # nosec B310
            return str(response.read().decode("utf-8"))

    return get


def collect_predictions(api_get: ApiGet, namespace: str, selector: str) -> list[str]:
    """Return the prediction lines logged by every pod matching ``selector``."""
    query = urllib.parse.urlencode({"labelSelector": selector})
    pods = json.loads(api_get(f"/api/v1/namespaces/{namespace}/pods?{query}"))["items"]
    lines: list[str] = []
    for pod in pods:
        name = pod["metadata"]["name"]
        log = api_get(f"/api/v1/namespaces/{namespace}/pods/{name}/log")
        lines += [line for line in log.splitlines() if line.startswith(PREDICTION_PREFIX)]
    logger.info("Collected %d predictions from %d pods", len(lines), len(pods))
    return lines


def push_result(
    gateway_url: str, features: list[str], current_rows: int, result: DriftResult | None
) -> None:
    """Push the drift result to the Pushgateway (``result`` is None without enough data)."""
    registry = CollectorRegistry()
    Gauge(
        "diabetes_drift_current_rows", "Predictions analyzed by the last check", registry=registry
    ).set(current_rows)
    Gauge(
        "diabetes_drift_last_run_timestamp_seconds",
        "Time of the last drift check",
        registry=registry,
    ).set_to_current_time()
    if result is not None:
        Gauge(
            "diabetes_drift_dataset_drift",
            "1 when at least half of the features have drifted",
            registry=registry,
        ).set(int(result.dataset_drift))
        Gauge("diabetes_drift_drifted_share", "Share of drifted features", registry=registry).set(
            result.drifted_share
        )
        feature_drift = Gauge(
            "diabetes_drift_feature_drifted",
            "1 when the distribution of the feature has drifted",
            ["feature"],
            registry=registry,
        )
        for feature in features:
            feature_drift.labels(feature=feature).set(int(feature in result.drifted_features))
    push_to_gateway(gateway_url, job=PUSH_JOB_NAME, registry=registry)


def run_check(
    api_get: ApiGet, gateway_url: str, namespace: str, selector: str
) -> DriftResult | None:
    """Collect, analyze and push; returns None when there are too few predictions."""
    features = load_config(DEFAULT_PARAMS_PATH).data.features
    lines = collect_predictions(api_get, namespace, selector)

    result = None
    if len(lines) < MIN_CURRENT_ROWS:
        logger.warning("Only %d predictions: drift not analyzed", len(lines))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            current = Path(tmp) / "predictions.jsonl"
            current.write_text("\n".join(lines) + "\n", encoding="utf-8")
            result = drift.run(current_path=current, report_dir=Path(tmp) / "report")
    push_result(gateway_url, features, len(lines), result)
    return result


def main(argv: list[str] | None = None) -> None:
    """Command line entry point of the CronJob."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--namespace", default=os.environ.get("POD_NAMESPACE", "diabetes"))
    parser.add_argument("--selector", default="app.kubernetes.io/name=diabetes-api")
    parser.add_argument("--pushgateway", default=os.environ.get("PUSHGATEWAY_URL", ""))
    args = parser.parse_args(argv)
    if not args.pushgateway:
        parser.error("--pushgateway or PUSHGATEWAY_URL is required")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_check(in_cluster_api_get(), args.pushgateway, args.namespace, args.selector)


if __name__ == "__main__":
    main()

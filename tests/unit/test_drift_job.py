"""Unit tests of the scheduled drift check (no cluster and no Pushgateway needed)."""

import json
from pathlib import Path
from typing import Any

import pytest
from prometheus_client import CollectorRegistry

from src.monitoring import job
from src.monitoring.drift import DriftResult

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]
PREDICTION = json.dumps({"event": "prediction", "features": {"Age": 50}})


def fake_api(logs: dict[str, str]) -> job.ApiGet:
    """Kubernetes API answering the pod list and the log of each pod."""

    def get(path: str) -> str:
        if path.endswith("/log"):
            return logs[path.split("/")[-2]]
        return json.dumps({"items": [{"metadata": {"name": name}} for name in logs]})

    return get


@pytest.fixture
def pushed(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Capture the samples pushed to the Pushgateway."""
    samples: dict[str, float] = {}

    def fake_push(url: str, job: str, registry: CollectorRegistry) -> None:
        assert job == "drift-monitor"
        for metric in registry.collect():
            for sample in metric.samples:
                labels = ",".join(sample.labels.values())
                samples[f"{sample.name}{{{labels}}}" if labels else sample.name] = sample.value

    monkeypatch.setattr(job, "push_to_gateway", fake_push)
    return samples


# ---------- Collection ----------
def test_predictions_are_collected_from_every_pod() -> None:
    api = fake_api({"pod-a": f"INFO: GET /health\n{PREDICTION}\n", "pod-b": f"{PREDICTION}\n"})

    lines = job.collect_predictions(api, "diabetes", "app.kubernetes.io/name=diabetes-api")

    assert lines == [PREDICTION, PREDICTION]


def test_in_cluster_client_uses_the_service_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "token").write_text("secret-token\n")
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("KUBERNETES_SERVICE_PORT", "443")
    monkeypatch.setattr(job.ssl, "create_default_context", lambda cafile: cafile)
    calls: list[Any] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"body"

    def fake_urlopen(request: Any, context: Any, timeout: float) -> FakeResponse:
        calls.append((request.full_url, request.headers["Authorization"], context))
        return FakeResponse()

    monkeypatch.setattr(job.urllib.request, "urlopen", fake_urlopen)

    assert job.in_cluster_api_get(tmp_path)("/api/v1/pods") == "body"
    assert calls == [
        ("https://10.0.0.1:443/api/v1/pods", "Bearer secret-token", str(tmp_path / "ca.crt"))
    ]


# ---------- Push ----------
def test_drift_result_is_pushed(pushed: dict[str, float]) -> None:
    result = DriftResult(True, ["Age", "Glucose"], 0.4, 614, 300)

    job.push_result("http://gateway", FEATURES, 300, result)

    assert pushed["diabetes_drift_dataset_drift"] == 1
    assert pushed["diabetes_drift_drifted_share"] == 0.4
    assert pushed["diabetes_drift_feature_drifted{Age}"] == 1
    assert pushed["diabetes_drift_feature_drifted{BMI}"] == 0
    assert pushed["diabetes_drift_current_rows"] == 300
    assert pushed["diabetes_drift_last_run_timestamp_seconds"] > 0


def test_without_result_only_the_volume_is_pushed(pushed: dict[str, float]) -> None:
    job.push_result("http://gateway", FEATURES, 12, None)

    assert pushed["diabetes_drift_current_rows"] == 12
    assert "diabetes_drift_dataset_drift" not in pushed


# ---------- Full check ----------
def test_too_few_predictions_are_not_analyzed(
    pushed: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(job.drift, "run", lambda **_: pytest.fail("drift must not run"))

    result = job.run_check(fake_api({"pod-a": PREDICTION}), "http://gateway", "diabetes", "x")

    assert result is None
    assert pushed["diabetes_drift_current_rows"] == 1


def test_enough_predictions_are_analyzed(
    pushed: dict[str, float], monkeypatch: pytest.MonkeyPatch
) -> None:
    analyzed: list[int] = []

    def fake_run(current_path: Path, report_dir: Path) -> DriftResult:
        analyzed.append(len(current_path.read_text().splitlines()))
        return DriftResult(False, [], 0.0, 614, analyzed[0])

    monkeypatch.setattr(job.drift, "run", fake_run)
    api = fake_api({"pod-a": "\n".join([PREDICTION] * 60)})

    result = job.run_check(api, "http://gateway", "diabetes", "x")

    assert analyzed == [60]
    assert result is not None
    assert result.dataset_drift is False
    assert pushed["diabetes_drift_dataset_drift"] == 0


# ---------- Command line ----------
def test_main_requires_a_pushgateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)

    with pytest.raises(SystemExit):
        job.main([])


def test_main_runs_the_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://gateway:9091")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.setattr(job, "in_cluster_api_get", lambda: "api")
    monkeypatch.setattr(job, "run_check", lambda *args: calls.append(args))

    job.main([])

    assert calls == [
        ("api", "http://gateway:9091", "diabetes", "app.kubernetes.io/name=diabetes-api", None, "")
    ]


def test_main_reports_to_github_when_a_token_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Any, ...]] = []
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://gateway:9091")
    monkeypatch.setenv("GITHUB_TOKEN", "github-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setattr(job, "in_cluster_api_get", lambda: "api")
    monkeypatch.setattr(job, "github_api", lambda token: f"github({token})")
    monkeypatch.setattr(job, "run_check", lambda *args: calls.append(args))

    job.main([])

    assert calls[0][-2:] == ("github(github-token)", "owner/repo")


@pytest.mark.parametrize(("dataset_drift", "opened"), [(True, 1), (False, 0)])
def test_an_issue_is_opened_only_on_drift(
    pushed: dict[str, float], monkeypatch: pytest.MonkeyPatch, dataset_drift: bool, opened: int
) -> None:
    issues: list[tuple[str, list[str]]] = []
    result = DriftResult(dataset_drift, ["Age"], 0.6 if dataset_drift else 0.2, 614, 60)
    monkeypatch.setattr(job.drift, "run", lambda **_: result)
    monkeypatch.setattr(
        job,
        "open_drift_issue",
        lambda api, repo, res, versions: issues.append((repo, sorted(set(versions)))),
    )
    line = json.dumps({"event": "prediction", "model_version": "abc1234", "features": {}})
    api = fake_api({"pod-a": "\n".join([line] * 60)})

    job.run_check(api, "http://gateway", "diabetes", "x", github=lambda *a: None, repository="o/r")

    assert issues == [("o/r", ["abc1234"])] * opened

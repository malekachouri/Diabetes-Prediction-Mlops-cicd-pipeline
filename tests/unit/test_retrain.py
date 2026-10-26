"""Unit tests of the retraining decision (no MLflow server needed)."""

import json
from pathlib import Path

import pytest

from src.models import retrain
from src.models.retrain import load_candidate_metric, should_open_pull_request, summarize


def test_candidate_metric_is_read_from_the_metrics_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"recall": 0.8, "cv_recall_mean": 0.7712}), encoding="utf-8")

    assert load_candidate_metric(path) == 0.7712


@pytest.mark.parametrize(
    ("candidate", "champion", "expected"),
    [
        (0.70, None, True),  # no champion yet: the first model is welcome
        (0.78, 0.7663, True),  # strictly better
        (0.7663, 0.7663, False),  # equal: keep the champion, no useless deployment
        (0.74, 0.7663, False),  # worse
    ],
)
def test_pull_request_only_for_a_strictly_better_candidate(
    candidate: float, champion: float | None, expected: bool
) -> None:
    assert should_open_pull_request(candidate, champion) is expected


def test_summary_shows_both_metrics_and_the_decision() -> None:
    text = summarize(0.7712, 0.7663, better=True)

    assert "| Candidate | 0.7712 |" in text
    assert "| Champion | 0.7663 |" in text
    assert "pull request is opened" in text
    assert "| Champion | none |" in summarize(0.7, None, better=True)
    assert "champion is kept" in summarize(0.7, 0.8, better=False)


def test_main_needs_the_mlflow_settings() -> None:
    with pytest.raises(SystemExit, match="MLFLOW_TRACKING_URI"):
        retrain.main([])


def test_main_writes_the_decision_for_the_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: "client")
    monkeypatch.setattr(retrain, "champion_metric", lambda client: 0.70)
    output, summary = tmp_path / "output", tmp_path / "summary.md"

    assert retrain.main(["--output", str(output), "--summary", str(summary)]) == 0

    candidate = load_candidate_metric(Path("reports/metrics.json"))  # committed metrics
    assert output.read_text() == f"better=true\ncandidate={candidate:.4f}\n"
    assert "| Champion | 0.7000 |" in summary.read_text()


def test_main_without_output_files_only_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example")
    monkeypatch.setattr("mlflow.tracking.MlflowClient", lambda: "client")
    monkeypatch.setattr(retrain, "champion_metric", lambda client: None)

    assert retrain.main([]) == 0

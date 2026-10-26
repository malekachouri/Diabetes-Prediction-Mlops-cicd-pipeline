"""Unit tests of the drift issue (fake GitHub API, no network)."""

import io
import json
import urllib.error
from typing import Any

import pytest

from src.monitoring import issues
from src.monitoring.drift import DriftResult
from src.monitoring.issues import DRIFT_LABEL, issue_body, open_drift_issue

RESULT = DriftResult(True, ["Age", "BMI", "Glucose"], 0.6, 614, 300)


class FakeGitHub:
    """Records the calls and answers like the GitHub REST API."""

    def __init__(self, open_issues: list[dict[str, Any]], label_exists: bool = False) -> None:
        self.open_issues = open_issues
        self.label_exists = label_exists
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def __call__(self, method: str, path: str, payload: dict[str, Any] | None) -> Any:
        self.calls.append((method, path, payload))
        if method == "GET":
            return self.open_issues
        if path.endswith("/labels") and self.label_exists:
            raise urllib.error.HTTPError(path, 422, "exists", {}, io.BytesIO())  # type: ignore[arg-type]
        return {"html_url": "https://github.com/o/r/issues/7"}


def test_an_issue_is_opened_when_none_is_open() -> None:
    github = FakeGitHub(open_issues=[])

    url = open_drift_issue(github, "o/r", RESULT, ["abc1234", "abc1234"])

    assert url == "https://github.com/o/r/issues/7"
    method, path, payload = github.calls[-1]
    assert (method, path) == ("POST", "/repos/o/r/issues")
    assert payload is not None
    assert payload["labels"] == [DRIFT_LABEL]
    assert payload["title"] == "Data drift detected on Age, BMI, Glucose"


def test_no_second_issue_while_one_is_open() -> None:
    github = FakeGitHub(open_issues=[{"html_url": "https://github.com/o/r/issues/3"}])

    url = open_drift_issue(github, "o/r", RESULT, ["abc1234"])

    assert url == "https://github.com/o/r/issues/3"
    assert [call[0] for call in github.calls] == ["GET"]


def test_an_existing_label_is_reused() -> None:
    github = FakeGitHub(open_issues=[], label_exists=True)

    assert open_drift_issue(github, "o/r", RESULT, []) == "https://github.com/o/r/issues/7"


def test_other_label_errors_are_raised() -> None:
    def failing(method: str, path: str, payload: dict[str, Any] | None) -> Any:
        if method == "GET":
            return []
        raise urllib.error.HTTPError(path, 403, "forbidden", {}, io.BytesIO())  # type: ignore[arg-type]

    with pytest.raises(urllib.error.HTTPError):
        open_drift_issue(failing, "o/r", RESULT, [])


def test_body_explains_the_drift_and_the_next_steps() -> None:
    body = issue_body(RESULT, ["b2", "a1", "b2"])

    assert "Age, BMI, Glucose (60% of the features)" in body
    assert "| Predictions analyzed | 300 |" in body
    assert "| Model versions | a1, b2 |" in body
    assert "Retrain" in body
    assert "| Model versions | unknown |" in issue_body(RESULT, [])


def test_github_api_sends_an_authenticated_json_request(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[Any] = []

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"html_url": "u"}'

    def fake_urlopen(request: Any, timeout: float) -> FakeResponse:
        sent.append(request)
        return FakeResponse()

    monkeypatch.setattr(issues.urllib.request, "urlopen", fake_urlopen)

    answer = issues.github_api("tok")("POST", "/repos/o/r/issues", {"title": "t"})

    assert answer == {"html_url": "u"}
    assert sent[0].full_url == "https://api.github.com/repos/o/r/issues"
    assert sent[0].get_method() == "POST"
    assert sent[0].headers["Authorization"] == "Bearer tok"
    assert json.loads(sent[0].data) == {"title": "t"}

"""Open a GitHub issue when the scheduled check detects a data drift.

Drift does not prove that the model is wrong, and retraining needs new labeled
data: the issue asks a human to collect it, then to run the Retrain workflow.
Only one issue is open at a time (label ``data-drift``): the check runs every
15 minutes and must not open a new issue each time.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from typing import Any

from src.monitoring.drift import DriftResult

logger = logging.getLogger(__name__)

GITHUB_API_URL = "https://api.github.com"
DRIFT_LABEL = "data-drift"

# Calls the GitHub REST API: (method, path, JSON payload or None) -> decoded JSON
GitHubApi = Callable[[str, str, dict[str, Any] | None], Any]


def github_api(token: str) -> GitHubApi:
    """Return a function calling the GitHub REST API with ``token``."""

    def call(method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        request = urllib.request.Request(  # noqa: S310 - fixed HTTPS API URL
            f"{GITHUB_API_URL}{path}",
            data=None if payload is None else json.dumps(payload).encode(),
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310  # nosec B310
            return json.loads(response.read().decode("utf-8") or "null")

    return call


def issue_body(result: DriftResult, model_versions: Iterable[str]) -> str:
    """Markdown body of the drift issue."""
    drifted = ", ".join(result.drifted_features)
    versions = ", ".join(sorted(set(model_versions))) or "unknown"
    return (
        "The scheduled drift check found that the production requests no longer "
        "follow the distribution of the training data.\n\n"
        "| | |\n|---|---|\n"
        f"| Drifted features | {drifted} ({result.drifted_share:.0%} of the features) |\n"
        f"| Predictions analyzed | {result.current_rows} |\n"
        f"| Training rows (reference) | {result.reference_rows} |\n"
        f"| Model versions | {versions} |\n\n"
        "Drift does not prove that the model is wrong, but its training data no "
        "longer represents these patients.\n\n"
        "**Next steps**\n"
        "1. Collect new labeled data (patients with their confirmed diagnosis) "
        "from the current population.\n"
        "2. Add it as a new version of `data/raw` with DVC, in a pull request.\n"
        "3. Run the **Retrain** workflow: the candidate is proposed only if it "
        "beats the champion.\n\n"
        "Close this issue once the model is retrained or the drift is explained. "
        "Opened by the `drift-monitor` CronJob."
    )


def open_drift_issue(
    api: GitHubApi, repository: str, result: DriftResult, model_versions: Iterable[str]
) -> str:
    """Open the drift issue unless one is already open; return its URL."""
    open_issues = api("GET", f"/repos/{repository}/issues?labels={DRIFT_LABEL}&state=open", None)
    if open_issues:
        url = str(open_issues[0]["html_url"])
        logger.info("A drift issue is already open: %s", url)
        return url

    try:
        api(
            "POST",
            f"/repos/{repository}/labels",
            {"name": DRIFT_LABEL, "color": "d93f0b", "description": "Data drift detected"},
        )
    except urllib.error.HTTPError as error:
        if error.code != 422:  # 422: the label already exists
            raise

    issue = api(
        "POST",
        f"/repos/{repository}/issues",
        {
            "title": f"Data drift detected on {', '.join(result.drifted_features)}",
            "body": issue_body(result, model_versions),
            "labels": [DRIFT_LABEL],
        },
    )
    url = str(issue["html_url"])
    logger.info("Opened the drift issue %s", url)
    return url

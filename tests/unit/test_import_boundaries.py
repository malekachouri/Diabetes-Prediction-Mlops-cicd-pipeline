"""Each Docker image installs only some dependency groups: check the imports.

The API image has the main dependencies only, the drift check image adds the
"monitoring" group. Locally and in the CI every group is installed, so a wrong
import (e.g. pandera through src.data.prepare) would only fail in the cluster.
"""

import subprocess
import sys

import pytest

# Packages of the groups missing from each image
NOT_IN_API_IMAGE = ["pandera", "dvc", "mlflow", "evidently", "pytest"]
NOT_IN_DRIFT_IMAGE = ["pandera", "dvc", "mlflow", "pytest"]


def import_without(module: str, missing: list[str]) -> subprocess.CompletedProcess[str]:
    """Import ``module`` in a new interpreter where the ``missing`` packages cannot load."""
    code = (
        "import sys\n"
        f"for name in {missing!r}:\n"
        "    sys.modules[name] = None  # any import of it raises ImportError\n"
        f"import {module}\n"
    )
    return subprocess.run(  # noqa: S603 - fixed command, no user input
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    ("module", "missing"),
    [
        ("src.serving.app", NOT_IN_API_IMAGE),
        ("src.monitoring.job", NOT_IN_DRIFT_IMAGE),
    ],
)
def test_image_entry_point_imports_only_installed_packages(module: str, missing: list[str]) -> None:
    result = import_without(module, missing)

    assert result.returncode == 0, result.stderr

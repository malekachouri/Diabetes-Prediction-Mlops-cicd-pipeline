"""Smoke test : vérifie que la structure du projet est importable."""

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    ["src", "src.data", "src.features", "src.models", "src.serving"],
)
def test_package_is_importable(module: str) -> None:
    assert importlib.import_module(module) is not None

"""Backend-free Muse registration contract used by the TDD cycle."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_muse_glimmer_aliases_and_adapter_package_exist():
    registry = runpy.run_path(ROOT / "src" / "edge0" / "registry.py")

    assert registry["TYPE_ALIASES"]["muse_glimmer"] == (
        "muse-glimmer:30b-mlx")
    assert registry["TYPE_ALIASES"]["muse_glimmer_text"] == (
        "muse-glimmer:30b-mlx")
    assert (
        ROOT / "src" / "edge0" / "models" / "muse_glimmer_30b_mlx"
        / "__init__.py"
    ).is_file()


def test_muse_glimmer_fetch_default_and_dependency_boundary():
    fetch = runpy.run_path(ROOT / "scripts" / "fetch_models.py")
    tier = "muse-glimmer:30b-mlx"

    assert fetch["TIER_REPOS"][tier] == (
        "MUSE_GLIMMER_30B_REPO",
        "mlx-community/Muse-Glimmer-30B-4bit",
    )
    assert fetch["TIER_DIRS"][tier] == "muse-glimmer-30b-mlx"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mlx-vlm' not in pyproject.lower()

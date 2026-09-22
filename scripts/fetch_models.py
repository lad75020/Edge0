#!/usr/bin/env python3
"""Download a tier's model directory and apply required preparation.

The Edge0 release repos co-locate the base checkpoint and trained LoRA /
prerouter adapters. Compatible upstream MLX repos provide their own complete
snapshots; Gemma 4 and Ornith are automatically converted to Edge0's text-only
layouts after download. Each tier ends as a model directory accepted by ``edge0 demo`` /
``edge0 serve``.

Requirements:
    pip install 'edge0[fetch]'        # or: pip install huggingface_hub

Repo ids default to the official releases (override via the environment
if you mirror them):

    export EDGE0_35B_REPO=Edge0/Edge0-35B-A3B-preview   # default
    export EDGE0_8B_REPO=Edge0/Edge0-8B-A1B-preview   # default
    export GEMMA4_31B_MLX_REPO=mlx-community/gemma-4-31b-4bit
    export MUSE_GLIMMER_30B_REPO=mlx-community/Muse-Glimmer-30B-4bit
    export ORNITH_35B_MLX_REPO=mlx-community/Ornith-1.0-35B-4bit
    export QWEN38_27B_MLX_REPO=mlx-community/Qwen3.8-27B-4bit  # default

Usage:
    python scripts/fetch_models.py --tier edge0-35b --target-dir models
    python scripts/fetch_models.py --tier gemma-4:31b-mlx --target-dir models
    python scripts/fetch_models.py --tier muse-glimmer:30b-mlx --target-dir models
    python scripts/fetch_models.py --tier ornith:35b-mlx --target-dir models
    python scripts/fetch_models.py --tier qwen3.8:27b-mlx --target-dir models
    python scripts/fetch_models.py --tier all --target-dir models

Afterwards point the tier names at what you downloaded:

    export EDGE0_35B_MODEL=$PWD/models/edge0-35b
    edge0 demo edge0-35b
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

TIER_REPOS = {
    "edge0-35b": ("EDGE0_35B_REPO", "Edge0/Edge0-35B-A3B-preview"),
    "edge0-8b": ("EDGE0_8B_REPO", "Edge0/Edge0-8B-A1B-preview"),
    "gemma-4:31b-mlx": (
        "GEMMA4_31B_MLX_REPO",
        "mlx-community/gemma-4-31b-4bit",
    ),
    "muse-glimmer:30b-mlx": (
        "MUSE_GLIMMER_30B_REPO",
        "mlx-community/Muse-Glimmer-30B-4bit",
    ),
    "ornith:35b-mlx": (
        "ORNITH_35B_MLX_REPO",
        "mlx-community/Ornith-1.0-35B-4bit",
    ),
    "qwen3.8:27b-mlx": (
        "QWEN38_27B_MLX_REPO", "mlx-community/Qwen3.8-27B-4bit"),
}

TIER_DIRS = {
    tier: tier.replace(":", "-") for tier in TIER_REPOS
}


def _repo_for(tier: str) -> str:
    env, default = TIER_REPOS[tier]
    repo = os.environ.get(env, default)
    return repo


def _prepare_ornith_checkpoint(model_dir: Path, *, backup: bool) -> None:
    """Convert the upstream VLM snapshot into Edge0's text-only layout."""
    try:
        from scripts.prepare_ornith_checkpoint import prepare_ornith_checkpoint
    except ImportError:  # direct ``python scripts/fetch_models.py``
        from prepare_ornith_checkpoint import prepare_ornith_checkpoint

    prepare_ornith_checkpoint(model_dir, backup=backup)


def _prepare_gemma4_checkpoint(model_dir: Path, *, backup: bool) -> None:
    """Convert the upstream VLM snapshot into Edge0's text-only layout."""
    try:
        from scripts.prepare_gemma4_checkpoint import prepare_gemma4_checkpoint
    except ImportError:  # direct ``python scripts/fetch_models.py``
        from prepare_gemma4_checkpoint import prepare_gemma4_checkpoint

    prepare_gemma4_checkpoint(model_dir, backup=backup)


def _download(tier: str, target_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "[fetch] huggingface_hub is required: pip install 'edge0[fetch]' "
            "or pip install huggingface_hub")
    repo = _repo_for(tier)
    dest = target_dir / TIER_DIRS[tier]
    print(f"[fetch] {tier} <- {repo}  (-> {dest})", flush=True)
    snapshot_download(repo_id=repo, local_dir=str(dest))
    if tier == "gemma-4:31b-mlx":
        _prepare_gemma4_checkpoint(dest, backup=False)
    elif tier == "ornith:35b-mlx":
        # This directory was just downloaded, so retaining another ~20 GB
        # source copy would be wasteful. Rewrites still use same-directory
        # temporary files and atomic replacement.
        _prepare_ornith_checkpoint(dest, backup=False)
    print(f"[fetch] done: {dest}\n"
          f"        export {TIER_REPOS[tier][0].replace('_REPO','_MODEL')}"
          f"={dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", required=True,
                    choices=list(TIER_REPOS) + ["all"])
    ap.add_argument("--target-dir", default="models",
                    help="directory to write <tier>/ under (default: models)")
    args = ap.parse_args()

    target = Path(args.target_dir).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    tiers = list(TIER_REPOS) if args.tier == "all" else [args.tier]
    for tier in tiers:
        _download(tier, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

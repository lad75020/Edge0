#!/usr/bin/env python3
"""Prepare an Ornith-1.0-35B-4bit snapshot for text-only Edge0 use.

The transform validates and preserves the upstream Qwen3.5-MoE model types,
strips unsupported vision tensors, updates safetensors index metadata, and
writes ``edge0_model_name=ornith:35b-mlx`` for unambiguous path detection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:  # package import in tests / tools
    from scripts.strip_vision_weights import (
        StripResult,
        _backup_path,
        _install_staged,
        _stage_json,
        strip_vision_weights,
    )
except ImportError:  # direct ``python scripts/prepare_ornith_checkpoint.py``
    from strip_vision_weights import (  # type: ignore[no-redef]
        StripResult,
        _backup_path,
        _install_staged,
        _stage_json,
        strip_vision_weights,
    )


EDGE0_MODEL_NAME = "ornith:35b-mlx"


def _validated_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("model_type") != "qwen3_5_moe":
        raise ValueError(
            "Ornith preparation requires model_type='qwen3_5_moe'")
    text_type = (config.get("text_config") or {}).get("model_type")
    if text_type != "qwen3_5_moe_text":
        raise ValueError(
            "Ornith preparation requires "
            "text_config.model_type='qwen3_5_moe_text'")
    marker = config.get("edge0_model_name")
    if marker not in (None, "", EDGE0_MODEL_NAME):
        raise ValueError(
            f"checkpoint already has edge0_model_name={marker!r}")
    return config


def prepare_ornith_checkpoint(
        model_dir: str | Path, *, backup: bool = True) -> StripResult:
    directory = Path(model_dir)
    config_path = directory / "config.json"
    config = _validated_config(config_path)

    config_backup = _backup_path(config_path, "ornith")
    if (backup and config.get("edge0_model_name") != EDGE0_MODEL_NAME
            and config_backup.exists()):
        raise FileExistsError(
            f"refusing to overwrite backup {config_backup}")

    report = strip_vision_weights(directory, backup=backup)
    if config.get("edge0_model_name") == EDGE0_MODEL_NAME:
        return report

    config["edge0_model_name"] = EDGE0_MODEL_NAME
    temporary = _stage_json(config_path, config)
    try:
        _install_staged(
            temporary, config_path, backup=backup, tag="ornith")
    finally:
        temporary.unlink(missing_ok=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not retain the original shards (for a fresh download)",
    )
    args = parser.parse_args(argv)
    prepare_ornith_checkpoint(args.model_dir, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Prepare gemma-4-31b-4bit for text-only Edge0 generation.

The transform validates the upstream Gemma model types, removes all indexed
``vision_tower.*`` and ``embed_vision.*`` tensors, updates safetensors index
metadata, writes the explicit Edge0 marker, and installs Edge0's attributed
canonical text chat template. It is safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:  # package import in tests / tools
    from scripts.strip_vision_weights import (
        StripResult,
        _backup_path,
        _install_staged,
        _stage_json,
        _temporary_path,
        strip_checkpoint_weights,
    )
except ImportError:  # direct ``python scripts/prepare_gemma4_checkpoint.py``
    from strip_vision_weights import (  # type: ignore[no-redef]
        StripResult,
        _backup_path,
        _install_staged,
        _stage_json,
        _temporary_path,
        strip_checkpoint_weights,
    )


EDGE0_MODEL_NAME = "gemma-4:31b-mlx"
DROP_PREFIXES = ("vision_tower.", "embed_vision.")
CHAT_TEMPLATE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src" / "edge0" / "models" / "gemma4_31b_mlx"
    / "chat_template.jinja"
)


def _validated_config(config_path: Path) -> dict:
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("model_type") != "gemma4":
        raise ValueError("Gemma 4 preparation requires model_type='gemma4'")
    text_type = (config.get("text_config") or {}).get("model_type")
    if text_type != "gemma4_text":
        raise ValueError(
            "Gemma 4 preparation requires "
            "text_config.model_type='gemma4_text'")
    marker = config.get("edge0_model_name")
    if marker not in (None, "", EDGE0_MODEL_NAME):
        raise ValueError(
            f"checkpoint already has edge0_model_name={marker!r}")
    return config


def _install_template(target: Path, *, backup: bool) -> None:
    payload = CHAT_TEMPLATE_SOURCE.read_bytes()
    if target.exists() and target.read_bytes() == payload:
        return
    if backup and target.exists() and _backup_path(target, "gemma4").exists():
        raise FileExistsError(
            f"refusing to overwrite backup {_backup_path(target, 'gemma4')}")
    temporary = _temporary_path(target)
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            _install_staged(
                temporary, target, backup=backup, tag="gemma4")
        else:
            os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_gemma4_checkpoint(
        model_dir: str | Path, *, backup: bool = True) -> StripResult:
    directory = Path(model_dir)
    config_path = directory / "config.json"
    config = _validated_config(config_path)
    template_path = directory / "chat_template.jinja"

    config_backup = _backup_path(config_path, "gemma4")
    if (backup and config.get("edge0_model_name") != EDGE0_MODEL_NAME
            and config_backup.exists()):
        raise FileExistsError(
            f"refusing to overwrite backup {config_backup}")
    if (backup and template_path.exists()
            and template_path.read_bytes() != CHAT_TEMPLATE_SOURCE.read_bytes()
            and _backup_path(template_path, "gemma4").exists()):
        raise FileExistsError(
            "refusing to overwrite backup "
            f"{_backup_path(template_path, 'gemma4')}")

    report = strip_checkpoint_weights(
        directory,
        drop_prefixes=DROP_PREFIXES,
        backup=backup,
        backup_tag="gemma4",
    )
    if config.get("edge0_model_name") != EDGE0_MODEL_NAME:
        config["edge0_model_name"] = EDGE0_MODEL_NAME
        temporary = _stage_json(config_path, config)
        try:
            _install_staged(
                temporary, config_path, backup=backup, tag="gemma4")
        finally:
            temporary.unlink(missing_ok=True)
    _install_template(template_path, backup=backup)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="do not retain original shards (for a fresh download)",
    )
    args = parser.parse_args(argv)
    prepare_gemma4_checkpoint(args.model_dir, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

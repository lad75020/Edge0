#!/usr/bin/env python3
"""Safely strip unsupported vision tensors from a sharded checkpoint.

The transformer streams retained tensor payloads into same-directory
temporary files, then atomically replaces each affected shard and the
index. It never materializes a shard in memory. By default the originals
are retained as ``*.bak_vision``; ``--no-backup`` is intended for freshly
downloaded snapshots where keeping another checkpoint-sized copy is wasteful.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


_COPY_CHUNK_BYTES = 16 * 1024 * 1024
_VISION_PREFIXES = ("vision_tower.", "model.visual.", "visual.")


@dataclass(frozen=True)
class StripResult:
    removed_tensors: int
    removed_bytes: int
    remaining_tensors: int
    total_size: int


def read_shard(path: Path) -> tuple[dict, int]:
    """Return a safetensors header and the absolute payload offset."""
    with path.open("rb") as stream:
        raw_length = stream.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"invalid safetensors header in {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        header = json.loads(stream.read(header_length))
        data_start = 8 + header_length
    return header, data_start


def _matches_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    return key.startswith(prefixes)


def _tensor_entries(header: dict) -> dict[str, dict]:
    return {
        key: value for key, value in header.items()
        if isinstance(value, dict) and "data_offsets" in value
    }


def _entry_size(entry: dict) -> int:
    start, end = entry["data_offsets"]
    return int(end) - int(start)


def _temporary_path(target: Path) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    return Path(raw_path)


def _copy_bytes(source: BinaryIO, target: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        block = source.read(min(remaining, _COPY_CHUNK_BYTES))
        if not block:
            raise EOFError("safetensors payload ended before data_offsets")
        target.write(block)
        remaining -= len(block)


def _stage_rewritten_shard(path: Path, drop: set[str]) -> Path:
    header, data_start = read_shard(path)
    tensors = _tensor_entries(header)
    retained = sorted(
        (item for item in tensors.items() if item[0] not in drop),
        key=lambda item: item[1]["data_offsets"][0],
    )

    new_header = {
        key: value for key, value in header.items() if key not in tensors
    }
    offset = 0
    for key, entry in retained:
        rewritten = dict(entry)
        size = _entry_size(entry)
        rewritten["data_offsets"] = [offset, offset + size]
        new_header[key] = rewritten
        offset += size

    header_bytes = json.dumps(
        new_header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    temporary = _temporary_path(path)
    try:
        with path.open("rb") as source, temporary.open("wb") as target:
            target.write(struct.pack("<Q", len(header_bytes)))
            target.write(header_bytes)
            for _key, entry in retained:
                start, end = entry["data_offsets"]
                source.seek(data_start + int(start))
                _copy_bytes(source, target, int(end) - int(start))
            target.flush()
            os.fsync(target.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _stage_json(path: Path, payload: dict) -> Path:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _backup_path(path: Path, tag: str) -> Path:
    return path.with_suffix(path.suffix + f".bak_{tag}")


def _install_staged(temporary: Path, target: Path, *, backup: bool,
                    tag: str) -> None:
    _install_staged_group([(temporary, target)], backup=backup, tag=tag)


def _install_staged_group(
        replacements: list[tuple[Path, Path]], *, backup: bool,
        tag: str) -> None:
    """Install a set of staged files and roll the whole set back on error."""
    if backup:
        existing = [
            _backup_path(target, tag) for _temporary, target in replacements
            if _backup_path(target, tag).exists()
        ]
        if existing:
            raise FileExistsError(
                f"refusing to overwrite backup {existing[0]}")

    installed: list[tuple[Path, Path]] = []
    transient_rollbacks: list[Path] = []
    try:
        for temporary, target in replacements:
            if backup:
                original = _backup_path(target, tag)
            else:
                original = _temporary_path(target)
                original.unlink()
                transient_rollbacks.append(original)
            os.replace(target, original)
            try:
                os.replace(temporary, target)
            except Exception:
                os.replace(original, target)
                raise
            installed.append((target, original))
    except Exception:
        for target, original in reversed(installed):
            target.unlink(missing_ok=True)
            os.replace(original, target)
        raise
    else:
        if not backup:
            for _target, original in installed:
                original.unlink(missing_ok=True)
    finally:
        for original in transient_rollbacks:
            original.unlink(missing_ok=True)


def strip_checkpoint_weights(
        model_dir: str | Path, *, drop_prefixes: tuple[str, ...],
        backup: bool = True, backup_tag: str = "weights") -> StripResult:
    """Drop selected indexed tensors and keep shard/index metadata coherent."""
    directory = Path(model_dir)
    index_path = directory / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as stream:
        index = json.load(stream)
    weight_map = dict(index["weight_map"])
    drop_keys = {
        key for key in weight_map if _matches_prefix(key, drop_prefixes)
    }

    sizes: dict[str, int] = {}
    for shard_name in sorted(set(weight_map.values())):
        header, _data_start = read_shard(directory / shard_name)
        tensors = _tensor_entries(header)
        mapped = {key for key, value in weight_map.items() if value == shard_name}
        missing = mapped - tensors.keys()
        if missing:
            sample = ", ".join(sorted(missing)[:3])
            raise ValueError(
                f"index maps tensors missing from {shard_name}: {sample}")
        sizes.update({key: _entry_size(tensors[key]) for key in mapped})

    removed_bytes = sum(sizes[key] for key in drop_keys)
    retained_map = {
        key: shard for key, shard in weight_map.items() if key not in drop_keys
    }
    total_size = sum(sizes[key] for key in retained_map)
    metadata = dict(index.get("metadata") or {})
    metadata["total_size"] = total_size
    rewritten_index = dict(index)
    rewritten_index["metadata"] = metadata
    rewritten_index["weight_map"] = retained_map

    affected = {
        shard_name: {key for key in drop_keys
                     if weight_map[key] == shard_name}
        for shard_name in sorted({weight_map[key] for key in drop_keys})
    }
    staged_shards: dict[str, Path] = {}
    staged_index: Path | None = None
    try:
        for shard_name, drop in affected.items():
            staged_shards[shard_name] = _stage_rewritten_shard(
                directory / shard_name, drop)
        current_total_size = (index.get("metadata") or {}).get("total_size")
        if drop_keys or current_total_size != total_size:
            staged_index = _stage_json(index_path, rewritten_index)

        replacements = [
            (temporary, directory / shard_name)
            for shard_name, temporary in staged_shards.items()
        ]
        if staged_index is not None:
            replacements.append((staged_index, index_path))
        _install_staged_group(
            replacements, backup=backup, tag=backup_tag)

        for shard_name in staged_shards:
            target = directory / shard_name
            print(
                f"{shard_name}: dropped {len(affected[shard_name])} tensors, "
                f"new size {target.stat().st_size / 1e9:.2f} GB")
    finally:
        for temporary in staged_shards.values():
            temporary.unlink(missing_ok=True)
        if staged_index is not None:
            staged_index.unlink(missing_ok=True)

    print(
        f"index: {len(retained_map)} keys remain "
        f"({len(drop_keys)} selected keys / {removed_bytes} bytes removed); "
        f"total_size={total_size}")
    return StripResult(
        removed_tensors=len(drop_keys),
        removed_bytes=removed_bytes,
        remaining_tensors=len(retained_map),
        total_size=total_size,
    )


def strip_vision_weights(
        model_dir: str | Path, *, backup: bool = True) -> StripResult:
    """Remove the established vision prefixes used by existing adapters."""
    return strip_checkpoint_weights(
        model_dir,
        drop_prefixes=_VISION_PREFIXES,
        backup=backup,
        backup_tag="vision",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    strip_vision_weights(args.model_dir, backup=not args.no_backup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

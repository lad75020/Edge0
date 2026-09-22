"""Safe preparation of the text-only Ornith checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file


def _write_source_checkpoint(model_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    model_dir.mkdir()
    language = {
        "language_model.model.embed_tokens.weight": np.arange(
            8, dtype=np.float32),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (
            np.arange(6, dtype=np.uint16)),
    }
    vision = {
        f"vision_tower.blocks.{i}.weight": np.array([i % 256], dtype=np.uint8)
        for i in range(333)
    }
    shard_one = {next(iter(language)): next(iter(language.values())), **vision}
    second_key = list(language)[1]
    shard_two = {second_key: language[second_key]}
    save_file(shard_one, model_dir / "model-00001-of-00002.safetensors")
    save_file(shard_two, model_dir / "model-00002-of-00002.safetensors")

    weight_map = {
        key: "model-00001-of-00002.safetensors" for key in shard_one
    }
    weight_map.update({
        key: "model-00002-of-00002.safetensors" for key in shard_two
    })
    total_size = sum(array.nbytes for array in (*language.values(), *vision.values()))
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size, "source": "fixture"},
        "weight_map": weight_map,
    }), encoding="utf-8")
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5_moe",
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
        },
    }), encoding="utf-8")
    return language, sum(array.nbytes for array in language.values())


def test_prepare_ornith_strips_333_vision_tensors_and_marks_checkpoint(
        tmp_path):
    from scripts.prepare_ornith_checkpoint import prepare_ornith_checkpoint

    model_dir = tmp_path / "source"
    language, expected_size = _write_source_checkpoint(model_dir)

    report = prepare_ornith_checkpoint(model_dir, backup=False)

    assert report.removed_tensors == 333
    assert report.removed_bytes == 333
    assert report.remaining_tensors == 2

    config = json.loads((model_dir / "config.json").read_text())
    assert config["edge0_model_name"] == "ornith:35b-mlx"
    assert config["model_type"] == "qwen3_5_moe"
    assert config["text_config"]["model_type"] == "qwen3_5_moe_text"

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"]["total_size"] == expected_size
    assert index["metadata"]["source"] == "fixture"
    assert set(index["weight_map"]) == set(language)

    for key, shard_name in index["weight_map"].items():
        tensors = load_file(model_dir / shard_name)
        np.testing.assert_array_equal(tensors[key], language[key])
        assert all(not name.startswith("vision_tower") for name in tensors)

    assert not list(model_dir.glob("*.bak_*"))
    assert not list(model_dir.glob(".*.tmp"))


def test_prepare_ornith_is_idempotent(tmp_path):
    from scripts.prepare_ornith_checkpoint import prepare_ornith_checkpoint

    model_dir = tmp_path / "source"
    _write_source_checkpoint(model_dir)
    prepare_ornith_checkpoint(model_dir, backup=False)

    report = prepare_ornith_checkpoint(model_dir, backup=False)

    assert report.removed_tensors == 0
    assert report.remaining_tensors == 2
    config = json.loads((model_dir / "config.json").read_text())
    assert config["edge0_model_name"] == "ornith:35b-mlx"

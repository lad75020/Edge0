"""Safe preparation of the text-only Gemma 4 checkpoint."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from jinja2 import Environment
from safetensors.numpy import load_file, save_file


def _write_source_checkpoint(model_dir: Path) -> tuple[dict[str, np.ndarray], int]:
    model_dir.mkdir()
    language = {
        "language_model.model.embed_tokens.weight": np.arange(
            8, dtype=np.float32),
        "language_model.model.layers.0.self_attn.q_proj.weight": np.arange(
            6, dtype=np.uint16),
    }
    vision = {
        f"vision_tower.encoder.layers.{i}.weight": np.array(
            [i % 256], dtype=np.uint8)
        for i in range(355)
    }
    embed_vision = {
        f"embed_vision.embedding_projection.{name}": np.array(
            [index], dtype=np.uint8)
        for index, name in enumerate(("weight", "scales", "biases"))
    }
    shard_one = {
        next(iter(language)): next(iter(language.values())),
        **vision,
        **embed_vision,
    }
    second_key = list(language)[1]
    shard_two = {second_key: language[second_key]}
    save_file(shard_one, model_dir / "model-00001-of-00002.safetensors")
    save_file(shard_two, model_dir / "model-00002-of-00002.safetensors")

    weight_map = {
        key: "model-00001-of-00002.safetensors" for key in shard_one
    }
    weight_map[second_key] = "model-00002-of-00002.safetensors"
    total_size = sum(
        array.nbytes
        for array in (*language.values(), *vision.values(), *embed_vision.values())
    )
    (model_dir / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": total_size, "source": "fixture"},
        "weight_map": weight_map,
    }), encoding="utf-8")
    (model_dir / "config.json").write_text(json.dumps({
        "model_type": "gemma4",
        "text_config": {
            "model_type": "gemma4_text",
            "num_hidden_layers": 60,
        },
    }), encoding="utf-8")
    return language, sum(array.nbytes for array in language.values())


def test_prepare_gemma4_strips_358_tensors_marks_and_installs_template(
        tmp_path):
    from scripts.prepare_gemma4_checkpoint import prepare_gemma4_checkpoint

    model_dir = tmp_path / "source"
    language, expected_size = _write_source_checkpoint(model_dir)

    report = prepare_gemma4_checkpoint(model_dir, backup=False)

    assert report.removed_tensors == 358
    assert report.removed_bytes == 358
    assert report.remaining_tensors == 2
    assert report.total_size == expected_size

    config = json.loads((model_dir / "config.json").read_text())
    assert config["edge0_model_name"] == "gemma-4:31b-mlx"
    assert config["model_type"] == "gemma4"
    assert config["text_config"]["model_type"] == "gemma4_text"

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text())
    assert index["metadata"] == {
        "total_size": expected_size,
        "source": "fixture",
    }
    assert set(index["weight_map"]) == set(language)
    for key, shard_name in index["weight_map"].items():
        tensors = load_file(model_dir / shard_name)
        np.testing.assert_array_equal(tensors[key], language[key])
        assert all(not name.startswith("vision_tower.") for name in tensors)
        assert all(not name.startswith("embed_vision.") for name in tensors)

    template = (model_dir / "chat_template.jinja").read_text(encoding="utf-8")
    assert "Google Gemma 4 Canonical Chat Template" in template
    assert "{{- bos_token -}}" in template
    assert "<|channel>thought" in template
    assert not list(model_dir.glob("*.bak_*"))
    assert not list(model_dir.glob(".*.tmp"))


def test_prepare_gemma4_is_idempotent(tmp_path):
    from scripts.prepare_gemma4_checkpoint import prepare_gemma4_checkpoint

    model_dir = tmp_path / "source"
    _write_source_checkpoint(model_dir)
    first = prepare_gemma4_checkpoint(model_dir, backup=False)
    config_before = (model_dir / "config.json").read_bytes()
    index_before = (model_dir / "model.safetensors.index.json").read_bytes()
    template_before = (model_dir / "chat_template.jinja").read_bytes()

    second = prepare_gemma4_checkpoint(model_dir, backup=False)

    assert first.removed_tensors == 358
    assert second.removed_tensors == 0
    assert second.remaining_tensors == 2
    assert (model_dir / "config.json").read_bytes() == config_before
    assert (model_dir / "model.safetensors.index.json").read_bytes() == index_before
    assert (model_dir / "chat_template.jinja").read_bytes() == template_before


def test_bundled_template_renders_one_bos_and_gemma_generation_suffix():
    template_path = (
        Path(__file__).resolve().parents[1]
        / "src" / "edge0" / "models" / "gemma4_31b_mlx"
        / "chat_template.jinja"
    )
    template = Environment().from_string(template_path.read_text())

    rendered = template.render(
        messages=[{"role": "user", "content": "Hello"}],
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token="<bos>",
    )

    assert rendered.count("<bos>") == 1
    assert "<|turn>user\nHello<turn|>" in rendered
    assert rendered.endswith(
        "<|turn>model\n<|channel>thought\n<channel|>")


def test_backend_free_registry_and_fetch_metadata():
    root = Path(__file__).resolve().parents[1]
    registry = runpy.run_path(root / "src" / "edge0" / "registry.py")
    fetch = runpy.run_path(root / "scripts" / "fetch_models.py")
    tier = "gemma-4:31b-mlx"

    assert registry["TYPE_ALIASES"]["gemma4"] == tier
    assert registry["TYPE_ALIASES"]["gemma4_text"] == tier
    assert fetch["TIER_REPOS"][tier] == (
        "GEMMA4_31B_MLX_REPO",
        "mlx-community/gemma-4-31b-4bit",
    )
    assert fetch["TIER_DIRS"][tier] == "gemma-4-31b-mlx"
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mlx-vlm' not in pyproject.lower()


def test_fetch_dispatches_gemma_preparation_without_checkpoint_backup(
        monkeypatch, tmp_path):
    import scripts.fetch_models as fetch

    seen = {}

    def fake_snapshot_download(*, repo_id, local_dir):
        seen["download"] = (repo_id, local_dir)

    def fake_prepare(model_dir, *, backup):
        seen["prepare"] = (model_dir, backup)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )
    monkeypatch.setattr(fetch, "_prepare_gemma4_checkpoint", fake_prepare)

    fetch._download("gemma-4:31b-mlx", tmp_path)

    destination = tmp_path / "gemma-4-31b-mlx"
    assert seen["download"] == (
        "mlx-community/gemma-4-31b-4bit", str(destination))
    assert seen["prepare"] == (destination, False)

"""Ornith MLX adapter contracts that do not require real weights."""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

from edge0 import AutoConfig
from edge0.cli import DEMO_PROMPTS, TIER_ENV, cmd_models
from scripts import fetch_models


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src" / "edge0" / "engine" / "ornith.py"


def test_ornith_profile_uses_exact_k8_gate_routing():
    cfg = AutoConfig.from_pretrained(name="ornith:35b-mlx")

    assert cfg.name == "ornith:35b-mlx"
    assert cfg.model_dir == ""
    assert cfg.dense_spec is None
    assert cfg.moe_spec.num_experts == 256
    assert cfg.moe_spec.top_k == 8
    assert cfg.moe_spec.intermediate_size == 512
    assert cfg.moe_spec.router.value == "softmax_topk"
    assert cfg.moe_spec.norm_topk_prob is True
    assert cfg.moe_spec.shared_experts == 1
    assert cfg.moe_spec.quant.bits == 4
    assert cfg.moe_spec.quant.group_size == 64
    assert cfg.moe_spec.quant.mode == "affine"
    assert cfg.moe_spec.layout.value == "separate"
    assert cfg.moe_spec.key_template == (
        "language_model.model.layers.{layer}.mlp.switch_mlp")
    assert cfg.moe_spec.block_path == (
        "language_model.model.layers.{layer}.mlp")
    assert cfg.moe_spec.layer_path == "language_model.model.layers.{layer}"

    assert cfg.options.top_k == 8
    assert cfg.options.staged is False
    assert cfg.options.staged_replace is False
    assert cfg.options.staged_n == 8
    assert cfg.options.history_prefetch is False
    assert cfg.options.full_layer_prefill is True
    assert cfg.options.prefill_full_layers == 0
    assert cfg.options.hot_per_layer == 0
    assert cfg.prerouter is None
    assert cfg.prerouter_top_k == 0
    assert cfg.history_slots is False
    assert cfg.lora == ""
    assert cfg.prefetch_history is False
    assert cfg.hot_window == 0
    assert cfg.intra_staging is False

    assert cfg.gen.temperature == 1.0
    assert cfg.gen.top_p == 0.95
    assert cfg.gen.top_k == 20
    assert cfg.gen.repetition_penalty == 1.0
    assert cfg.gen.eos_ids == (248046, 248044)
    assert cfg.gen.max_new_tokens == 2048
    assert cfg.port == 8087
    assert cfg.target_tok_s == 0.0
    assert cfg.peak_active_mem_mb == 0.0


def test_ornith_adapter_reuses_qwen_loader_and_dedicated_engine(monkeypatch):
    adapter = importlib.import_module("edge0.models.ornith_35b_mlx")
    model = object()
    seen = {}

    def load_installed(model_dir, cfg):
        seen["load"] = (model_dir, cfg)
        return model, {"model_type": "qwen3_5_moe"}, [], {}

    class OrnithEngine:
        def __init__(self, model_dir, cfg):
            seen["engine"] = (model_dir, cfg)

    monkeypatch.setitem(
        sys.modules,
        "edge0.engine.qwen",
        SimpleNamespace(load_installed=load_installed),
    )
    monkeypatch.setitem(
        sys.modules,
        "edge0.engine.ornith",
        SimpleNamespace(Ornith35BMLXEngine=OrnithEngine),
    )

    assert adapter.build_model("/checkpoint") is model
    assert seen["load"][0] == "/checkpoint"
    assert seen["load"][1].name == "ornith:35b-mlx"

    engine = adapter.build_engine("/checkpoint")
    assert isinstance(engine, OrnithEngine)
    assert seen["engine"][0] == "/checkpoint"
    assert seen["engine"][1].name == "ornith:35b-mlx"


def test_ornith_engine_subclasses_qwen_without_duplicate_model_math():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Ornith35BMLXEngine"
    ]

    assert len(classes) == 1
    assert any(
        isinstance(base, ast.Name) and base.id == "Qwen35Engine"
        for base in classes[0].bases
    )
    assert 'name = "ornith:35b-mlx"' in source
    assert "mlx_vlm" not in source
    assert "install_streaming_experts" not in source


def test_ornith_cli_and_fetch_metadata(capsys):
    tier = "ornith:35b-mlx"
    assert TIER_ENV[tier] == "ORNITH_35B_MLX_MODEL"
    assert DEMO_PROMPTS[tier]
    assert fetch_models.TIER_REPOS[tier] == (
        "ORNITH_35B_MLX_REPO",
        "mlx-community/Ornith-1.0-35B-4bit",
    )
    assert fetch_models.TIER_DIRS[tier] == "ornith-35b-mlx"

    assert cmd_models(SimpleNamespace()) == 0
    output = capsys.readouterr().out
    assert "ornith:35b-mlx  (port 8087, target 0.0 tok/s" in output
    assert "experts=256 top_k=8 quant=4bit/g64" in output

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mlx-vlm' not in pyproject.lower()


def test_ornith_fetch_downloads_then_prepares_without_backup(
        monkeypatch, tmp_path):
    seen = []

    def snapshot_download(**kwargs):
        seen.append(("download", kwargs))

    def prepare(model_dir, *, backup):
        seen.append(("prepare", Path(model_dir), backup))

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(
        fetch_models, "_prepare_ornith_checkpoint", prepare)

    fetch_models._download("ornith:35b-mlx", tmp_path)

    destination = tmp_path / "ornith-35b-mlx"
    assert seen == [
        ("download", {
            "repo_id": "mlx-community/Ornith-1.0-35B-4bit",
            "local_dir": str(destination),
        }),
        ("prepare", destination, False),
    ]

"""Dense Qwen3.5-family adapter and engine contracts (no weights needed)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

from edge0.cli import DEMO_PROMPTS, TIER_ENV, cmd_models
from edge0.models.qwen3_8_27b_mlx import Qwen38_27BMLXConfig
from scripts.fetch_models import TIER_DIRS, TIER_REPOS


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src" / "edge0" / "engine" / "qwen_dense.py"


def test_cli_metadata_and_dense_listing(capsys):
    assert TIER_ENV["qwen3.8:27b-mlx"] == "QWEN38_27B_MLX_MODEL"
    assert DEMO_PROMPTS["qwen3.8:27b-mlx"]
    assert TIER_REPOS["qwen3.8:27b-mlx"] == (
        "QWEN38_27B_MLX_REPO",
        "mlx-community/Qwen3.8-27B-4bit",
    )
    assert TIER_DIRS["qwen3.8:27b-mlx"] == "qwen3.8-27b-mlx"

    assert cmd_models(SimpleNamespace()) == 0
    output = capsys.readouterr().out
    assert "qwen3.8:27b-mlx  (port 8084" in output
    assert "dense layers=64 hidden=5120 intermediate=17408" in output
    assert "quant=4bit/g64 affine" in output


def test_adapter_uses_dedicated_dense_loader_and_engine(monkeypatch):
    import edge0.models.qwen3_8_27b_mlx as adapter

    model = object()
    seen = {}

    def load_dense(model_dir, cfg):
        seen["load"] = (model_dir, cfg)
        return model, {"model_type": "qwen3_5"}

    class DenseEngine:
        def __init__(self, model_dir, cfg):
            seen["engine"] = (model_dir, cfg)

    fake_engine_module = SimpleNamespace(
        Qwen38_27BMLXEngine=DenseEngine,
        load_dense=load_dense,
    )
    monkeypatch.setitem(
        sys.modules, "edge0.engine.qwen_dense", fake_engine_module)

    assert adapter.build_model("/checkpoint") is model
    assert seen["load"][0] == "/checkpoint"
    assert isinstance(seen["load"][1], Qwen38_27BMLXConfig)

    engine = adapter.build_engine("/checkpoint")
    assert isinstance(engine, DenseEngine)
    assert seen["engine"][0] == "/checkpoint"
    assert isinstance(seen["engine"][1], Qwen38_27BMLXConfig)


def test_dense_engine_reuses_vendored_model_and_shared_generation_loop():
    source = ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)

    engine_classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "Qwen38_27BMLXEngine"
    ]
    assert len(engine_classes) == 1
    assert any(
        isinstance(base, ast.Name) and base.id == "Edge0Engine"
        for base in engine_classes[0].bases
    )
    assert "edge0.backends.mlx._impl.qwen3_5" in source
    assert "install_streaming_experts" not in source
    assert "open_shards" not in source

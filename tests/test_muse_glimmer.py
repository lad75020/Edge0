"""Muse Glimmer text-only adapter, architecture, and engine contracts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.cache import KVCache, RotatingKVCache

from edge0.backends.mlx._impl.muse_glimmer import Model, ModelArgs
from edge0.cli import DEMO_PROMPTS, TIER_ENV, cmd_models
from edge0.engine.muse_glimmer_dense import (
    MuseGlimmer30BMLXEngine,
    _get_model_classes,
    load_dense,
)
from edge0.models.muse_glimmer_30b_mlx import MuseGlimmer30BMLXConfig
from scripts.fetch_models import TIER_DIRS, TIER_REPOS


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src" / "edge0" / "engine" / "muse_glimmer_dense.py"


def _tiny_args(model_type: str = "muse_glimmer") -> ModelArgs:
    text = {
        "model_type": "muse_glimmer_text",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "rms_norm_eps": 1e-5,
        "post_norm_eps": 1e-8,
        "qk_scale_factor": 3.87,
        "output_multiplier": 0.25,
        "final_logit_softcapping": 2.0,
    }
    if model_type == "muse_glimmer_text":
        return ModelArgs.from_dict({"model_type": model_type, **text})
    return ModelArgs.from_dict({"model_type": model_type, "text_config": text})


def test_tiny_text_model_prefill_cached_decode_and_reset():
    model = Model(_tiny_args())
    cache = model.make_cache()

    assert hasattr(model, "language_model")
    assert not hasattr(model, "vision_tower")
    assert not hasattr(model, "vision_adapter")
    assert not hasattr(model, "vision_projection")

    assert [type(item) for item in cache] == [
        RotatingKVCache,
        RotatingKVCache,
        RotatingKVCache,
        KVCache,
    ]
    assert [item.offset for item in cache] == [0, 0, 0, 0]

    prefill_logits = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(prefill_logits)
    assert prefill_logits.shape == (1, 3, 32)
    assert bool(mx.all(mx.isfinite(prefill_logits)).item())
    assert [item.offset for item in cache] == [3, 3, 3, 3]

    decode_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(decode_logits)
    assert decode_logits.shape == (1, 1, 32)
    assert bool(mx.all(mx.isfinite(decode_logits)).item())
    assert [item.offset for item in cache] == [4, 4, 4, 4]

    engine = object.__new__(MuseGlimmer30BMLXEngine)
    engine._lm = model.language_model
    engine.cache = cache
    engine._reset_state()
    assert [item.offset for item in engine.cache] == [0, 0, 0, 0]


def test_language_model_returns_scaled_softcapped_raw_logits():
    model = Model(_tiny_args())
    ids = mx.array([[1, 2]])

    hidden = model.language_model.model(ids)
    projected = model.language_model.lm_head(hidden)
    expected = mx.tanh((projected * 0.25) / 2.0) * 2.0
    actual = model(ids)
    mx.eval(actual, expected)

    assert actual.shape == (1, 2, 32)
    assert bool(mx.allclose(actual, expected).item())
    assert float(mx.max(mx.abs(actual)).item()) <= 2.0


def test_flat_text_config_is_wrapped_for_loader_hook():
    args = _tiny_args("muse_glimmer_text")
    model = Model(args)

    assert args.model_type == "muse_glimmer_text"
    assert args.text_config.model_type == "muse_glimmer_text"
    assert model.language_model.args.num_hidden_layers == 4
    assert _get_model_classes({}) == (Model, ModelArgs)


def test_sanitizer_retains_text_keys_maps_official_keys_and_drops_vision():
    model = Model(_tiny_args())
    tensor = mx.ones((1,))
    weights = {
        "language_model.model.embed_tokens.weight": tensor,
        "model.language_model.layers.0.self_attn.q_proj.weight": tensor,
        "lm_head.weight": tensor,
        "vision_tower.layers.0.weight": tensor,
        "vision_adapter.weight": tensor,
        "vision_projection.weight": tensor,
        "model.vision_tower.layers.0.weight": tensor,
        "perception.weight": tensor,
        "model.perception_encoder.weight": tensor,
    }

    sanitized = model.sanitize(weights)

    assert set(sanitized) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.lm_head.weight",
    }
    assert sanitized["language_model.model.embed_tokens.weight"] is tensor


def test_loader_and_adapter_use_dedicated_resident_dense_hooks(monkeypatch):
    import edge0.engine.muse_glimmer_dense as engine_module
    import edge0.models.muse_glimmer_30b_mlx as adapter

    sentinel_model = object()
    seen = {}

    def fake_load_model(model_dir, **kwargs):
        seen["load_model"] = (model_dir, kwargs)
        return sentinel_model, {"model_type": "muse_glimmer"}

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    loaded, config = load_dense("/checkpoint", object())
    assert loaded is sentinel_model
    assert config == {"model_type": "muse_glimmer"}
    assert seen["load_model"][0] == "/checkpoint"
    assert seen["load_model"][1] == {
        "lazy": True,
        "strict": True,
        "get_model_classes": _get_model_classes,
    }

    built_model = object()

    def fake_load_dense(model_dir, cfg):
        seen["adapter_model"] = (model_dir, cfg)
        return built_model, {}

    class DenseEngine:
        def __init__(self, model_dir, cfg):
            seen["adapter_engine"] = (model_dir, cfg)

    fake_engine_module = SimpleNamespace(
        MuseGlimmer30BMLXEngine=DenseEngine,
        load_dense=fake_load_dense,
    )
    monkeypatch.setitem(
        sys.modules, "edge0.engine.muse_glimmer_dense", fake_engine_module)

    assert adapter.build_model("/checkpoint") is built_model
    assert isinstance(seen["adapter_model"][1], MuseGlimmer30BMLXConfig)
    assert isinstance(adapter.build_engine("/checkpoint"), DenseEngine)
    assert isinstance(seen["adapter_engine"][1], MuseGlimmer30BMLXConfig)

    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert "install_streaming_experts" not in source
    assert "open_shards" not in source


def test_loader_strictly_loads_a_tiny_local_checkpoint(tmp_path):
    config = {
        "model_type": "muse_glimmer",
        "quantization": {
            "group_size": 32,
            "bits": 4,
            "mode": "affine",
        },
        "text_config": {
            "model_type": "muse_glimmer_text",
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "sliding_window": 8,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "qk_scale_factor": 3.87,
            "output_multiplier": 0.25,
            "final_logit_softcapping": 2.0,
        },
    }
    (tmp_path / "config.json").write_text(
        json.dumps(config), encoding="utf-8")
    source = Model(ModelArgs.from_dict(config))
    nn.quantize(source, group_size=32, bits=4, mode="affine")
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(source.parameters())),
    )

    loaded, loaded_config = load_dense(str(tmp_path), object())
    cache = loaded.make_cache()
    logits = loaded(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(logits)

    assert loaded_config["model_type"] == "muse_glimmer"
    assert logits.shape == (1, 3, 64)
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert [item.offset for item in cache] == [3, 3, 3, 3]


def test_engine_chat_encoding_does_not_duplicate_template_bos():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Hello"}]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            return "<bos>rendered"

        def encode(self, text, **kwargs):
            assert text == "<bos>rendered"
            assert kwargs == {"add_special_tokens": False}
            return [200000, 42]

    engine = object.__new__(MuseGlimmer30BMLXEngine)
    engine._tok = Tokenizer()

    assert engine.encode_chat(
        [{"role": "user", "content": "Hello"}], think=False
    ) == [200000, 42]


def test_cli_fetch_metadata_and_dense_listing(capsys):
    tier = "muse-glimmer:30b-mlx"
    assert TIER_ENV[tier] == "MUSE_GLIMMER_30B_MODEL"
    assert DEMO_PROMPTS[tier]
    assert TIER_REPOS[tier] == (
        "MUSE_GLIMMER_30B_REPO",
        "mlx-community/Muse-Glimmer-30B-4bit",
    )
    assert TIER_DIRS[tier] == "muse-glimmer-30b-mlx"

    assert cmd_models(SimpleNamespace()) == 0
    output = capsys.readouterr().out
    assert "muse-glimmer:30b-mlx  (port 8086, target 0.0 tok/s" in output
    assert "dense layers=52 hidden=6656 intermediate=19968" in output
    assert "quant=4bit/g64 affine" in output

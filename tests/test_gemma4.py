"""Gemma 4 text-only architecture, adapter, engine, and prompt contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from jinja2 import Environment
from mlx_lm.models.cache import KVCache, RotatingKVCache

from edge0.backends.mlx._impl.gemma4 import Model, ModelArgs, TextModelArgs
from edge0.cli import DEMO_PROMPTS, TIER_ENV, _display_text, cmd_models
from edge0.engine.gemma4_dense import (
    Gemma4_31BMLXEngine,
    _get_model_classes,
    load_dense,
)
from edge0.models.gemma4_31b_mlx import (
    CHAT_TEMPLATE_PATH,
    Gemma4_31BMLXConfig,
)
from scripts.fetch_models import TIER_DIRS, TIER_REPOS


ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = ROOT / "src" / "edge0" / "engine" / "gemma4_dense.py"


def _tiny_args(model_type: str = "gemma4") -> ModelArgs:
    text = {
        "model_type": "gemma4_text",
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 6,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 8,
        "global_partial_rotary_factor": 0.25,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "sliding_window_pattern": 6,
        "attention_k_eq_v": True,
        "hidden_size_per_layer_input": 0,
        "num_kv_shared_layers": 0,
        "final_logit_softcapping": 2.0,
        "tie_word_embeddings": True,
        "rope_parameters": {
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
                "rope_type": "proportional",
            },
            "sliding_attention": {
                "rope_theta": 10_000.0,
                "rope_type": "default",
            },
        },
    }
    if model_type == "gemma4_text":
        return ModelArgs.from_dict({"model_type": model_type, **text})
    return ModelArgs.from_dict({"model_type": model_type, "text_config": text})


def test_default_text_args_match_gemma4_31b_attention_profile():
    args = TextModelArgs()

    assert args.model_type == "gemma4_text"
    assert (args.num_hidden_layers, args.hidden_size) == (60, 5376)
    assert (args.intermediate_size, args.vocab_size) == (21504, 262144)
    assert args.tie_word_embeddings is True
    assert (args.num_attention_heads, args.num_key_value_heads) == (32, 16)
    assert (args.num_global_key_value_heads, args.head_dim) == (4, 256)
    assert args.global_head_dim == 512
    assert args.attention_k_eq_v is True
    assert args.layer_types == (
        ["sliding_attention"] * 5 + ["full_attention"]
    ) * 10
    assert args.sliding_window == 1024
    assert args.rope_parameters["sliding_attention"] == {
        "rope_theta": 10_000.0,
        "rope_type": "default",
    }
    assert args.rope_parameters["full_attention"] == {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1_000_000.0,
        "rope_type": "proportional",
    }
    assert args.rms_norm_eps == 1e-6
    assert args.final_logit_softcapping == 30.0
    assert args.hidden_size_per_layer_input == 0
    assert args.num_kv_shared_layers == 0


def test_tiny_text_model_prefill_cached_decode_and_reset():
    model = Model(_tiny_args())
    cache = model.make_cache()

    assert hasattr(model, "language_model")
    assert not hasattr(model, "vision_tower")
    assert not hasattr(model, "embed_vision")
    assert [type(item) for item in cache] == [
        RotatingKVCache,
        RotatingKVCache,
        RotatingKVCache,
        RotatingKVCache,
        RotatingKVCache,
        KVCache,
    ]
    assert [item.offset for item in cache] == [0] * 6

    prefill_logits = model(mx.array([[1, 2, 3]]), cache=cache)
    mx.eval(prefill_logits)
    assert prefill_logits.shape == (1, 3, 32)
    assert bool(mx.all(mx.isfinite(prefill_logits)).item())
    assert [item.offset for item in cache] == [3] * 6

    decode_logits = model(mx.array([[4]]), cache=cache)
    mx.eval(decode_logits)
    assert decode_logits.shape == (1, 1, 32)
    assert bool(mx.all(mx.isfinite(decode_logits)).item())
    assert [item.offset for item in cache] == [4] * 6

    engine = object.__new__(Gemma4_31BMLXEngine)
    engine._lm = model.language_model
    engine.cache = cache
    engine._reset_state()
    assert [item.offset for item in engine.cache] == [0] * 6


def test_tiny_profile_uses_distinct_attention_shapes_and_softcapped_logits():
    args = _tiny_args()
    model = Model(args)

    assert args.text_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    sliding = model.layers[0].self_attn
    full = model.layers[5].self_attn
    assert (sliding.head_dim, sliding.n_kv_heads) == (4, 1)
    assert (full.head_dim, full.n_kv_heads) == (8, 1)
    assert hasattr(sliding, "v_proj")
    assert not hasattr(full, "v_proj")

    logits = model(mx.array([[1, 2]]))
    mx.eval(logits)
    assert logits.shape == (1, 2, 32)
    assert float(mx.max(mx.abs(logits)).item()) <= 2.0


def test_flat_text_config_is_wrapped_for_loader_hook():
    args = _tiny_args("gemma4_text")
    model = Model(args)

    assert args.model_type == "gemma4_text"
    assert isinstance(args.text_config, TextModelArgs)
    assert args.text_config.model_type == "gemma4_text"
    assert model.language_model.args.num_hidden_layers == 6
    assert _get_model_classes({}) == (Model, ModelArgs)


def test_sanitizer_maps_text_layouts_and_drops_unsupported_modalities():
    model = Model(_tiny_args())
    tensor = mx.ones((1,))
    weights = {
        "language_model.model.embed_tokens.weight": tensor,
        "language_model.layers.0.self_attn.q_proj.weight": tensor,
        "model.language_model.model.norm.weight": tensor,
        "model.language_model.layers.1.mlp.up_proj.weight": tensor,
        "model.layers.2.mlp.down_proj.weight": tensor,
        "vision_tower.encoder.layers.0.weight": tensor,
        "model.vision_tower.encoder.layers.0.weight": tensor,
        "embed_vision.embedding_projection.weight": tensor,
        "model.embed_vision.embedding_projection.weight": tensor,
        "perception_encoder.weight": tensor,
        "audio_tower.weight": tensor,
        "video_tower.weight": tensor,
        "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": tensor,
    }

    sanitized = model.sanitize(weights)

    assert set(sanitized) == {
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.norm.weight",
        "language_model.model.layers.1.mlp.up_proj.weight",
        "language_model.model.layers.2.mlp.down_proj.weight",
    }
    assert sanitized["language_model.model.embed_tokens.weight"] is tensor


def test_loader_and_adapter_use_dedicated_resident_dense_hooks(monkeypatch):
    import edge0.engine.gemma4_dense as engine_module
    import edge0.models.gemma4_31b_mlx as adapter

    sentinel_model = object()
    seen = {}

    def fake_load_model(model_dir, **kwargs):
        seen["load_model"] = (model_dir, kwargs)
        return sentinel_model, {"model_type": "gemma4"}

    monkeypatch.setattr(engine_module, "load_model", fake_load_model)
    loaded, config = load_dense("/checkpoint", object())
    assert loaded is sentinel_model
    assert config == {"model_type": "gemma4"}
    assert seen["load_model"] == ("/checkpoint", {
        "lazy": True,
        "strict": True,
        "get_model_classes": _get_model_classes,
    })

    built_model = object()

    def fake_load_dense(model_dir, cfg):
        seen["adapter_model"] = (model_dir, cfg)
        return built_model, {}

    class DenseEngine:
        def __init__(self, model_dir, cfg):
            seen["adapter_engine"] = (model_dir, cfg)

    monkeypatch.setitem(sys.modules, "edge0.engine.gemma4_dense", SimpleNamespace(
        Gemma4_31BMLXEngine=DenseEngine,
        load_dense=fake_load_dense,
    ))

    assert adapter.build_model("/checkpoint") is built_model
    assert isinstance(seen["adapter_model"][1], Gemma4_31BMLXConfig)
    assert isinstance(adapter.build_engine("/checkpoint"), DenseEngine)
    assert isinstance(seen["adapter_engine"][1], Gemma4_31BMLXConfig)

    source = ENGINE_PATH.read_text(encoding="utf-8")
    assert "install_streaming_experts" not in source
    assert "open_shards" not in source


def test_canonical_prompt_template_has_one_bos_and_gemma_generation_suffix():
    template = Environment().from_string(
        CHAT_TEMPLATE_PATH.read_text(encoding="utf-8"))
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


def test_engine_chat_encoding_uses_template_thinking_and_no_duplicate_bos():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "Hello"}]
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
                "enable_thinking": True,
            }
            return "<bos><|turn>user\nHello<turn|>\n<|turn>model\n"

        def encode(self, text, **kwargs):
            assert text.count("<bos>") == 1
            assert kwargs == {"add_special_tokens": False}
            return [2, 105, 42]

    engine = object.__new__(Gemma4_31BMLXEngine)
    engine._tok = Tokenizer()

    assert engine.encode_chat(
        [{"role": "user", "content": "Hello"}], think=True
    ) == [2, 105, 42]


def test_cli_hides_gemma_thought_channel_unless_requested():
    raw = (
        "<|channel>thought\nprivate chain\n<channel|>"
        "Public answer.<turn|>"
    )

    hidden = _display_text(raw, show_thinking=False)
    assert "private chain" not in hidden
    assert "thought" not in hidden
    assert "Public answer." in hidden
    assert _display_text(raw, show_thinking=True) == raw
    assert _display_text(
        "<|channel>thought\nunclosed private chain",
        show_thinking=False,
    ) == ""


def test_cli_fetch_metadata_and_dense_listing(capsys):
    tier = "gemma-4:31b-mlx"
    assert TIER_ENV[tier] == "GEMMA4_31B_MLX_MODEL"
    assert DEMO_PROMPTS[tier]
    assert TIER_REPOS[tier] == (
        "GEMMA4_31B_MLX_REPO",
        "mlx-community/gemma-4-31b-4bit",
    )
    assert TIER_DIRS[tier] == "gemma-4-31b-mlx"

    assert cmd_models(SimpleNamespace()) == 0
    output = capsys.readouterr().out
    assert "gemma-4:31b-mlx  (port 8088, target 0.0 tok/s" in output
    assert "dense layers=60 hidden=5376 intermediate=21504" in output
    assert "quant=4bit/g64 affine" in output

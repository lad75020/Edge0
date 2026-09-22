"""Text-only Ornith-1.0-35B MLX streaming-MoE adapter.

The prepared checkpoint preserves the upstream ``qwen3_5_moe`` model type
and fused ``switch_mlp`` key layout. Edge0 reuses its vendored Qwen3.5 MoE
implementation and exact gate-routed streaming engine; vision weights and
multimodal inputs are deliberately outside this adapter's scope.
"""

from __future__ import annotations

import sys

from edge0.config import GenerationConfig
from edge0.models.base import ModelConfig
from edge0.moe.spec import MoESpec, QuantSpec, RouterKind, WeightLayout
from edge0.registry import register_model
from edge0.streaming.options import LayerOptions


class Ornith35BMLXConfig(ModelConfig):
    """Configuration for ``mlx-community/Ornith-1.0-35B-4bit``."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "Ornith35BMLXConfig":
        return cls(
            name="ornith:35b-mlx",
            model_dir=model_dir,
            moe_spec=MoESpec(
                num_experts=256,
                top_k=8,
                intermediate_size=512,
                router=RouterKind.SOFTMAX_TOPK,
                norm_topk_prob=True,
                shared_experts=1,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
                layout=WeightLayout.SEPARATE,
                key_template=(
                    "language_model.model.layers.{layer}.mlp.switch_mlp"),
                block_path="language_model.model.layers.{layer}.mlp",
                layer_path="language_model.model.layers.{layer}",
            ),
            options=LayerOptions(
                staged=False,
                staged_replace=False,
                staged_n=8,
                staged_trigger=8,
                staged_sync=True,
                history_prefetch=False,
                hot_per_layer=0,
                cache_slots=64,
                prefetch_cap=48,
                load_threads=8,
                prefetch_threads=4,
                use_compile=True,
                top_k=8,
                full_layer_prefill=True,
                prefill_full_layers=0,
                prefill_hot=0,
            ),
            dense_spec=None,
            prerouter=None,
            prerouter_top_k=0,
            history_slots=False,
            lora="",
            gen=GenerationConfig(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                repetition_penalty=1.0,
                max_new_tokens=2048,
                eos_ids=(248046, 248044),
            ),
            prefill_chunk=2048,
            hot_window=0,
            intra_staging=False,
            prefetch_history=False,
            port=8087,
            target_tok_s=0.0,
            peak_active_mem_mb=0.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load the vendored Qwen3.5 MoE skeleton and streaming experts."""
    from edge0.engine.qwen import load_installed

    cfg = Ornith35BMLXConfig.from_pretrained(model_dir, **overrides)
    model, _model_config, _shards, _installs = load_installed(
        cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build the dedicated exact-routing Ornith generation engine."""
    from edge0.engine.ornith import Ornith35BMLXEngine

    cfg = Ornith35BMLXConfig.from_pretrained(model_dir, **overrides)
    return Ornith35BMLXEngine(cfg.model_dir, cfg)


register_model("ornith:35b-mlx", sys.modules[__name__])


Config = Ornith35BMLXConfig  # registry contract: adapter.Config

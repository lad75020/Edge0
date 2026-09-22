"""Dense Qwen3.8-27B MLX adapter.

The ``mlx-community/Qwen3.8-27B-4bit`` checkpoint uses the dense
Qwen3.5-family hybrid-attention backbone: 64 layers, hidden size 5120,
intermediate size 17408, and 4-bit affine weights with group size 64.
It has no routed experts, prerouter, LoRA, or streaming installation.
"""

from __future__ import annotations

import sys

from edge0.config import GenerationConfig
from edge0.models.base import DenseSpec, ModelConfig
from edge0.moe.spec import QuantSpec
from edge0.registry import register_model


class Qwen38_27BMLXConfig(ModelConfig):
    """Configuration for ``mlx-community/Qwen3.8-27B-4bit``."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "Qwen38_27BMLXConfig":
        return cls(
            name="qwen3.8:27b-mlx",
            model_dir=model_dir,
            moe_spec=None,
            options=None,
            dense_spec=DenseSpec(
                num_hidden_layers=64,
                hidden_size=5120,
                intermediate_size=17408,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
            ),
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
            port=8084,
            target_tok_s=6.0,
            peak_active_mem_mb=15100.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load the resident dense model without installing MoE machinery."""
    from edge0.engine.qwen_dense import load_dense

    cfg = Qwen38_27BMLXConfig.from_pretrained(model_dir, **overrides)
    model, _model_config = load_dense(cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build the dense Qwen engine backed by the shared generation loop."""
    from edge0.engine.qwen_dense import Qwen38_27BMLXEngine

    cfg = Qwen38_27BMLXConfig.from_pretrained(model_dir, **overrides)
    return Qwen38_27BMLXEngine(cfg.model_dir, cfg)


register_model("qwen3.8:27b-mlx", sys.modules[__name__])


Config = Qwen38_27BMLXConfig  # registry contract: adapter.Config

"""Text-only resident dense Gemma 4 31B MLX adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from edge0.config import GenerationConfig
from edge0.models.base import DenseSpec, ModelConfig
from edge0.moe.spec import QuantSpec
from edge0.registry import register_model


CHAT_TEMPLATE_PATH = Path(__file__).with_name("chat_template.jinja")


class Gemma4_31BMLXConfig(ModelConfig):
    """Configuration for ``mlx-community/gemma-4-31b-4bit``."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "Gemma4_31BMLXConfig":
        return cls(
            name="gemma-4:31b-mlx",
            model_dir=model_dir,
            moe_spec=None,
            options=None,
            dense_spec=DenseSpec(
                num_hidden_layers=60,
                hidden_size=5376,
                intermediate_size=21504,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
            ),
            prerouter=None,
            prerouter_top_k=0,
            history_slots=False,
            lora="",
            gen=GenerationConfig(
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                repetition_penalty=1.0,
                max_new_tokens=2048,
                eos_ids=(1,),
            ),
            prefill_chunk=2048,
            hot_window=0,
            intra_staging=False,
            prefetch_history=False,
            port=8088,
            target_tok_s=0.0,
            peak_active_mem_mb=0.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load only the resident dense language model."""
    from edge0.engine.gemma4_dense import load_dense

    cfg = Gemma4_31BMLXConfig.from_pretrained(model_dir, **overrides)
    model, _model_config = load_dense(cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build the resident text-generation engine."""
    from edge0.engine.gemma4_dense import Gemma4_31BMLXEngine

    cfg = Gemma4_31BMLXConfig.from_pretrained(model_dir, **overrides)
    return Gemma4_31BMLXEngine(cfg.model_dir, cfg)


register_model("gemma-4:31b-mlx", sys.modules[__name__])


Config = Gemma4_31BMLXConfig  # registry contract: adapter.Config

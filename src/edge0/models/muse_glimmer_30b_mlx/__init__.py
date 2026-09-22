"""Text-only resident dense Muse-Glimmer-30B MLX adapter."""

from __future__ import annotations

import sys

from edge0.config import GenerationConfig
from edge0.models.base import DenseSpec, ModelConfig
from edge0.moe.spec import QuantSpec
from edge0.registry import register_model


class MuseGlimmer30BMLXConfig(ModelConfig):
    """Configuration for ``mlx-community/Muse-Glimmer-30B-4bit``."""

    @classmethod
    def _defaults(cls, model_dir: str) -> "MuseGlimmer30BMLXConfig":
        return cls(
            name="muse-glimmer:30b-mlx",
            model_dir=model_dir,
            moe_spec=None,
            options=None,
            dense_spec=DenseSpec(
                num_hidden_layers=52,
                hidden_size=6656,
                intermediate_size=19968,
                quant=QuantSpec(bits=4, group_size=64, mode="affine"),
            ),
            prerouter=None,
            prerouter_top_k=0,
            history_slots=False,
            lora="",
            gen=GenerationConfig(
                temperature=0.0,
                top_p=1.0,
                top_k=0,
                repetition_penalty=1.0,
                max_new_tokens=2048,
                eos_ids=(200001, 200008),
            ),
            prefill_chunk=2048,
            hot_window=0,
            intra_staging=False,
            prefetch_history=False,
            port=8086,
            target_tok_s=0.0,
            peak_active_mem_mb=0.0,
        )


def build_model(model_dir: str | None = None, **overrides):
    """Load only the resident dense language model."""
    from edge0.engine.muse_glimmer_dense import load_dense

    cfg = MuseGlimmer30BMLXConfig.from_pretrained(model_dir, **overrides)
    model, _model_config = load_dense(cfg.model_dir, cfg)
    return model


def build_engine(model_dir: str | None = None, **overrides):
    """Build the resident text-generation engine."""
    from edge0.engine.muse_glimmer_dense import MuseGlimmer30BMLXEngine

    cfg = MuseGlimmer30BMLXConfig.from_pretrained(model_dir, **overrides)
    return MuseGlimmer30BMLXEngine(cfg.model_dir, cfg)


register_model("muse-glimmer:30b-mlx", sys.modules[__name__])


Config = MuseGlimmer30BMLXConfig  # registry contract: adapter.Config


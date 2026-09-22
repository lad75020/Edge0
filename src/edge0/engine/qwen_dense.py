"""Resident dense Qwen3.5-family engine for Qwen3.8-27B MLX."""

from __future__ import annotations

from edge0.backends import core
from edge0.backends.mlx._impl.qwen3_5 import Model as Qwen35DenseModel
from edge0.backends.mlx._impl.qwen3_5 import ModelArgs as Qwen35DenseArgs
from edge0.backends.mlx.io import load_model, load_tokenizer
from edge0.engine.base import Edge0Engine


def _get_model_classes(config):
    """mlx-lm class hook for the vendored dense Qwen3.5 backbone."""
    return Qwen35DenseModel, Qwen35DenseArgs


def load_dense(model_dir: str, cfg):
    """Load a dense checkpoint without opening or streaming expert shards."""
    return load_model(
        model_dir,
        lazy=True,
        strict=True,
        model_config={"model_type": "qwen3_5"},
        get_model_classes=_get_model_classes,
    )


class Qwen38_27BMLXEngine(Edge0Engine):
    """Dense Qwen3.8-27B engine using ``Edge0Engine.generate``."""

    name = "qwen3.8:27b-mlx"

    def _build(self):
        self.model, self.model_config = load_dense(self.cfg.model_dir, self.cfg)
        self._lm = self.model.language_model
        self._all_stream_layers = {}
        if self._tok is None:
            try:
                self._tok = load_tokenizer(self.cfg.model_dir)
            except Exception:  # noqa: BLE001 - tokenizer optional for API use
                pass
        self.cache = self._make_cache(self._lm)

    def _forward(self, ids, intra_stage: bool = True) -> core.array:
        del intra_stage
        inputs = core.array(ids)[None, :]
        logits = self._lm(inputs, cache=self.cache)[0, -1]
        core.eval(logits)
        return logits

    def _reset_state(self) -> None:
        self.cache = self._make_cache(self._lm)

    def _lm_logits(self, h: core.array) -> core.array:
        if self._lm.args.tie_word_embeddings:
            return self._lm.model.embed_tokens.as_linear(h[0, -1])
        return self._lm.lm_head(h[0, -1])

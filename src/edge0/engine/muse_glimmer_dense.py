"""Resident dense text engine for Muse-Glimmer-30B MLX."""

from __future__ import annotations

from edge0.backends import core
from edge0.backends.mlx._impl.muse_glimmer import Model, ModelArgs
from edge0.backends.mlx.io import load_model, load_tokenizer
from edge0.engine.base import Edge0Engine


def _get_model_classes(config):
    """mlx-lm class hook for the vendored Muse Glimmer text model."""
    del config
    return Model, ModelArgs


def load_dense(model_dir: str, cfg):
    """Load resident text weights without any streaming-expert path."""
    del cfg
    return load_model(
        model_dir,
        lazy=True,
        strict=True,
        get_model_classes=_get_model_classes,
    )


class MuseGlimmer30BMLXEngine(Edge0Engine):
    """Text-only Muse Glimmer engine using Edge0's generation loop."""

    name = "muse-glimmer:30b-mlx"

    def _build(self):
        self.model, self.model_config = load_dense(
            self.cfg.model_dir, self.cfg)
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

    def encode_chat(self, messages, think=None) -> list[int]:
        """Render Muse's template without adding a second BOS token.

        The template already begins with ``bos_token``. This tokenizer adds
        another BOS from ``encode`` by default, so the generic ChatSession
        path would otherwise produce ``[BOS, BOS, ...]``.
        """
        del think  # Muse controls effort through its template, not this flag.
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        return list(self._tok.encode(text, add_special_tokens=False))

    def _lm_logits(self, h: core.array) -> core.array:
        logits = self._lm.lm_head(h[0, -1]) * self._lm.output_multiplier
        softcap = self._lm.final_logit_softcapping
        return core.tanh(logits / softcap) * softcap

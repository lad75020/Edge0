"""Resident dense text engine for Gemma 4 31B MLX."""

from __future__ import annotations

from pathlib import Path

from edge0.backends import core
from edge0.backends.mlx._impl.gemma4 import Model, ModelArgs
from edge0.backends.mlx.io import load_model, load_tokenizer
from edge0.engine.base import Edge0Engine


def _get_model_classes(config):
    """mlx-lm class hook for the vendored Gemma 4 text model."""
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


class Gemma4_31BMLXEngine(Edge0Engine):
    """Text-only Gemma 4 engine using Edge0's generation loop."""

    name = "gemma-4:31b-mlx"

    def _build(self):
        self.model, self.model_config = load_dense(
            self.cfg.model_dir, self.cfg)
        self._lm = self.model.language_model
        self._all_stream_layers = {}
        if self._tok is None:
            try:
                self._tok = load_tokenizer(self.cfg.model_dir)
                if not getattr(self._tok, "chat_template", None):
                    template = Path(self.cfg.model_dir) / "chat_template.jinja"
                    self._tok.chat_template = template.read_text(encoding="utf-8")
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
        """Render canonical Gemma 4 chat without adding another BOS."""
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": bool(think),
        }
        try:
            text = self._tok.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking")
            text = self._tok.apply_chat_template(messages, **kwargs)
        return list(self._tok.encode(text, add_special_tokens=False))

    def _lm_logits(self, h: core.array) -> core.array:
        logits = self._lm.model.embed_tokens.as_linear(h[0, -1])
        softcap = self._lm.final_logit_softcapping
        return core.tanh(logits / softcap) * softcap

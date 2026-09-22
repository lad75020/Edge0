# Copyright © 2025 Prince Canuma
# SPDX-License-Identifier: MIT
"""Gemma 4 text architecture adapted from mlx-vlm 0.4.3.

This is a deliberately text-only adaptation of
``mlx_vlm.models.gemma4``. The vision, audio, video, perception, and
multimodal embedding paths are omitted. Imports target APIs present in
Edge0's pinned mlx-lm 0.31.0 runtime.

The full upstream MIT license is preserved in ``LICENSE.mlx-vlm`` beside
this file. Gemma checkpoint weights remain subject to the Gemma terms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache



@dataclass
class TextModelArgs(BaseModelArgs):
    """Configuration for the Gemma 4 language backbone."""

    model_type: str = "gemma4_text"
    hidden_size: int = 5376
    num_hidden_layers: int = 60
    intermediate_size: int = 21504
    num_attention_heads: int = 32
    head_dim: int = 256
    global_head_dim: int = 512
    global_partial_rotary_factor: float = 0.25
    rms_norm_eps: float = 1e-6
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    num_key_value_heads: int = 16
    num_global_key_value_heads: Optional[int] = 4
    num_kv_shared_layers: int = 0
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: Union[int, list[int]] = 1
    hidden_activation: str = "gelu_pytorch_tanh"
    hidden_size_per_layer_input: int = 0
    rope_traditional: bool = False
    rope_parameters: Optional[dict] = None
    sliding_window: int = 1024
    sliding_window_pattern: int = 6
    max_position_embeddings: int = 262144
    attention_bias: bool = False
    attention_dropout: float = 0.0
    attention_k_eq_v: bool = True
    final_logit_softcapping: float = 30.0
    use_double_wide_mlp: bool = False
    enable_moe_block: bool = False
    layer_types: Optional[list[str]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "full_attention": {
                    "partial_rotary_factor": self.global_partial_rotary_factor,
                    "rope_theta": 1_000_000.0,
                    "rope_type": "proportional",
                },
                "sliding_attention": {
                    "rope_theta": 10_000.0,
                    "rope_type": "default",
                },
            }
        if self.layer_types is None:
            pattern = ["sliding_attention"] * (
                self.sliding_window_pattern - 1
            ) + ["full_attention"]
            repeats = self.num_hidden_layers // len(pattern) + 1
            self.layer_types = (pattern * repeats)[:self.num_hidden_layers]


@dataclass
class ModelArgs(BaseModelArgs):
    """Top-level mlx-lm loader args retaining only the language model."""

    model_type: str = "gemma4"
    text_config: TextModelArgs = field(default_factory=TextModelArgs)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        model_type = str(params.get("model_type", "gemma4"))
        text_config = params.get("text_config", params)
        if isinstance(text_config, dict):
            text_config = TextModelArgs.from_dict(text_config)
        return cls(model_type=model_type, text_config=text_config)


class RMSNormNoScale(nn.Module):
    """RMSNorm without a learned scale."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


@partial(mx.compile, shapeless=True)
def logit_softcap(softcap: float, x: mx.array) -> mx.array:
    return mx.tanh(x / softcap) * softcap


class MLP(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=False)
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(
            nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.is_sliding = self.layer_type == "sliding_attention"
        self.head_dim = (
            args.head_dim if self.is_sliding else args.global_head_dim)
        self.n_heads = args.num_attention_heads
        self.use_k_eq_v = args.attention_k_eq_v and not self.is_sliding
        self.n_kv_heads = (
            args.num_global_key_value_heads
            if self.use_k_eq_v
            and args.num_global_key_value_heads is not None
            else args.num_key_value_heads
        )
        self.scale = 1.0

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        if not self.use_k_eq_v:
            self.v_proj = nn.Linear(
                dim, self.n_kv_heads * self.head_dim,
                bias=args.attention_bias)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.v_norm = RMSNormNoScale(args.rms_norm_eps)

        rope_key = (
            "sliding_attention" if self.is_sliding else "full_attention")
        rope_config = dict(args.rope_parameters.get(rope_key, {}))
        rope_theta = float(rope_config.get("rope_theta", 10_000.0))
        rope_dims = int(
            self.head_dim * float(
                rope_config.get("partial_rotary_factor", 1.0)
            )
        )
        # Gemma calls this configuration "proportional" because only a
        # proportion of each head is rotated.  mlx-lm 0.31.0's generic RoPE
        # factory does not recognize that name; mlx-vlm implements the model
        # with a regular RoPE whose dimension is reduced by the factor.
        self.rope = nn.RoPE(
            rope_dims,
            traditional=args.rope_traditional,
            base=rope_theta,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(
            batch, length, self.n_heads, self.head_dim)
        keys = self.k_proj(x).reshape(
            batch, length, self.n_kv_heads, self.head_dim)
        values = (
            keys
            if self.use_k_eq_v
            else self.v_proj(x).reshape(
                batch, length, self.n_kv_heads, self.head_dim)
        )

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = self.v_norm(values).transpose(0, 2, 1, 3)
        offset = cache.offset if cache is not None else 0
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        if mask is not None and isinstance(mask, mx.array):
            if mask.shape[-1] != keys.shape[-2]:
                mask = mask[..., -keys.shape[-2]:]

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.layer_type = args.layer_types[layer_idx]
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.pre_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.post_feedforward_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps)
        self.layer_scalar = mx.ones((1,))

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        h = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = residual + self.post_attention_layernorm(h)
        residual = h
        h = self.mlp(self.pre_feedforward_layernorm(h))
        h = residual + self.post_feedforward_layernorm(h)
        return h * self.layer_scalar


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_scale = args.hidden_size**0.5
        self.layers = [
            DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.full_attention_idx = self.layer_types.index("full_attention")
        self.sliding_attention_idx = self.layer_types.index(
            "sliding_attention")

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = (
            self.embed_tokens(inputs)
            if inputs_embeds is None
            else inputs_embeds
        ) * self.embed_scale
        if cache is None:
            cache = [None] * len(self.layers)

        global_mask = mask
        sliding_mask = mask
        if mask is None:
            global_mask = create_attention_mask(
                hidden_states, cache[self.full_attention_idx])
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )
        for layer, layer_cache in zip(self.layers, cache):
            layer_mask = (
                sliding_mask
                if layer.layer_type == "sliding_attention"
                else global_mask
            )
            hidden_states = layer(
                hidden_states, mask=layer_mask, cache=layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    """Gemma language model returning the raw logits array Edge0 expects."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.final_logit_softcapping = args.final_logit_softcapping

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        hidden_states = self.model(
            inputs,
            cache=cache,
            inputs_embeds=inputs_embeds,
            mask=mask,
        )
        logits = self.model.embed_tokens.as_linear(hidden_states)
        softcap = self.final_logit_softcapping
        return logit_softcap(softcap, logits) if softcap is not None else logits

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window, keep=0)
                if layer.layer_type == "sliding_attention"
                else KVCache()
            )
            for layer in self.layers
        ]


class Model(nn.Module):
    """Text-only Gemma 4 model compatible with mlx-lm's loader hook."""

    _UNSUPPORTED_PARTS = (
        "vision_tower",
        "embed_vision",
        "perception",
        "audio_tower",
        "embed_audio",
        "video_tower",
        "embed_video",
    )

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args.text_config)

    def __call__(self, inputs: mx.array, cache=None, **kwargs) -> mx.array:
        return self.language_model(inputs, cache=cache, **kwargs)

    def sanitize(self, weights):
        """Keep normalized language tensors and drop unsupported modalities."""
        sanitized = {}
        for key, value in weights.items():
            parts = key.split(".")
            if any(part.startswith(prefix)
                   for part in parts for prefix in self._UNSUPPORTED_PARTS):
                continue
            if "rotary_emb" in key:
                continue
            if any(name in key for name in (
                    "input_max", "input_min", "output_max", "output_min")):
                continue

            if key.startswith("model."):
                key = key[len("model."):]
            if key.startswith("language_model.model."):
                pass
            elif key.startswith("language_model."):
                key = "language_model.model." + key[len("language_model."):]
            elif self.model_type == "gemma4_text" or key.startswith(
                ("layers.", "embed_tokens.", "norm.")
            ):
                key = "language_model.model." + key
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

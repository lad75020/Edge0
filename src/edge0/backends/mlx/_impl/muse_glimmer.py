# Copyright © 2025 Prince Canuma
# SPDX-License-Identifier: MIT
"""Muse Glimmer text architecture adapted from mlx-vlm 0.6.12.

This is a deliberately text-only adaptation of
``mlx_vlm.models.muse_glimmer``.  The vision tower, adapter, projection,
and multimodal input path are omitted; their checkpoint tensors are removed
by :meth:`Model.sanitize`.  Imports target APIs present in Edge0's pinned
mlx-lm 0.31.0 runtime.

The full upstream MIT license is preserved in ``LICENSE.mlx-vlm`` beside
this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import KVCache, RotatingKVCache
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class TextModelArgs(BaseModelArgs):
    """Configuration for Muse Glimmer's dense language backbone."""

    model_type: str = "muse_glimmer_text"
    vocab_size: int = 202048
    hidden_size: int = 6656
    intermediate_size: int = 19968
    num_hidden_layers: int = 52
    num_attention_heads: int = 32
    num_key_value_heads: int = 2
    head_dim: int = 128
    hidden_activation: str = "silu"
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    post_norm_eps: float = 1e-8
    attention_bias: bool = False
    attention_dropout: float = 0.0
    sliding_window: int = 2048
    rope_parameters: Optional[dict] = None
    layer_types: Optional[list[str]] = None
    layer_rope_theta: Optional[list[Union[int, float]]] = None
    qk_scale_factor: float = 3.87
    output_multiplier: float = 0.19611613513818404
    final_logit_softcapping: float = 20.0
    tie_word_embeddings: bool = False
    bos_token_id: Optional[int] = 200000
    eos_token_id: Optional[Union[int, list[int]]] = 200001
    pad_token_id: Optional[int] = None

    def __post_init__(self):
        if self.rope_parameters is None:
            self.rope_parameters = {
                "rope_theta": 500000.0,
                "rope_type": "default",
            }
        if self.layer_types is None:
            self.layer_types = [
                (
                    "full_attention"
                    if (self.num_hidden_layers - 1 - idx) % 4 == 0
                    else "sliding_attention"
                )
                for idx in range(self.num_hidden_layers)
            ]
        if self.layer_rope_theta is None:
            theta = self.rope_parameters.get("rope_theta", 500000.0)
            self.layer_rope_theta = [
                0 if layer_type == "full_attention" else theta
                for layer_type in self.layer_types
            ]


@dataclass
class ModelArgs(BaseModelArgs):
    """Top-level loader args with only the language model retained."""

    model_type: str = "muse_glimmer"
    text_config: TextModelArgs = field(default_factory=TextModelArgs)

    @classmethod
    def from_dict(cls, params):
        params = dict(params)
        model_type = str(params.get("model_type", "muse_glimmer"))
        if "text_config" in params:
            text_config = params["text_config"]
        else:
            # A standalone muse_glimmer_text checkpoint keeps the text
            # configuration at the top level.
            text_config = params
        if isinstance(text_config, dict):
            text_config = TextModelArgs.from_dict(text_config)
        return cls(model_type=model_type, text_config=text_config)


@mx.compile
def _centered_rms_norm(
    x: mx.array, weight: mx.array, eps: float
) -> mx.array:
    dtype = x.dtype
    x = x.astype(mx.float32)
    variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    x = x * (1.0 + weight.astype(mx.float32))
    return x.astype(dtype)


class RMSNormNoScale(nn.Module):
    def __init__(self, eps: float):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, None, self.eps)


class CenteredRMSNorm(nn.Module):
    """RMSNorm whose checkpoint scale is centered at zero (``1 + w``)."""

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = mx.zeros((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return _centered_rms_norm(x, self.weight, self.eps)


class MLP(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.gate_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False)
        self.up_proj = nn.Linear(
            args.hidden_size, args.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            args.intermediate_size, args.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Attention(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5
        self.qk_scale_factor = args.qk_scale_factor
        self.use_rope = bool(args.layer_rope_theta[layer_idx])
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

        dim = args.hidden_size
        self.q_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(
            dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.gate_proj = nn.Linear(
            dim, self.n_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, dim, bias=args.attention_bias)
        self.qk_norm = RMSNormNoScale(args.rms_norm_eps)

        theta = (
            float(args.layer_rope_theta[layer_idx])
            if self.use_rope
            else float(args.rope_parameters.get("rope_theta", 500000.0))
        )
        self.rope = initialize_rope(
            self.head_dim,
            base=theta,
            traditional=False,
            scaling_config={"rope_type": "default", "rope_theta": theta},
            max_position_embeddings=args.max_position_embeddings,
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
        values = self.v_proj(x).reshape(
            batch, length, self.n_kv_heads, self.head_dim)

        queries = (
            self.qk_norm(queries) * self.qk_scale_factor
        ).transpose(0, 2, 1, 3)
        keys = self.qk_norm(keys).transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if self.use_rope:
            offset = cache.offset if cache is not None else 0
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(
            batch, length, -1)
        output = output * mx.sigmoid(self.gate_proj(x))
        return self.o_proj(output)


class DecoderLayer(nn.Module):
    def __init__(self, args: TextModelArgs, layer_idx: int):
        super().__init__()
        self.self_attn = Attention(args, layer_idx)
        self.mlp = MLP(args)
        self.input_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps)
        self.post_attention_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps)
        self.pre_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.rms_norm_eps)
        self.post_feedforward_layernorm = CenteredRMSNorm(
            args.hidden_size, args.post_norm_eps)
        self.is_sliding = args.layer_types[layer_idx] == "sliding_attention"

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        x = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        x = residual + self.post_attention_layernorm(x)
        residual = x
        x = self.mlp(self.pre_feedforward_layernorm(x))
        return residual + self.post_feedforward_layernorm(x)


class TextModel(nn.Module):
    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.embed_norm = RMSNormNoScale(args.rms_norm_eps)
        self.layers = [
            DecoderLayer(args, idx) for idx in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layer_types = args.layer_types
        self.sliding_window = args.sliding_window
        self.full_attention_idx = self.layer_types.index("full_attention")
        self.sliding_attention_idx = (
            self.layer_types.index("sliding_attention")
            if "sliding_attention" in self.layer_types
            else None
        )

    def __call__(
        self,
        inputs: Optional[mx.array],
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        hidden_states = inputs_embeds
        if hidden_states is None:
            hidden_states = self.embed_norm(self.embed_tokens(inputs))
        if cache is None:
            cache = [None] * len(self.layers)

        full_mask = create_attention_mask(
            hidden_states, cache[self.full_attention_idx])
        sliding_mask = None
        if self.sliding_attention_idx is not None:
            sliding_mask = create_attention_mask(
                hidden_states,
                cache[self.sliding_attention_idx],
                window_size=self.sliding_window,
            )
        for layer, layer_cache in zip(self.layers, cache):
            mask = sliding_mask if layer.is_sliding else full_mask
            hidden_states = layer(
                hidden_states, mask=mask, cache=layer_cache)
        return self.norm(hidden_states)


class LanguageModel(nn.Module):
    """Language model returning the raw logits array Edge0 expects."""

    def __init__(self, args: TextModelArgs):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = TextModel(args)
        self.lm_head = nn.Linear(
            args.hidden_size, args.vocab_size, bias=False)
        self.final_logit_softcapping = args.final_logit_softcapping
        self.output_multiplier = args.output_multiplier

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        cache=None,
        inputs_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        hidden_states = self.model(
            inputs, cache=cache, inputs_embeds=inputs_embeds)
        logits = self.lm_head(hidden_states) * self.output_multiplier
        softcap = self.final_logit_softcapping
        return mx.tanh(logits / softcap) * softcap

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                RotatingKVCache(max_size=self.args.sliding_window)
                if layer.is_sliding
                else KVCache()
            )
            for layer in self.layers
        ]


class Model(nn.Module):
    """Text-only Muse Glimmer model compatible with mlx-lm's loader hook."""

    _VISION_PARTS = {
        "vision_tower",
        "vision_adapter",
        "vision_projection",
        "perception",
    }

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.language_model = LanguageModel(args.text_config)

    def __call__(self, inputs: mx.array, cache=None, **kwargs) -> mx.array:
        return self.language_model(inputs, cache=cache, **kwargs)

    def sanitize(self, weights):
        """Keep text tensors and normalize supported checkpoint layouts."""
        sanitized = {}
        for key, value in weights.items():
            if any(
                part.startswith(prefix)
                for part in key.split(".")
                for prefix in self._VISION_PARTS
            ):
                continue
            if key.startswith("model.language_model."):
                rest = key[len("model.language_model."):]
                key = (
                    "language_model." + rest
                    if rest.startswith("model.")
                    else "language_model.model." + rest
                )
            elif key.startswith("language_model."):
                pass
            elif key.startswith("lm_head."):
                key = "language_model." + key
            elif self.model_type == "muse_glimmer_text" and key.startswith(
                "model."
            ):
                key = "language_model." + key
            sanitized[key] = value
        return sanitized

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

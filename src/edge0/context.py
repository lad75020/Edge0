"""Context-window validation and model-cache construction helpers."""

from __future__ import annotations

import re
from typing import Any


MAX_CONTEXT_SIZE = 512 * 1024

_CONTEXT_SIZE_PATTERN = re.compile(
    r"^(?P<count>[0-9]+)(?P<suffix>k|ki|ko)?$",
    re.IGNORECASE,
)


def parse_context_size(value: str) -> int:
    """Parse a CLI context size expressed as tokens.

    Bare values are token counts. The case-insensitive ``k``, ``ki``, and
    French ``ko`` suffixes use a binary multiplier, so ``512k`` is 524,288
    tokens.
    """
    match = _CONTEXT_SIZE_PATTERN.fullmatch(value)
    if match is None:
        raise ValueError(
            "context size must be an integer optionally followed "
            "by k, ki, or ko")
    parsed = int(match.group("count"))
    if match.group("suffix") is not None:
        parsed *= 1024
    return validate_context_size(parsed)


def validate_context_size(value: int) -> int:
    """Return a valid context size, or raise ``ValueError``."""
    if value <= 0:
        raise ValueError("context size must be a positive integer")
    if value > MAX_CONTEXT_SIZE:
        raise ValueError(
            f"context size must not exceed {MAX_CONTEXT_SIZE} tokens (512K)")
    return value


def completion_budget(
    prompt_tokens: int,
    requested_tokens: int,
    context_size: int | None,
) -> int:
    """Return the completion tokens available within a total context cap.

    A configured context covers both prompt and completion tokens. Prompts
    larger than that cap cannot be represented and are rejected. With no
    configured cap, the requested completion size is returned unchanged.
    """
    if context_size is None:
        return requested_tokens
    validate_context_size(context_size)
    if prompt_tokens > context_size:
        raise ValueError(
            f"prompt has {prompt_tokens} tokens, exceeding context size "
            f"{context_size}")
    return min(requested_tokens, context_size - prompt_tokens)


def make_model_cache(
    model: Any,
    context_size: int | None = None,
) -> Any:
    """Build a model cache, optionally capping every attention KV cache.

    Recurrent ``ArraysCache`` entries are deliberately preserved: they hold
    fixed-size recurrent state rather than a token history. Model-defined
    sliding-window caches retain their smaller window when it is below the
    requested global context cap.
    """
    cache = model.make_cache()
    if context_size is None:
        return cache

    validate_context_size(context_size)
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    capped = []
    for item in cache:
        if isinstance(item, RotatingKVCache):
            max_size = min(item.max_size, context_size)
            keep = min(item.keep, max(0, max_size - 1))
            item = RotatingKVCache(max_size=max_size, keep=keep)
        elif isinstance(item, KVCache):
            keep = min(4, max(0, context_size - 1))
            item = RotatingKVCache(max_size=context_size, keep=keep)
        capped.append(item)
    return capped

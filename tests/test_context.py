"""Focused context-size validation and cache-construction tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from edge0.config import GenerationConfig
from edge0.context import (
    MAX_CONTEXT_SIZE,
    completion_budget,
    make_model_cache,
    parse_context_size,
    validate_context_size,
)


@pytest.mark.parametrize("value", [1, MAX_CONTEXT_SIZE])
def test_context_size_accepts_boundaries(value):
    assert validate_context_size(value) == value


@pytest.mark.parametrize("value", [0, -1, MAX_CONTEXT_SIZE + 1])
def test_context_size_rejects_out_of_range_values(value):
    with pytest.raises(ValueError, match="context size"):
        validate_context_size(value)


def test_context_size_suffix_uses_binary_tokens():
    assert parse_context_size("512ko") == MAX_CONTEXT_SIZE


def test_completion_budget_includes_prompt_and_requested_completion():
    assert completion_budget(3, 10, 5) == 2
    assert completion_budget(3, 10, None) == 10


def test_completion_budget_rejects_oversized_prompt():
    with pytest.raises(ValueError, match="prompt.*context size"):
        completion_budget(6, 1, 5)


def test_default_cache_is_left_unchanged():
    original = [object()]
    model = SimpleNamespace(make_cache=lambda: original)

    assert make_model_cache(model) is original


def test_context_size_caps_attention_caches_and_preserves_recurrent_state(
        monkeypatch):
    class KVCache:
        pass

    class RotatingKVCache:
        def __init__(self, max_size, keep=0):
            self.max_size = max_size
            self.keep = keep

    class ArraysCache:
        pass

    mlx_lm = ModuleType("mlx_lm")
    models = ModuleType("mlx_lm.models")
    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.KVCache = KVCache
    cache_module.RotatingKVCache = RotatingKVCache
    mlx_lm.models = models
    models.cache = cache_module
    monkeypatch.setitem(sys.modules, "mlx_lm", mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.models", models)
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)

    recurrent = ArraysCache()
    model = SimpleNamespace(make_cache=lambda: [
        KVCache(),
        RotatingKVCache(max_size=4096, keep=2),
        recurrent,
    ])

    capped = make_model_cache(model, 1024)

    assert isinstance(capped[0], RotatingKVCache)
    assert (capped[0].max_size, capped[0].keep) == (1024, 4)
    assert (capped[1].max_size, capped[1].keep) == (1024, 2)
    assert capped[2] is recurrent


def test_one_token_context_uses_a_valid_zero_keep_cache(monkeypatch):
    class KVCache:
        pass

    class RotatingKVCache:
        def __init__(self, max_size, keep=0):
            self.max_size = max_size
            self.keep = keep

    cache_module = ModuleType("mlx_lm.models.cache")
    cache_module.KVCache = KVCache
    cache_module.RotatingKVCache = RotatingKVCache
    monkeypatch.setitem(sys.modules, "mlx_lm.models.cache", cache_module)

    capped = make_model_cache(
        SimpleNamespace(make_cache=lambda: [KVCache()]), 1)

    assert (capped[0].max_size, capped[0].keep) == (1, 0)


def test_generation_caps_completion_to_remaining_context(monkeypatch):
    import edge0.engine.base as engine_base

    engine = object.__new__(engine_base.Edge0Engine)
    engine.cfg = SimpleNamespace(
        context_size=5,
        gen=GenerationConfig(max_new_tokens=20, first_token_greedy=False),
    )
    engine._last_logits = object()
    engine.pos = 0
    engine.prefill = lambda token_ids: len(token_ids)
    engine.next_logits = lambda: engine._last_logits
    engine.step = lambda token_id: engine._last_logits
    monkeypatch.setattr(engine_base, "sample", lambda *args, **kwargs: 7)

    output = engine.generate([1, 2, 3], max_new_tokens=10)

    assert output == [7, 7]
    assert len([1, 2, 3]) + len(output) == engine.cfg.context_size


def test_generation_rejects_prompt_larger_than_context():
    import edge0.engine.base as engine_base

    engine = object.__new__(engine_base.Edge0Engine)
    engine.cfg = SimpleNamespace(
        context_size=2,
        gen=GenerationConfig(first_token_greedy=False),
    )

    with pytest.raises(ValueError, match="prompt.*context size"):
        engine.generate([1, 2, 3])

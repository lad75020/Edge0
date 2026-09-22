"""edge0 engines: streaming prefill/decode lifecycles per model family.

``edge0.engine.base.Edge0Engine`` owns the shared prefill / step /
generate loops; ``edge0.engine.qwen.Qwen35Engine`` and
``edge0.engine.ling.Ling8BEngine`` implement the streaming family hooks;
``edge0.engine.qwen_dense.Qwen38_27BMLXEngine`` supplies the resident
dense Qwen hooks; ``Gemma4_31BMLXEngine`` and ``MuseGlimmer30BMLXEngine``
supply resident text-only multimodal-checkpoint hooks. ``Ornith35BMLXEngine`` reuses the Qwen
streaming hooks with an exact K=8 gate-routing profile.
"""

from edge0.engine.base import Edge0Engine
from edge0.engine.hooks import (
    make_history_prefetch,
    make_intra_after_layer,
    make_prefill_before_layer,
)
from edge0.engine.ling import Ling8BEngine
from edge0.engine.gemma4_dense import Gemma4_31BMLXEngine
from edge0.engine.muse_glimmer_dense import MuseGlimmer30BMLXEngine
from edge0.engine.ornith import Ornith35BMLXEngine
from edge0.engine.qwen import Qwen35Engine
from edge0.engine.qwen_dense import Qwen38_27BMLXEngine

__all__ = [
    "Edge0Engine",
    "Qwen35Engine",
    "Qwen38_27BMLXEngine",
    "Ling8BEngine",
    "Gemma4_31BMLXEngine",
    "MuseGlimmer30BMLXEngine",
    "Ornith35BMLXEngine",
    "make_history_prefetch",
    "make_intra_after_layer",
    "make_prefill_before_layer",
]

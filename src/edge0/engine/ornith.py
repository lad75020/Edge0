"""Text-only Ornith-1.0-35B engine using the Qwen streaming path."""

from __future__ import annotations

from edge0.engine.qwen import Qwen35Engine


class Ornith35BMLXEngine(Qwen35Engine):
    """Qwen3.5-MoE engine with Ornith's exact K=8 gate-routing profile."""

    name = "ornith:35b-mlx"

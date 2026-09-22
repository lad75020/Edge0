"""Model adapters: one package per tier, registered on import.

``edge0.models.base`` carries the shared ``ModelConfig``; the model
packages (``edge0.models.edge0_35b``, ``edge0.models.edge0_8b``)
define their family configs and the ``build_model`` / ``build_engine``
entry points, then call ``register_model`` so ``AutoConfig`` /
``AutoModel`` / ``AutoEngine`` can resolve them by name.
"""

from __future__ import annotations

from edge0.models.base import DenseSpec, ModelConfig

# Importing the adapter packages populates edge0.registry.MODEL_REGISTRY.
from edge0.models import (  # noqa: F401
    edge0_8b,
    edge0_35b,
    gemma4_31b_mlx,
    muse_glimmer_30b_mlx,
    ornith_35b_mlx,
    qwen3_8_27b_mlx,
)

__all__ = [
    "DenseSpec",
    "ModelConfig",
    "edge0_8b",
    "edge0_35b",
    "gemma4_31b_mlx",
    "muse_glimmer_30b_mlx",
    "ornith_35b_mlx",
    "qwen3_8_27b_mlx",
]

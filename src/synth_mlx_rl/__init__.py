"""Native reinforcement learning for Apple MLX.

Two OpenAI-compatible surfaces over one renderer and one rollout record, a
resident LoRA learner with frozen policy snapshots, and the SFT / GRPO / CISPO
objective family.

Seeded from the MIT-licensed `mlx-local-rl` prototype; see NOTICE for the
origin, its terms, and what is and is not derived from it.
"""

from __future__ import annotations

from .client import AdamParams, Datum, SamplingParams, ServiceClient
from .objective_spec import (
    SUPPORTED_OBJECTIVES,
    UNAVAILABLE_OBJECTIVES,
    ObjectiveError,
    ObjectiveSpec,
)

__all__ = [
    "AdamParams",
    "Datum",
    "ObjectiveError",
    "ObjectiveSpec",
    "SUPPORTED_OBJECTIVES",
    "SamplingParams",
    "ServiceClient",
    "UNAVAILABLE_OBJECTIVES",
]
__version__ = "0.6.0"

"""Reward bookkeeping. Not model math -- this runs before any array reaches MLX."""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import Sequence


def group_normalize_rewards(rewards: Sequence[float], eps: float = 1e-4) -> list[float]:
    """Zero-mean group advantages over a population standard deviation.

    A constant-reward group carries no preference signal, so it returns zeros
    rather than amplifying float noise into a direction. This is the check that
    a hosted CISPO canary failed on eight Banking77 rollouts: four groups, no
    variance, no optimizer step.
    """

    values = [float(value) for value in rewards]
    if not values:
        raise ValueError("rewards must be a non-empty sequence")
    mean = fmean(values)
    std = pstdev(values)
    if std < eps:
        return [0.0] * len(values)
    return [(value - mean) / (std + eps) for value in values]

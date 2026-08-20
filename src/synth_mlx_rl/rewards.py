"""Reward bookkeeping. Not model math -- this runs before any array reaches MLX."""

from __future__ import annotations

import math
from statistics import fmean, stdev
from typing import Sequence


class RewardGroupError(ValueError):
    """A reward group that cannot produce a learning signal."""


def normalize_group_rewards(rewards: Sequence[float]) -> list[float]:
    """Group advantages, matching slime's sample-standard-deviation convention.

    This is the same normalization the hosted CISPO runner uses, deliberately:
    a local result and a hosted one are only comparable if the advantage
    definition is identical. Sample stdev (n-1), not population, and a 1e-6
    denominator floor.

    Raises rather than returning zeros for a degenerate group. A constant-reward
    group carries no preference signal, and zeroing it silently would spend a
    training step on nothing -- the hosted lane treats that as a filtered group
    and resamples, and so does the local one. A hosted CISPO canary once reached
    eight Banking77 rollouts where all four groups had zero variance and no
    optimizer step was ever produced; that is the failure this refuses to hide.
    """

    values = [float(value) for value in rewards]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise RewardGroupError("cispo_reward_group_invalid")
    mean = fmean(values)
    denominator = stdev(values) + 1e-6
    return [(value - mean) / denominator for value in values]


def has_learning_signal(rewards: Sequence[float]) -> bool:
    """True when a group's rewards differ at all, so an advantage is meaningful."""

    values = [float(value) for value in rewards]
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        return False
    return any(value != values[0] for value in values[1:])

"""Pure-NumPy reference objectives: the parity oracle.

Derived from the MIT-licensed `mlx-local-rl` prototype (see NOTICE) and extended
with the CISPO family.

This module is written *independently* of ``kernel.py`` on purpose. ``kernel.py``
is the expression tree MLX evaluates; this is a second implementation of the same
mathematics in plain NumPy, and the test suite asserts the two agree. A shared
helper between them would make that agreement a tautology, so there isn't one.

Everything here runs on any machine, with no MLX and no Apple Silicon.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .objective_spec import LOG_RATIO_CLAMP, ObjectiveSpec

__all__ = [
    "PolicyLossMetrics",
    "group_normalize_rewards",
    "cross_entropy_loss",
    "policy_loss",
    "importance_sampling_loss",
    "clipped_policy_loss",
    "cispo_loss",
]


@dataclass(frozen=True, slots=True)
class PolicyLossMetrics:
    loss: float
    policy_loss: float
    reference_kl: float
    approx_kl: float
    clip_fraction: float
    mean_ratio: float
    token_count: float
    clamped_token_count: float = 0.0
    nonfinite_token_count: float = 0.0
    entropy: float = 0.0


def group_normalize_rewards(
    rewards: list[float] | np.ndarray, eps: float = 1e-4
) -> np.ndarray:
    """Return zero-mean group advantages with a population standard deviation.

    A constant-reward group carries no preference signal, so it returns zeros
    rather than amplifying float noise into a direction.
    """

    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rewards must be a non-empty one-dimensional sequence")
    centered = values - values.mean()
    std = values.std()
    if std < eps:
        return np.zeros_like(values)
    return centered / (std + eps)


def cross_entropy_loss(
    current_logprobs: np.ndarray, weights: np.ndarray
) -> PolicyLossMetrics:
    """Masked cross-entropy reduced over unmasked tokens."""

    current = np.asarray(current_logprobs, dtype=np.float64)
    mask = np.asarray(weights, dtype=np.float64)
    if current.shape != mask.shape:
        raise ValueError("current_logprobs and weights must have equal shapes")
    denom = float(mask.sum())
    if denom <= 0:
        raise ValueError("weights must select at least one token")
    mean_logprob = float((current * mask).sum() / denom)
    return PolicyLossMetrics(
        loss=-mean_logprob,
        policy_loss=-mean_logprob,
        reference_kl=0.0,
        approx_kl=0.0,
        clip_fraction=0.0,
        mean_ratio=1.0,
        token_count=denom,
    )


def policy_loss(
    spec: ObjectiveSpec,
    *,
    current_logprobs: np.ndarray,
    behavior_logprobs: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    reference_logprobs: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
) -> PolicyLossMetrics:
    """Reference value of any policy objective in :mod:`objective_spec`.

    ``advantages`` may hold one value per target position; broadcasting a scalar
    is the caller's job so that the masked reduction stays explicit.
    """

    current = np.asarray(current_logprobs, dtype=np.float64)
    behavior = np.asarray(behavior_logprobs, dtype=np.float64)
    adv = np.asarray(advantages, dtype=np.float64)
    mask = np.asarray(weights, dtype=np.float64)
    shapes = {current.shape, behavior.shape, adv.shape, mask.shape}
    if len(shapes) != 1:
        raise ValueError(
            "current_logprobs, behavior_logprobs, advantages, and weights must "
            "have equal shapes"
        )

    # Match the kernel's sanitization exactly: a non-finite log-ratio or
    # current log-probability is zeroed and its position leaves the reduction
    # denominator. The value has to be replaced, not only masked -- CISPO
    # multiplies by current_logprobs and 0 * nan is nan.
    current_f32 = current.astype(np.float32).astype(np.float64)
    raw = current_f32 - behavior.astype(np.float32).astype(np.float64)
    finite = (np.isfinite(raw) & np.isfinite(current_f32)).astype(np.float64)
    nonfinite_tokens = float(((1.0 - finite) * mask).sum())
    safe = np.where(finite > 0.0, raw, 0.0)
    current = np.where(finite > 0.0, current, 0.0)
    mask = mask * finite

    denom = float(mask.sum())
    if denom <= 0:
        raise ValueError("weights must select at least one finite token")

    log_ratio = np.clip(safe, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    clamped_tokens = float(((np.abs(safe) > LOG_RATIO_CLAMP) * mask).sum())
    ratio = np.exp(log_ratio)

    if spec.is_cispo:
        truncated = np.clip(ratio, spec.clip_low, spec.clip_high)
        per_token = truncated * adv * current
        clipped_tokens = float(((np.abs(ratio - truncated) > 0.0) * mask).sum())
    elif spec.name == "grpo":
        clipped_ratio = np.clip(ratio, spec.clip_low, spec.clip_high)
        per_token = np.minimum(ratio * adv, clipped_ratio * adv)
        clipped_tokens = float(((np.abs(ratio - clipped_ratio) > 0.0) * mask).sum())
    else:
        per_token = ratio * adv
        clipped_tokens = 0.0

    surrogate = -float((per_token * mask).sum() / denom)
    approx_kl = float((np.square(log_ratio) * mask).sum() / (2.0 * denom))
    mean_ratio = float((ratio * mask).sum() / denom)

    reference_kl = 0.0
    if reference_logprobs is not None:
        reference = np.asarray(reference_logprobs, dtype=np.float64)
        if reference.shape != current.shape:
            raise ValueError("reference_logprobs must match current_logprobs")
        ref_mask = (
            np.ones_like(mask)
            if reference_mask is None
            else np.asarray(reference_mask, dtype=np.float64)
        )
        ref_weights = mask * ref_mask
        ref_count = max(float(ref_weights.sum()), 1e-8)
        log_ref_ratio = np.clip(
            reference.astype(np.float32).astype(np.float64)
            - current.astype(np.float32).astype(np.float64),
            -LOG_RATIO_CLAMP,
            LOG_RATIO_CLAMP,
        )
        k3 = np.exp(log_ref_ratio) - log_ref_ratio - 1.0
        reference_kl = float((k3 * ref_weights).sum() / ref_count)

    return PolicyLossMetrics(
        loss=surrogate + spec.kl_beta * reference_kl,
        policy_loss=surrogate,
        reference_kl=reference_kl,
        approx_kl=approx_kl,
        clip_fraction=clipped_tokens / denom,
        mean_ratio=mean_ratio,
        token_count=denom,
        clamped_token_count=clamped_tokens,
        nonfinite_token_count=nonfinite_tokens,
    )


def importance_sampling_loss(
    current_logprobs: np.ndarray,
    behavior_logprobs: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
) -> PolicyLossMetrics:
    """Unclipped token-level importance-sampled policy gradient."""

    return policy_loss(
        ObjectiveSpec(name="importance_sampling"),
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        weights=weights,
    )


def clipped_policy_loss(
    current_logprobs: np.ndarray,
    behavior_logprobs: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    *,
    clip_epsilon: float = 0.2,
    reference_logprobs: np.ndarray | None = None,
    kl_beta: float = 0.0,
) -> PolicyLossMetrics:
    """Masked GRPO clipped surrogate with an optional k3 reference penalty."""

    return policy_loss(
        ObjectiveSpec(name="grpo", clip_epsilon=clip_epsilon, kl_beta=kl_beta),
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        weights=weights,
        reference_logprobs=reference_logprobs,
    )


def cispo_loss(
    current_logprobs: np.ndarray,
    behavior_logprobs: np.ndarray,
    advantages: np.ndarray,
    weights: np.ndarray,
    *,
    variant: str = "cispo_minimax",
    eps_low: float = 1.0,
    eps_high: float = 4.0,
    reference_logprobs: np.ndarray | None = None,
    kl_beta: float = 0.0,
) -> PolicyLossMetrics:
    """CISPO (MiniMax-M1, Eq. 4-5).

    ``variant="cispo_minimax"`` refuses ``eps_low < 1.0``; ask for
    ``variant="cispo_two_sided"`` if an active lower bound is what you want.
    """

    return policy_loss(
        ObjectiveSpec(
            name=variant, eps_low=eps_low, eps_high=eps_high, kl_beta=kl_beta
        ),
        current_logprobs=current_logprobs,
        behavior_logprobs=behavior_logprobs,
        advantages=advantages,
        weights=weights,
        reference_logprobs=reference_logprobs,
    )

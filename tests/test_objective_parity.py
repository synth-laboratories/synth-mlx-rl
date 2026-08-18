"""NumPy reference vs. the kernel MLX executes.

``objectives.py`` and ``kernel.py`` are two independent implementations of the
same mathematics. This file runs the kernel over the NumPy backend -- the same
source MLX will evaluate, just with a different array namespace -- and asserts
it agrees with the reference on randomized inputs.

A disagreement means one of the two is wrong. That is the point: without MLX on
this machine, an unchecked kernel would be unexecuted code all the way to the
first real run.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_mlx_rl.backends import NumpyOps
from synth_mlx_rl.kernel import policy_terms, sft_terms
from synth_mlx_rl.objective_spec import ObjectiveSpec
from synth_mlx_rl.objectives import cross_entropy_loss, policy_loss

SPECS = [
    ObjectiveSpec(name="importance_sampling"),
    ObjectiveSpec(name="grpo", clip_epsilon=0.2),
    ObjectiveSpec(name="grpo", clip_epsilon=0.05),
    ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0),
    ObjectiveSpec(name="cispo_minimax", eps_low=2.0, eps_high=0.5),
    ObjectiveSpec(name="cispo_two_sided", eps_low=0.2, eps_high=0.28),
    ObjectiveSpec(name="grpo", clip_epsilon=0.2, kl_beta=0.7),
    ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0, kl_beta=0.3),
]


def _random_batch(rng: np.random.Generator, size: int = 32):
    behavior = rng.normal(-1.5, 1.0, size)
    # A wide spread on purpose: some ratios land outside every clip bound.
    current = behavior + rng.normal(0.0, 1.5, size)
    advantages = rng.normal(0.0, 1.0, size)
    weights = (rng.random(size) > 0.25).astype(np.float64)
    weights[0] = 1.0
    reference = behavior + rng.normal(0.0, 0.5, size)
    return current, behavior, advantages, weights, reference


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.name}-{s.clip_low}-{s.clip_high}-kl{s.kl_beta}")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_kernel_matches_numpy_reference(spec: ObjectiveSpec, seed: int) -> None:
    rng = np.random.default_rng(seed)
    current, behavior, advantages, weights, reference = _random_batch(rng)
    use_reference = spec.kl_beta > 0.0

    expected = policy_loss(
        spec,
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=advantages,
        weights=weights,
        reference_logprobs=reference if use_reference else None,
        reference_mask=np.ones_like(weights) if use_reference else None,
    )
    loss, terms = policy_terms(
        NumpyOps(),
        spec,
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=advantages,
        weights=weights,
        reference_logprobs=reference if use_reference else None,
        reference_mask=np.ones_like(weights) if use_reference else None,
    )

    assert np.isclose(float(loss), expected.loss, rtol=1e-10, atol=1e-12)
    assert np.isclose(float(terms["policy_loss"]), expected.policy_loss, rtol=1e-10)
    assert np.isclose(float(terms["approx_kl"]), expected.approx_kl, rtol=1e-10)
    assert np.isclose(float(terms["mean_ratio"]), expected.mean_ratio, rtol=1e-10)
    assert np.isclose(float(terms["clip_fraction"]), expected.clip_fraction, rtol=1e-10)
    assert np.isclose(float(terms["token_count"]), expected.token_count)
    assert np.isclose(float(terms["reference_kl"]), expected.reference_kl, rtol=1e-10)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sft_kernel_matches_numpy_reference(seed: int) -> None:
    rng = np.random.default_rng(seed)
    current = rng.normal(-2.0, 1.0, 16)
    weights = (rng.random(16) > 0.3).astype(np.float64)
    weights[0] = 1.0
    expected = cross_entropy_loss(current, weights)
    loss, terms = sft_terms(NumpyOps(), current, weights)
    assert np.isclose(float(loss), expected.loss, rtol=1e-12)
    assert np.isclose(float(terms["token_count"]), expected.token_count)


def test_fractional_weights_keep_their_scale() -> None:
    """Half-weighted tokens must halve their contribution, not round to one."""

    current = np.array([-1.0, -3.0])
    weights = np.array([0.5, 1.0])
    loss, terms = sft_terms(NumpyOps(), current, weights)
    assert np.isclose(float(terms["token_count"]), 1.5)
    assert np.isclose(float(loss), (0.5 * 1.0 + 1.0 * 3.0) / 1.5)


@pytest.mark.parametrize("spec", SPECS[:6], ids=lambda s: s.name)
def test_extreme_log_ratios_are_clamped_and_counted(spec: ObjectiveSpec) -> None:
    """A ratio that would overflow float32 is clamped, and the clamp is visible."""

    current = np.array([0.0, 0.0, 0.0, 0.0])
    behavior = np.array([-500.0, 500.0, 0.0, -1.0])
    advantages = np.array([1.0, 1.0, 1.0, 1.0])
    weights = np.ones(4)

    loss, terms = policy_terms(
        NumpyOps(),
        spec,
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=advantages,
        weights=weights,
    )
    assert np.isfinite(float(loss))
    assert float(terms["clamped_token_count"]) == 2.0
    assert float(terms["nonfinite_token_count"]) == 0.0

    expected = policy_loss(
        spec,
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=advantages,
        weights=weights,
    )
    assert np.isclose(float(loss), expected.loss, rtol=1e-10)


@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan])
@pytest.mark.parametrize("spec", SPECS[:6], ids=lambda s: s.name)
def test_nonfinite_log_ratios_are_dropped_not_propagated(
    spec: ObjectiveSpec, bad: float
) -> None:
    """One poisoned token must not decide a step, and must be reported."""

    current = np.array([-1.0, bad, -0.5])
    behavior = np.array([-1.2, -1.0, -0.4])
    advantages = np.array([1.0, 1.0, -1.0])
    weights = np.ones(3)

    loss, terms = policy_terms(
        NumpyOps(),
        spec,
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=advantages,
        weights=weights,
    )
    assert np.isfinite(float(loss))
    assert float(terms["nonfinite_token_count"]) == 1.0
    assert float(terms["token_count"]) == 2.0

    # The surviving two tokens produce exactly the loss they would have alone.
    clean_loss, _ = policy_terms(
        NumpyOps(),
        spec,
        current_logprobs=current[[0, 2]],
        behavior_logprobs=behavior[[0, 2]],
        advantages=advantages[[0, 2]],
        weights=weights[[0, 2]],
    )
    assert np.isclose(float(loss), float(clean_loss), rtol=1e-10)

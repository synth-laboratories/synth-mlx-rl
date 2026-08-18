"""The pure-NumPy reference objectives, on their own terms."""

from __future__ import annotations

import numpy as np
import pytest

from synth_mlx_rl.objective_spec import ObjectiveError, ObjectiveSpec
from synth_mlx_rl.objectives import (
    cispo_loss,
    clipped_policy_loss,
    cross_entropy_loss,
    group_normalize_rewards,
    importance_sampling_loss,
    policy_loss,
)


def test_group_normalize_rewards() -> None:
    advantages = group_normalize_rewards([0.0, 1.0, 2.0, 3.0])
    assert np.isclose(advantages.mean(), 0.0, atol=1e-8)
    assert np.isclose(advantages.std(), 1.0, atol=2e-4)


def test_constant_group_has_no_signal() -> None:
    assert np.array_equal(group_normalize_rewards([1.0, 1.0]), np.zeros(2))


def test_cross_entropy_is_masked_mean_negative_logprob() -> None:
    metrics = cross_entropy_loss(
        np.array([-1.0, -3.0, -100.0]), np.array([1.0, 1.0, 0.0])
    )
    assert np.isclose(metrics.loss, 2.0)
    assert metrics.token_count == 2.0


def test_clipped_policy_loss_at_behavior_policy() -> None:
    behavior = np.array([-1.0, -2.0])
    metrics = clipped_policy_loss(
        behavior.copy(), behavior, np.array([1.0, -1.0]), np.array([1.0, 1.0])
    )
    assert np.isclose(metrics.loss, 0.0)
    assert np.isclose(metrics.mean_ratio, 1.0)
    assert np.isclose(metrics.clip_fraction, 0.0)
    assert np.isclose(metrics.approx_kl, 0.0)


def test_grpo_clipping_handles_positive_and_negative_advantages() -> None:
    metrics = clipped_policy_loss(
        np.log(np.array([2.0, 0.25])),
        np.zeros(2),
        np.array([1.0, -1.0]),
        np.ones(2),
        clip_epsilon=0.2,
    )
    # Positive advantage clips at 1.2. For a negative advantage the pessimistic
    # branch is 0.8 * -1, not 0.25 * -1. Mean surrogate = (1.2 - 0.8) / 2 = 0.2.
    assert np.isclose(metrics.policy_loss, -0.2)
    assert np.isclose(metrics.clip_fraction, 1.0)


def test_importance_sampling_is_unclipped() -> None:
    metrics = importance_sampling_loss(
        np.log(np.asarray([2.0, 0.5])),
        np.zeros(2),
        np.asarray([1.0, -1.0]),
        np.ones(2),
    )
    assert np.isclose(metrics.policy_loss, -0.75)
    assert np.isclose(metrics.clip_fraction, 0.0)


def test_cispo_minimax_refuses_an_active_lower_bound() -> None:
    with pytest.raises(ObjectiveError) as excinfo:
        ObjectiveSpec(name="cispo_minimax", eps_low=0.2, eps_high=4.0)
    message = str(excinfo.value)
    assert "single-sided" in message
    assert "cispo_two_sided" in message


def test_cispo_minimax_accepts_the_canonical_setting() -> None:
    spec = ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0)
    assert spec.clip_low == 0.0
    assert spec.clip_high == 5.0


def test_cispo_two_sided_permits_what_minimax_refuses() -> None:
    spec = ObjectiveSpec(name="cispo_two_sided", eps_low=0.2, eps_high=0.28)
    assert np.isclose(spec.clip_low, 0.8)
    assert np.isclose(spec.clip_high, 1.28)


def test_cispo_value_matches_the_written_formula() -> None:
    current = np.log(np.array([0.5, 0.25]))
    behavior = np.log(np.array([0.5, 0.5]))
    advantages = np.array([1.0, -2.0])
    weights = np.ones(2)
    metrics = cispo_loss(
        current, behavior, advantages, weights, eps_low=1.0, eps_high=4.0
    )
    ratio = np.exp(current - behavior)
    truncated = np.clip(ratio, 0.0, 5.0)
    expected = -float((truncated * advantages * current).mean())
    assert np.isclose(metrics.policy_loss, expected)


def test_cispo_two_sided_truncates_where_minimax_would_not() -> None:
    current = np.log(np.array([0.5]))
    behavior = np.log(np.array([1.0]))  # ratio 0.5, below a 0.8 lower bound
    kwargs = dict(
        current_logprobs=current,
        behavior_logprobs=behavior,
        advantages=np.array([1.0]),
        weights=np.ones(1),
    )
    single = policy_loss(ObjectiveSpec(name="cispo_minimax", eps_low=1.0), **kwargs)
    double = policy_loss(
        ObjectiveSpec(name="cispo_two_sided", eps_low=0.2, eps_high=0.28), **kwargs
    )
    assert single.clip_fraction == 0.0
    assert double.clip_fraction == 1.0
    assert not np.isclose(single.policy_loss, double.policy_loss)


def test_ppo_is_refused_by_name() -> None:
    with pytest.raises(ObjectiveError) as excinfo:
        ObjectiveSpec(name="ppo")
    assert "no value head" in str(excinfo.value)


def test_reference_kl_is_non_negative_k3() -> None:
    current = np.array([-1.0, -2.0])
    metrics = policy_loss(
        ObjectiveSpec(name="grpo", kl_beta=1.0),
        current_logprobs=current,
        behavior_logprobs=current.copy(),
        advantages=np.zeros(2),
        weights=np.ones(2),
        reference_logprobs=np.array([-3.0, -0.5]),
        reference_mask=np.ones(2),
    )
    assert metrics.reference_kl > 0.0

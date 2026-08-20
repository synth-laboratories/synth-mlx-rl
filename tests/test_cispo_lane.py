"""The on-policy lane's arithmetic and refusals, with no model involved.

These are the parts that must match the hosted Tinker runner exactly: if the
advantage definition or the datum layout differs, a local number and a hosted
number are not measuring the same thing, and comparing them is worse than having
only one of them.
"""

from __future__ import annotations

import math

import pytest

from synth_mlx_rl.models import RolloutTarget, TrainingConfig
from synth_mlx_rl.rewards import (
    RewardGroupError,
    has_learning_signal,
    normalize_group_rewards,
)
from synth_mlx_rl.rollout_client import RolloutAction, RolloutError
from synth_mlx_rl.runner import _datum_from_action
from synth_mlx_rl.schemas import Datum


def test_group_advantages_use_the_sample_standard_deviation() -> None:
    # slime's convention, which the hosted runner also uses: stdev with n-1 and
    # a 1e-6 floor. The population stdev would scale every advantage by
    # sqrt(n/(n-1)) and quietly change the effective step size.
    rewards = [1.0, 0.0, 0.0, 0.0]
    advantages = normalize_group_rewards(rewards)
    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-9)
    expected_denominator = 0.5 + 1e-6  # sample stdev of [1,0,0,0] is 0.5
    assert math.isclose(advantages[0], 0.75 / expected_denominator, rel_tol=1e-9)
    assert advantages[0] > 0 and all(value < 0 for value in advantages[1:])


def test_a_group_with_no_reward_variance_is_filtered_before_it_is_normalized() -> None:
    # Normalization alone does not protect against a constant group, and it is
    # worse than returning zeros: dividing float residue by a 1e-6 floor
    # amplifies it, so [0.4, 0.4, 0.4] comes back as advantages around 1e-11 --
    # a direction, derived from nothing. The variance check is the guard, and
    # the lane runs it before normalizing.
    assert not has_learning_signal([1.0, 1.0, 1.0])
    assert not has_learning_signal([0.0, 0.0])
    noise = normalize_group_rewards([0.4, 0.4, 0.4])
    assert all(abs(value) < 1e-6 for value in noise)
    assert any(value != 0.0 for value in noise)


def test_a_group_of_one_cannot_define_an_advantage() -> None:
    assert not has_learning_signal([1.0])
    with pytest.raises(RewardGroupError):
        normalize_group_rewards([1.0])


def test_non_finite_rewards_are_refused() -> None:
    assert not has_learning_signal([1.0, float("nan")])
    with pytest.raises(RewardGroupError):
        normalize_group_rewards([1.0, float("inf")])


def test_only_completion_tokens_carry_weight_and_advantage() -> None:
    action = RolloutAction(
        prompt_token_ids=(1, 2, 3), token_ids=(4, 5), log_probs=(-0.5, -0.25)
    )
    datum, completions = _datum_from_action(Datum, action, 1.5, sequence_cap=100)
    assert completions == 2
    # Pre-shifted: inputs are tokens[:-1], targets are tokens[1:].
    assert datum.input_ids == [1, 2, 3, 4]
    assert datum.target_ids == [2, 3, 4, 5]
    # The observation is scored but neither credited nor blamed for the reward.
    assert datum.weights == [0.0, 0.0, 1.0, 1.0]
    assert datum.advantages == [0.0, 0.0, 1.5, 1.5]
    # Behaviour log-probabilities sit in the completion positions, so the
    # importance ratio compares the same tokens under two policy versions.
    assert datum.behavior_logprobs == [0.0, 0.0, -0.5, -0.25]


def test_a_sequence_over_the_cap_is_refused_before_it_reaches_the_model() -> None:
    action = RolloutAction(
        prompt_token_ids=tuple(range(50)), token_ids=(1, 2), log_probs=(-0.1, -0.1)
    )
    with pytest.raises(RuntimeError, match="rollout_sequence_cap_exceeded"):
        _datum_from_action(Datum, action, 1.0, sequence_cap=8)


def test_a_token_receipt_that_does_not_line_up_is_refused() -> None:
    with pytest.raises(RolloutError, match="unaligned"):
        RolloutAction(
            prompt_token_ids=(1, 2), token_ids=(3, 4), log_probs=(-0.1,)
        ).validate()
    with pytest.raises(RolloutError, match="empty"):
        RolloutAction(prompt_token_ids=(), token_ids=(3,), log_probs=(-0.1,)).validate()


def test_the_lane_refuses_a_configuration_it_cannot_honour() -> None:
    target = RolloutTarget(url="http://127.0.0.1:8114", task_id="banking77")
    # An on-policy lane with no environment has no reward to learn from.
    with pytest.raises(ValueError, match="rollout target"):
        TrainingConfig(backend="cispo", output_dir="out")
    # cispo_minimax is single-sided by definition.
    with pytest.raises(ValueError, match="single-sided"):
        TrainingConfig(
            backend="cispo", output_dir="out", rollout=target, eps_low=0.2
        )
    # Greedy rollouts cannot differ, so a group of them has no signal.
    with pytest.raises(ValueError):
        RolloutTarget(url="http://x", task_id="t", temperature=0.0)
    # And the offline lane still needs its dataset.
    with pytest.raises(ValueError, match="dataset"):
        TrainingConfig(backend="qwen_lora", output_dir="out")


def test_a_valid_on_policy_configuration_is_accepted() -> None:
    config = TrainingConfig(
        backend="cispo",
        output_dir="out",
        rollout=RolloutTarget(url="http://127.0.0.1:8114", task_id="banking77"),
    )
    assert config.objective == "cispo_minimax"
    assert config.eps_low == 1.0 and config.eps_high == 4.0
    assert config.group_size >= 2
    assert config.dataset is None

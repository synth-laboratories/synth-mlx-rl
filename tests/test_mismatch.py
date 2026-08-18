"""The sampling/training mismatch stage.

Two ratios exist and they answer different questions. This module tests the
second one -- `exp(behavior - rollout)`, whether the sampler and the trainer
agree about what the same weights predict for the same tokens -- and the policy
that decides whether a disagreement is tolerable, correctable, or disqualifying.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_mlx_rl.mismatch import (
    MismatchError,
    MismatchPolicy,
    measure_mismatch,
    tis_weights,
)


def test_perfect_agreement_needs_no_correction() -> None:
    values = [-1.0, -2.0, -0.5, -3.25]
    report = MismatchPolicy().evaluate(
        measure_mismatch(behavior_logprobs=values, rollout_logprobs=values)
    )
    assert report.verdict == "ok"
    assert report.max_abs_diff == 0.0
    assert report.ess_ratio == 1.0
    assert report.reason is None


def test_small_drift_is_corrected_and_large_drift_is_refused() -> None:
    behavior = [-1.0, -2.0, -0.5]
    drifted = MismatchPolicy().evaluate(
        measure_mismatch(behavior_logprobs=behavior, rollout_logprobs=[-1.02, -2.01, -0.49])
    )
    assert drifted.verdict == "correct_with_tis"

    broken = MismatchPolicy().evaluate(
        measure_mismatch(behavior_logprobs=[-1.0, -8.0, -0.5], rollout_logprobs=behavior)
    )
    # Past a bound the collected data was not produced by the policy the trainer
    # is scoring, and TIS-weighting it yields a confident update in an arbitrary
    # direction. Refusing is the correct outcome, not a conservative one.
    assert broken.verdict == "refuse"
    assert "exceeds" in (broken.reason or "")


def test_misalignment_is_refused_rather_than_truncated() -> None:
    """Silently comparing the shorter of two arrays is how an off-by-one becomes
    a plausible-looking metric."""
    with pytest.raises(MismatchError, match="align one-to-one"):
        measure_mismatch(behavior_logprobs=[-1.0, -2.0], rollout_logprobs=[-1.0])


def test_an_all_masked_comparison_is_a_failure_not_a_zero() -> None:
    with pytest.raises(MismatchError, match="failed collection"):
        measure_mismatch(
            behavior_logprobs=[-1.0, -2.0],
            rollout_logprobs=[-1.0, -2.0],
            weights=[0.0, 0.0],
        )


def test_nonfinite_tokens_disqualify_by_default() -> None:
    report = measure_mismatch(
        behavior_logprobs=[-1.0, float("-inf"), -0.5],
        rollout_logprobs=[-1.0, -2.0, -0.5],
    )
    assert report.nonfinite_token_count == 1
    # Excluded from the statistics rather than poisoning them...
    assert report.max_abs_diff == 0.0
    # ...but still disqualifying, because a nonfinite logprob is a broken
    # forward pass, not a token that happened to be unlikely.
    assert MismatchPolicy().evaluate(report).verdict == "refuse"


def test_ess_falls_when_a_few_tokens_dominate() -> None:
    """One token carrying almost all the weight is not a small mismatch, however
    good the mean looks."""
    behavior = [-1.0] * 9 + [4.0]
    rollout = [-1.0] * 9 + [-1.0]
    report = measure_mismatch(behavior_logprobs=behavior, rollout_logprobs=rollout)
    assert report.ess_ratio < 0.5
    judged = MismatchPolicy(max_abs_diff=100.0).evaluate(report)
    assert judged.verdict == "refuse" and "effective sample size" in (judged.reason or "")


def test_tis_clamps_and_reports_the_preclamp_ratio() -> None:
    result = tis_weights(
        behavior_logprobs=[-1.0, -2.0, -1.0],
        rollout_logprobs=[-1.5, -2.0, 1.0],
        clip_low=0.5,
        clip_high=1.5,
    )
    assert result.weights[0] == pytest.approx(1.5)   # exp(0.5)=1.6487 -> clamped
    assert result.weights[1] == pytest.approx(1.0)   # exact agreement
    assert result.weights[2] == pytest.approx(0.5)   # exp(-2)=0.135 -> clamped
    assert result.clipped_fraction == pytest.approx(2 / 3)
    # Metrics report the PRE-clamp ratio, so a clamp doing heavy lifting stays
    # visible rather than hiding behind its own output. Here the post-clamp mean
    # is exactly (1.5 + 1.0 + 0.5)/3 = 1.0, which would look like perfect
    # agreement; the pre-clamp mean is 0.928 and does not.
    assert result.weights.mean() == pytest.approx(1.0)
    assert result.metrics["tis"] == pytest.approx(
        (np.exp(0.5) + 1.0 + np.exp(-2.0)) / 3
    )
    assert result.metrics["tis"] != pytest.approx(result.weights.mean())
    assert result.metrics["tis_clipfrac"] == pytest.approx(2 / 3)


def test_tis_bounds_must_bracket_one() -> None:
    with pytest.raises(MismatchError, match="clip_low <= 1 <= clip_high"):
        tis_weights(
            behavior_logprobs=[-1.0], rollout_logprobs=[-1.0], clip_low=1.2, clip_high=1.5
        )


def test_a_nonfinite_token_contributes_zero_weight_not_a_nan() -> None:
    result = tis_weights(
        behavior_logprobs=[-1.0, float("nan")], rollout_logprobs=[-1.0, -2.0]
    )
    assert result.weights[1] == 0.0
    assert np.isfinite(result.weights).all()


def test_tis_is_a_separate_stage_from_the_objective_clip() -> None:
    """Applying TIS must scale the loss without touching the policy gradient's
    own clipping decision. A codebase that folds them together can report
    neither honestly."""
    from synth_mlx_rl.backends import NumpyOps
    from synth_mlx_rl.kernel import policy_terms
    from synth_mlx_rl.objective_spec import ObjectiveSpec

    spec = ObjectiveSpec(name="grpo", clip_epsilon=0.2)
    common = dict(
        current_logprobs=np.array([-1.0, -2.0]),
        behavior_logprobs=np.array([-1.0, -2.0]),
        advantages=np.array([1.0, 1.0]),
        weights=np.array([1.0, 1.0]),
    )
    ops = NumpyOps()
    _, plain = policy_terms(ops, spec, **common)
    _, weighted = policy_terms(
        ops, spec, **common, tis_weights=np.array([0.5, 0.5])
    )
    assert float(weighted["policy_loss"]) == pytest.approx(float(plain["policy_loss"]) * 0.5)
    # The clip fraction is a property of the policy ratio and must not move.
    assert float(weighted["clip_fraction"]) == float(plain["clip_fraction"])

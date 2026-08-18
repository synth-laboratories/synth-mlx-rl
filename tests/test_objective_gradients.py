"""Where the gradient goes.

CISPO's whole claim is that the gradient reaches the policy only through
``current_logprobs``, with the importance ratio detached. If a stop-gradient is
missing, the run still produces numbers -- just numbers from a different
objective. So the claim is checked directly, by evaluating the same kernel
source over dual numbers and reading the derivative.

Seeding input ``x`` with direction ``d`` and reading the loss's tangent gives
the directional derivative ``<grad_x loss, d>``. Zero for every direction means
no gradient flows.
"""

from __future__ import annotations

import numpy as np
import pytest

from synth_mlx_rl.kernel import policy_terms
from synth_mlx_rl.objective_spec import ObjectiveSpec
from synth_mlx_rl.testing.autodiff import Dual, DualOps

CISPO = ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0)
CISPO_TWO_SIDED = ObjectiveSpec(name="cispo_two_sided", eps_low=0.2, eps_high=0.28)
GRPO = ObjectiveSpec(name="grpo", clip_epsilon=0.2)
IS = ObjectiveSpec(name="importance_sampling")

ALL_SPECS = [CISPO, CISPO_TWO_SIDED, GRPO, IS]


def _inputs():
    current = np.array([-1.0, -0.4, -2.2, -0.9])
    behavior = np.array([-1.0, -2.0, -0.3, -0.95])
    advantages = np.array([1.0, -1.0, 2.0, 0.5])
    weights = np.array([1.0, 1.0, 1.0, 0.0])
    reference = np.array([-1.1, -0.5, -2.0, -1.0])
    return current, behavior, advantages, weights, reference


def _directional_derivative(spec, *, seed_on: str, direction, kl_beta_reference=True):
    current, behavior, advantages, weights, reference = _inputs()
    values = {
        "current": current,
        "behavior": behavior,
        "advantages": advantages,
        "reference": reference,
    }
    duals = {
        name: Dual.seed(value, direction if name == seed_on else np.zeros_like(value))
        for name, value in values.items()
    }
    loss, _ = policy_terms(
        DualOps(),
        spec,
        current_logprobs=duals["current"],
        behavior_logprobs=duals["behavior"],
        advantages=duals["advantages"],
        weights=Dual(weights),
        reference_logprobs=duals["reference"] if kl_beta_reference else None,
        reference_mask=Dual(np.ones_like(weights)) if kl_beta_reference else None,
    )
    return float(loss.tangent)


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
@pytest.mark.parametrize("blocked", ["behavior", "advantages", "reference"])
def test_no_gradient_through_data_inputs(spec, blocked) -> None:
    """behavior, rollout, reference, and advantages are data, not parameters."""

    for index in range(4):
        direction = np.zeros(4)
        direction[index] = 1.0
        derivative = _directional_derivative(spec, seed_on=blocked, direction=direction)
        assert derivative == 0.0, (
            f"{spec.name}: gradient leaked into {blocked} at position {index}"
        )


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: s.name)
def test_gradient_flows_through_current_logprobs(spec) -> None:
    direction = np.array([1.0, 1.0, 1.0, 0.0])
    derivative = _directional_derivative(spec, seed_on="current", direction=direction)
    assert derivative != 0.0


def test_cispo_gradient_equals_the_written_formula() -> None:
    """d loss / d current_i = -sg(clip(ratio_i)) * A_i * w_i / sum(w)."""

    current, behavior, advantages, weights, _ = _inputs()
    ratio = np.exp(current - behavior)
    truncated = np.clip(ratio, CISPO.clip_low, CISPO.clip_high)
    denominator = weights.sum()

    for index in range(4):
        direction = np.zeros(4)
        direction[index] = 1.0
        derivative = _directional_derivative(
            CISPO, seed_on="current", direction=direction, kl_beta_reference=False
        )
        expected = -truncated[index] * advantages[index] * weights[index] / denominator
        # rtol is float32-scale: the kernel computes the log-ratio in float32
        # on purpose, and this expectation is float64.
        assert np.isclose(derivative, expected, rtol=1e-6, atol=1e-12)


def test_cispo_keeps_gradient_on_a_clipped_token() -> None:
    """The defining property: a truncated token still contributes gradient.

    Under PPO-style clipping the clipped branch is flat, so a token outside the
    trust region contributes nothing. Under CISPO the ratio is detached and the
    gradient rides on ``current_logprobs``, so the token keeps pushing -- with a
    bounded weight. This is the difference the two names exist to mark.
    """

    # ratio = exp(3) ~ 20.1: far above the cispo_two_sided upper bound of 1.28
    # and above the grpo bound of 1.2.
    current = np.array([0.0])
    behavior = np.array([-3.0])
    advantages = np.array([1.0])
    weights = np.array([1.0])

    def derivative(spec):
        loss, _ = policy_terms(
            DualOps(),
            spec,
            current_logprobs=Dual.seed(current, np.ones(1)),
            behavior_logprobs=Dual(behavior),
            advantages=Dual(advantages),
            weights=Dual(weights),
        )
        return float(loss.tangent)

    grpo = derivative(GRPO)
    cispo = derivative(CISPO_TWO_SIDED)

    assert grpo == 0.0, "PPO-style clipping should flatten a clipped token"
    assert cispo != 0.0, "CISPO must keep gradient on a clipped token"
    assert np.isclose(cispo, -1.28), "gradient weight is the truncated ratio"


def test_cispo_gradient_is_bounded_by_the_truncation() -> None:
    """An enormous ratio cannot produce an enormous gradient."""

    huge = np.array([0.0])
    tiny_behavior = np.array([-50.0])
    loss, _ = policy_terms(
        DualOps(),
        CISPO,
        current_logprobs=Dual.seed(huge, np.ones(1)),
        behavior_logprobs=Dual(tiny_behavior),
        advantages=Dual(np.array([1.0])),
        weights=Dual(np.array([1.0])),
    )
    assert np.isclose(float(loss.tangent), -CISPO.clip_high)


def test_nonfinite_token_contributes_no_gradient() -> None:
    current = np.array([-1.0, np.nan])
    loss, _ = policy_terms(
        DualOps(),
        CISPO,
        current_logprobs=Dual.seed(current, np.array([0.0, 1.0])),
        behavior_logprobs=Dual(np.array([-1.2, -1.0])),
        advantages=Dual(np.array([1.0, 1.0])),
        weights=Dual(np.ones(2)),
    )
    assert float(loss.tangent) == 0.0
    assert np.isfinite(float(loss.value))


def test_dual_backend_reproduces_the_numpy_values() -> None:
    """The autodiff backend must not be a different objective."""

    from synth_mlx_rl.backends import NumpyOps

    current, behavior, advantages, weights, reference = _inputs()
    for spec in ALL_SPECS:
        kwargs = dict(
            current_logprobs=current,
            behavior_logprobs=behavior,
            advantages=advantages,
            weights=weights,
            reference_logprobs=reference,
            reference_mask=np.ones_like(weights),
        )
        numpy_loss, _ = policy_terms(NumpyOps(), spec, **kwargs)
        dual_loss, _ = policy_terms(
            DualOps(),
            spec,
            **{
                key: (Dual(value) if value is not None else None)
                for key, value in kwargs.items()
            },
        )
        assert np.isclose(float(numpy_loss), float(dual_loss.value), rtol=1e-12)

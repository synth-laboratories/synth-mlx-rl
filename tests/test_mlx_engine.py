"""Tests that need a real MLX runtime.

Every one of these is skipped on a host without MLX, which today is every host
this package is developed on (finalized plan, correction C11). They are the
milestone-0 checklist, not a regression suite: nothing here has ever run.

The portable suite deliberately covers the same *logic* through the fake engine
and the NumPy/dual backends, so what remains unchecked until this file runs is
specifically the MLX bindings -- array semantics, autograd, and the adapter swap.
"""

from __future__ import annotations

import pytest

try:  # pragma: no cover - depends on the host
    import mlx.core as mlx
    import mlx_lm  # noqa: F401

    HAS_MLX = True
except Exception:  # pragma: no cover - the normal case here
    mlx = None
    HAS_MLX = False

# Skipped per test rather than at import, so a run *reports* what it did not
# check. A module that silently collects zero tests looks like a passing suite.
pytestmark = [
    pytest.mark.mlx,
    pytest.mark.skipif(
        not HAS_MLX,
        reason="requires a working mlx runtime (Apple Silicon); not installable here",
    ),
]


@pytest.fixture(scope="module")
def mlx_engine():
    from synth_mlx_rl.config import Settings
    from synth_mlx_rl.engine import MLXEngine

    return MLXEngine(Settings(lora_rank=8, max_seq_length=512))


def test_engine_loads_and_publishes_an_initial_snapshot(mlx_engine) -> None:
    state = mlx_engine.state()
    assert state.ready
    assert state.trainable_parameters > 0
    assert state.latest_policy_snapshot_id is not None


def test_mlx_kernel_matches_the_numpy_reference() -> None:
    """The parity claim, finally checked on the array library that runs it."""

    import numpy as np

    from synth_mlx_rl.backends import MlxOps
    from synth_mlx_rl.kernel import policy_terms
    from synth_mlx_rl.objective_spec import ObjectiveSpec
    from synth_mlx_rl.objectives import policy_loss

    rng = np.random.default_rng(0)
    behavior = rng.normal(-1.5, 1.0, 32)
    current = behavior + rng.normal(0.0, 1.5, 32)
    advantages = rng.normal(0.0, 1.0, 32)
    weights = np.ones(32)

    for spec in (
        ObjectiveSpec(name="grpo"),
        ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0),
        ObjectiveSpec(name="cispo_two_sided", eps_low=0.2, eps_high=0.28),
        ObjectiveSpec(name="importance_sampling"),
    ):
        expected = policy_loss(
            spec,
            current_logprobs=current,
            behavior_logprobs=behavior,
            advantages=advantages,
            weights=weights,
        )
        loss, _ = policy_terms(
            MlxOps(mlx),
            spec,
            current_logprobs=mlx.array(current.astype("float32")),
            behavior_logprobs=mlx.array(behavior.astype("float32")),
            advantages=mlx.array(advantages.astype("float32")),
            weights=mlx.array(weights.astype("float32")),
        )
        assert abs(float(loss.item()) - expected.loss) < 1e-4, spec.name


def test_mlx_stop_gradient_blocks_the_cispo_denominator() -> None:
    """The gradient claim, on MLX autograd rather than dual numbers."""

    from synth_mlx_rl.backends import MlxOps
    from synth_mlx_rl.kernel import policy_terms
    from synth_mlx_rl.objective_spec import ObjectiveSpec

    spec = ObjectiveSpec(name="cispo_minimax", eps_low=1.0, eps_high=4.0)
    behavior = mlx.array([-1.0, -2.0], dtype=mlx.float32)
    advantages = mlx.array([1.0, -1.0], dtype=mlx.float32)
    weights = mlx.array([1.0, 1.0], dtype=mlx.float32)

    def loss_of_current(current):
        loss, _ = policy_terms(
            MlxOps(mlx),
            spec,
            current_logprobs=current,
            behavior_logprobs=behavior,
            advantages=advantages,
            weights=weights,
        )
        return loss

    def loss_of_behavior(b):
        loss, _ = policy_terms(
            MlxOps(mlx),
            spec,
            current_logprobs=mlx.array([-1.1, -1.9], dtype=mlx.float32),
            behavior_logprobs=b,
            advantages=advantages,
            weights=weights,
        )
        return loss

    grad_current = mlx.grad(loss_of_current)(
        mlx.array([-1.1, -1.9], dtype=mlx.float32)
    )
    grad_behavior = mlx.grad(loss_of_behavior)(behavior)
    mlx.eval(grad_current, grad_behavior)

    assert any(abs(float(v)) > 0 for v in grad_current.tolist())
    assert all(abs(float(v)) == 0 for v in grad_behavior.tolist())


def test_sampling_is_pinned_across_a_training_step(mlx_engine) -> None:
    """The D2 claim on real weights: a step must not move a pinned snapshot."""

    from synth_mlx_rl.schemas import (
        AdamParams,
        Datum,
        ForwardBackwardRequest,
        SampleRequest,
    )

    pinned = mlx_engine.publish_snapshot()
    before = mlx_engine.sample(
        SampleRequest(
            prompt="2 + 2 =",
            max_tokens=8,
            temperature=0.0,
            policy_snapshot_id=pinned.id,
        )
    )
    tokens = mlx_engine.encode("hello world")
    mlx_engine.forward_backward(
        ForwardBackwardRequest(
            data=[
                Datum(
                    input_ids=tokens[:-1],
                    target_ids=tokens[1:],
                    weights=[1.0] * (len(tokens) - 1),
                )
            ]
        )
    )
    mlx_engine.optim_step(AdamParams(learning_rate=1e-3))

    after = mlx_engine.sample(
        SampleRequest(
            prompt="2 + 2 =",
            max_tokens=8,
            temperature=0.0,
            policy_snapshot_id=pinned.id,
        )
    )
    assert (
        before.samples[0].completion_token_ids
        == after.samples[0].completion_token_ids
    ), "a pinned snapshot must not move underneath a training step"
    assert mlx_engine.state().training_version > pinned.training_version


def test_rollout_logprobs_come_from_the_untruncated_distribution(mlx_engine) -> None:
    """top_p must not change the recorded log-probability of a chosen token."""

    from synth_mlx_rl.schemas import SampleRequest

    wide = mlx_engine.sample(
        SampleRequest(prompt="hello", max_tokens=4, temperature=0.0, top_p=1.0, seed=7)
    ).samples[0]
    narrow = mlx_engine.sample(
        SampleRequest(prompt="hello", max_tokens=4, temperature=0.0, top_p=0.1, seed=7)
    ).samples[0]
    assert wide.completion_token_ids == narrow.completion_token_ids
    for a, b in zip(wide.rollout_logprobs, narrow.rollout_logprobs):
        assert abs(a - b) < 1e-5


def test_a_snapshot_survives_an_adapter_swap_round_trip(mlx_engine) -> None:
    """Restoring the training adapter must not corrupt it."""

    from synth_mlx_rl.schemas import SampleRequest

    snapshot = mlx_engine.publish_snapshot()
    live_before = mlx_engine._capture_adapter()
    mlx_engine.sample(
        SampleRequest(
            prompt="hello", max_tokens=2, policy_snapshot_id=snapshot.id
        )
    )
    live_after = mlx_engine._capture_adapter()
    assert set(live_before) == set(live_after)
    for name in live_before:
        assert bool(
            mlx.all(mlx.isclose(live_before[name], live_after[name])).item()
        ), name

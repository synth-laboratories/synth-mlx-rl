"""Policy snapshots: immutable, pinned, and never silently substituted."""

from __future__ import annotations

import pytest

from synth_mlx_rl.snapshots import (
    SnapshotEvictedError,
    SnapshotNotFoundError,
    SnapshotPool,
)


def _publish(pool: SnapshotPool, version: int, payload=None):
    return pool.publish(
        payload=payload if payload is not None else {"w": float(version)},
        training_version=version,
        step=version,
        base_model="Qwen/Qwen3.5-0.8B",
        lora_rank=8,
        lora_scale=2.0,
        tokenizer_digest="tok",
        template_digest="tpl",
    )


def test_ids_are_unique_and_latest_tracks_publication_order() -> None:
    pool = SnapshotPool(capacity=4)
    first = _publish(pool, 1)
    second = _publish(pool, 2)
    assert first.id != second.id
    assert pool.latest().id == second.id


def test_resolve_none_returns_the_newest_snapshot() -> None:
    pool = SnapshotPool(capacity=4)
    _publish(pool, 1)
    newest = _publish(pool, 2)
    assert pool.resolve(None).id == newest.id


def test_resolve_none_refuses_when_nothing_is_published() -> None:
    pool = SnapshotPool(capacity=2)
    with pytest.raises(SnapshotNotFoundError):
        pool.resolve(None)


def test_evicted_snapshot_fails_loudly_and_never_falls_back() -> None:
    pool = SnapshotPool(capacity=2)
    oldest = _publish(pool, 1)
    _publish(pool, 2)
    _publish(pool, 3)  # pushes `oldest` out

    with pytest.raises(SnapshotEvictedError) as excinfo:
        pool.resolve(oldest.id)
    assert "will not fall back" in str(excinfo.value)
    # And it is a *different* error from an id that never existed.
    with pytest.raises(SnapshotNotFoundError):
        pool.resolve("snap_never_issued")


def test_explicit_eviction_is_also_a_tombstone() -> None:
    pool = SnapshotPool(capacity=4)
    snapshot = _publish(pool, 1)
    pool.evict(snapshot.id)
    with pytest.raises(SnapshotEvictedError):
        pool.get(snapshot.id)


def test_a_published_snapshot_is_a_copy_not_an_alias(engine) -> None:
    """The point of D2: a step must not change what a snapshot holds.

    Checked through what the snapshot *answers*, not through its weights. The
    payload holds MLX arrays that are only valid on the engine's own thread, so
    reading them from here raises "There is no Stream(gpu, N) in current
    thread" -- and a test that reaches around the engine to inspect them is
    asserting on an implementation detail anyway.
    """

    from synth_mlx_rl.schemas import (
        AdamParams,
        Datum,
        ForwardBackwardRequest,
    )

    tokens = [1, 2, 3, 4]
    snapshot = engine.publish_snapshot()
    pinned_before = engine.score_logprobs(tokens, policy_snapshot_id=snapshot.id)

    engine.forward_backward(
        ForwardBackwardRequest(
            data=[Datum(input_ids=[1, 2], target_ids=[2, 3], weights=[1.0, 1.0])]
        )
    )
    engine.optim_step(AdamParams(learning_rate=0.1))

    # The pinned snapshot answers exactly as it did before the step...
    assert engine.score_logprobs(tokens, policy_snapshot_id=snapshot.id) == pinned_before
    # ...and the live policy has actually moved, or the check above proves nothing.
    assert engine.score_logprobs(tokens, policy_snapshot_id=None) != pinned_before

def test_optim_step_bumps_the_training_version_without_publishing(engine) -> None:
    from synth_mlx_rl.schemas import AdamParams, Datum, ForwardBackwardRequest

    before_version = engine.state().training_version
    before_count = len(engine.snapshots)
    latest_before = engine.snapshots.latest().id

    engine.forward_backward(
        ForwardBackwardRequest(
            data=[Datum(input_ids=[1, 2], target_ids=[2, 3], weights=[1.0, 1.0])]
        )
    )
    engine.optim_step(AdamParams())

    assert engine.state().training_version == before_version + 1
    assert len(engine.snapshots) == before_count
    assert engine.snapshots.latest().id == latest_before


def test_a_pinned_sample_is_unaffected_by_a_step(engine) -> None:
    from synth_mlx_rl.schemas import (
        AdamParams,
        Datum,
        ForwardBackwardRequest,
        SampleRequest,
    )

    pinned = engine.publish_snapshot()
    engine.forward_backward(
        ForwardBackwardRequest(
            data=[Datum(input_ids=[1, 2], target_ids=[2, 3], weights=[1.0, 1.0])]
        )
    )
    engine.optim_step(AdamParams())

    response = engine.sample(
        SampleRequest(prompt="hello", policy_snapshot_id=pinned.id)
    )
    sample = response.samples[0]
    assert sample.policy_snapshot_id == pinned.id
    assert sample.training_version == pinned.training_version
    assert engine.state().training_version > pinned.training_version

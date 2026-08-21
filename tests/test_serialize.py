"""Engine calls are pinned to one thread, and so is engine construction.

MLX streams are thread-local. A model built on one thread cannot be evaluated on
another, and FastAPI runs `def` endpoints in a worker pool -- so this is not a
tidiness property, it is the difference between a working service and a 500 from
inside `mx.eval` with no hint about which thread boundary caused it.
"""

from __future__ import annotations

import threading

import pytest

from synth_mlx_rl.serialize import SingleThreadEngine


class _ThreadRecorder:
    """Records the thread each call ran on, like an MLX engine would care."""

    def __init__(self) -> None:
        self.built_on = threading.get_ident()
        self.calls: list[int] = []

    def state(self):
        self.calls.append(threading.get_ident())
        return "state"

    def sample(self, *_args, **_kwargs):
        self.calls.append(threading.get_ident())
        return "sample"

    def forward_backward(self, *_args, **_kwargs):
        self.calls.append(threading.get_ident())
        return "fb"

    settings = "plain-data"


def test_construction_happens_on_the_worker_not_the_caller() -> None:
    """Wrapping an already-built engine is not enough: a model carries the
    streams of whichever thread created it, and dispatching afterwards does not
    move them."""
    caller = threading.get_ident()
    engine = SingleThreadEngine(factory=_ThreadRecorder)
    try:
        assert engine._engine.built_on != caller
    finally:
        engine.shutdown()


def test_every_proxied_call_runs_on_that_same_thread() -> None:
    engine = SingleThreadEngine(factory=_ThreadRecorder)
    try:
        engine.state()
        engine.sample()
        engine.forward_backward()
        threads = set(engine._engine.calls)
        assert len(threads) == 1, f"calls spread across {len(threads)} threads"
        assert threads == {engine._engine.built_on}
    finally:
        engine.shutdown()


def test_calls_from_many_caller_threads_still_land_on_one_worker() -> None:
    """The failure mode in production: concurrent requests, each on a different
    anyio worker."""
    engine = SingleThreadEngine(factory=_ThreadRecorder)
    try:
        callers = [threading.Thread(target=engine.sample) for _ in range(8)]
        for thread in callers:
            thread.start()
        for thread in callers:
            thread.join()
        assert set(engine._engine.calls) == {engine._engine.built_on}
    finally:
        engine.shutdown()


def test_plain_attributes_are_not_dispatched() -> None:
    """Sending an attribute read through the worker would deadlock if it were
    ever performed from inside the worker itself."""
    engine = SingleThreadEngine(factory=_ThreadRecorder)
    try:
        assert engine.settings == "plain-data"
    finally:
        engine.shutdown()


def test_exactly_one_of_engine_or_factory() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        SingleThreadEngine(_ThreadRecorder(), factory=_ThreadRecorder)
    with pytest.raises(ValueError, match="exactly one"):
        SingleThreadEngine()


def test_concurrent_same_prompt_samples_become_one_generate_batch() -> None:
    """Two num_samples=1 calls that share a prompt must merge into generate_batch."""

    from synth_mlx_rl.schemas import Sample, SampleRequest, SampleResponse

    class _BatchRecorder:
        def __init__(self) -> None:
            self.generate_batch_calls = 0
            self.sample_widths: list[int] = []

        def generate_batch(self, token_ids, request):
            self.generate_batch_calls += 1
            return [([7], [-0.2], "stop")] * request.num_samples

        def sample(self, request, **_kwargs):
            self.sample_widths.append(request.num_samples)
            if request.num_samples > 1:
                self.generate_batch(request.prompt_token_ids, request)
            dummy = Sample(
                text="ok",
                prompt_token_ids=list(request.prompt_token_ids or [1]),
                completion_token_ids=[7],
                rollout_logprobs=[-0.2],
                finish_reason="stop",
                policy_snapshot_id="snap",
                training_version=0,
                proxy_request_id="prid",
            )
            return SampleResponse(samples=[dummy] * request.num_samples)

    inner = _BatchRecorder()
    engine = SingleThreadEngine(engine=inner, sample_coalesce_s=0.05)
    request = SampleRequest(prompt_token_ids=[1, 2, 3], num_samples=1, max_tokens=8)
    barrier = threading.Barrier(2)
    results: list = []

    def _worker() -> None:
        barrier.wait()
        results.append(engine.sample(request))

    try:
        threads = [threading.Thread(target=_worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == 2
        assert all(len(response.samples) == 1 for response in results)
        assert inner.generate_batch_calls == 1
        assert inner.sample_widths == [2]
    finally:
        engine.shutdown()


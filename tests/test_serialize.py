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

    def register_policy(self, *_args, **_kwargs):
        self.calls.append(threading.get_ident())
        return "registered"

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
        engine.register_policy()
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

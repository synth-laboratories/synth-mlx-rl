"""Pin every engine call to one thread.

MLX streams are **thread-local**. A model built on one thread cannot be
evaluated on another: MLX raises `RuntimeError: There is no Stream(gpu, N) in
current thread`. FastAPI runs `def` endpoints in an anyio worker pool, so a
service that builds its engine at startup and then serves requests hits this on
the first sampling call -- and it surfaces as a 500 from deep inside `mx.eval`,
nowhere near the thread boundary that caused it.

An `RLock` does not help. It serializes access, which is a different property
from running on one particular thread.

`SingleThreadEngine` funnels every engine call through one dedicated worker --
**including construction**. Wrapping an already-built engine is not enough: the
model must be created on the same thread that later evaluates it, so the
MLX path must be handed a factory rather than an instance.
That is not a compromise for this service, it is the design: there is one
resident model, generation caches and optimizer state are shared mutable state,
and operations were already meant to be linearizable. Doing them all on one
thread makes stream affinity automatic instead of a rule to remember, and keeps
the event loop free so health checks answer while a generation is in flight.

The worker is a daemon thread and the executor is shut down on app teardown, so
it cannot outlive the process holding the model.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

#: Every method of `LearnerEngine` that touches MLX state.
_PROXIED = (
    "state",
    "encode",
    "decode",
    "render_chat",
    "sample",
    "score_logprobs",
    "forward_backward",
    "optim_step",
    "zero_grad",
    "publish_snapshot",
    "resolve_snapshot",
    "save_checkpoint",
    "load_checkpoint",
)


class SingleThreadEngine:
    """A `LearnerEngine` whose every call runs on one dedicated thread."""

    def __init__(
        self,
        engine: Any | None = None,
        *,
        factory: Callable[[], Any] | None = None,
        thread_name: str = "mlx-engine",
    ):
        if (engine is None) == (factory is None):
            raise ValueError("pass exactly one of engine= or factory=")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name
        )
        # Build on the worker when given a factory. An engine constructed on the
        # caller's thread carries that thread's streams with it, and no amount
        # of dispatching afterwards moves them.
        self._engine = engine if factory is None else self._submit(factory)

    def _submit(self, fn: Callable[[], T]) -> T:
        future: Future[T] = self._executor.submit(fn)
        return future.result()

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._engine, name)
        if name not in _PROXIED or not callable(attribute):
            # Plain data (`rollouts`, `settings`, `snapshots`) is read directly:
            # sending an attribute read through the worker would deadlock if it
            # were ever performed from inside the worker itself.
            return attribute

        def call(*args: Any, **kwargs: Any) -> Any:
            return self._submit(lambda: attribute(*args, **kwargs))

        call.__name__ = name
        return call

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

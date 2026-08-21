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

import time
import threading
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

T = TypeVar("T")

#: How long the engine worker waits to merge concurrent `sample()` calls that
#: share a prompt. The wait is on the GPU thread on purpose: it preserves
#: queue order against later `forward_backward` / `optim_step` calls. Distinct
#: prompts still run sequentially after the window; that is not a bug.
_DEFAULT_SAMPLE_COALESCE_S = 0.008

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


def _sample_coalesce_key(request: Any) -> tuple[Any, ...] | None:
    """Group concurrent `sample()` calls that can share one `generate_batch`.

    Returns None when the call is not a SampleRequest-shaped object (the
    serialize tests call `sample()` with no arguments).
    """

    if getattr(request, "num_samples", None) is None:
        return None
    if not any(
        getattr(request, name, None) is not None
        for name in ("prompt", "messages", "prompt_token_ids")
    ):
        return None
    ids = getattr(request, "prompt_token_ids", None)
    if ids is not None:
        prompt_key: tuple[Any, ...] = ("ids", tuple(ids))
    else:
        messages = getattr(request, "messages", None)
        if messages is not None:
            prompt_key = (
                "messages",
                tuple(
                    (getattr(message, "role", None), getattr(message, "content", None))
                    for message in messages
                ),
            )
        else:
            prompt_key = ("prompt", getattr(request, "prompt", None))
    stop = getattr(request, "stop", None)
    if isinstance(stop, str):
        stop_key: tuple[str, ...] | None = (stop,)
    elif stop is not None:
        stop_key = tuple(stop)
    else:
        stop_key = None
    stop_ids = getattr(request, "stop_token_ids", None)
    tools = getattr(request, "tools", None)
    return (
        prompt_key,
        getattr(request, "policy_snapshot_id", None),
        getattr(request, "max_tokens", None),
        getattr(request, "temperature", None),
        getattr(request, "top_p", None),
        getattr(request, "min_p", None),
        getattr(request, "top_k", None),
        # Seed is excluded on purpose. CISPO group members share a prompt and
        # must land in one generate_batch; per-request seeds would split them
        # back into serial decodes. Diversity comes from batched sampling.
        stop_key,
        tuple(stop_ids) if stop_ids is not None else None,
        getattr(request, "enable_thinking", None),
        getattr(request, "add_generation_prompt", None),
        str(tools) if tools is not None else None,
    )


class SingleThreadEngine:
    """A `LearnerEngine` whose every call runs on one dedicated thread."""

    def __init__(
        self,
        engine: Any | None = None,
        *,
        factory: Callable[[], Any] | None = None,
        thread_name: str = "mlx-engine",
        lazy: bool = False,
        sample_coalesce_s: float = _DEFAULT_SAMPLE_COALESCE_S,
    ):
        if (engine is None) == (factory is None):
            raise ValueError("pass exactly one of engine= or factory=")
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=thread_name
        )
        # Keep factory construction lazy so capability and preflight routes can
        # refuse a missing managed model before MLX imports or allocates. The
        # first operation still builds on this worker, preserving stream
        # affinity.
        self._engine = engine
        self._factory = factory
        self._build_lock = threading.Lock()
        self._sample_coalesce_s = max(0.0, float(sample_coalesce_s))
        self._coalesce_lock = threading.Lock()
        self._pending_samples: list[tuple[tuple[Any, ...], dict[str, Any], Future[Any]]] = []
        self._flush_scheduled = False
        self._closed = False
        if factory is not None and not lazy:
            self._engine = self._submit(factory)
            self._factory = None

    def _submit(self, fn: Callable[[], T]) -> T:
        future: Future[T] = self._executor.submit(fn)
        return future.result()

    def _get_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._build_lock:
            if self._engine is None:
                assert self._factory is not None
                self._engine = self._submit(self._factory)
                self._factory = None
        return self._engine

    def _coalesced_sample(self, *args: Any, **kwargs: Any) -> Any:
        """Merge concurrent same-prompt samples into one `generate_batch`.

        FastAPI runs `def sample` in a thread pool, so many `num_samples=1`
        calls can be in flight at once. Without this they queue independently
        on the engine worker and never hit `EngineBase.sample`'s batch path.
        The merge itself still runs on that one worker; MLX is never called
        from the caller threads or a timer thread.
        """

        self._get_engine()
        result: Future[Any] = Future()
        with self._coalesce_lock:
            if self._closed:
                raise RuntimeError("engine is shut down")
            self._pending_samples.append((args, kwargs, result))
            should_schedule = not self._flush_scheduled
            if should_schedule:
                self._flush_scheduled = True
        if should_schedule:
            self._executor.submit(self._flush_samples)
        return result.result()

    def _flush_samples(self) -> None:
        # Already on the engine worker. Do not `_submit` from here: the pool
        # has one thread and that would deadlock.
        if self._sample_coalesce_s > 0:
            with self._coalesce_lock:
                pending_n = len(self._pending_samples)
                wait = pending_n < 2 and any(
                    _sample_coalesce_key(args[0] if args else None) is not None
                    for args, _kwargs, _future in self._pending_samples
                )
            if wait:
                time.sleep(self._sample_coalesce_s)
        with self._coalesce_lock:
            batch = self._pending_samples
            self._pending_samples = []
            self._flush_scheduled = False
        if self._closed:
            for _args, _kwargs, future in batch:
                future.cancel()
            return
        engine = self._engine
        grouped: dict[tuple[Any, ...] | None, list[tuple[tuple[Any, ...], dict[str, Any], Future[Any]]]] = (
            defaultdict(list)
        )
        order: list[tuple[Any, ...] | None] = []
        for item in batch:
            args, _kwargs, _future = item
            key = _sample_coalesce_key(args[0] if args else None)
            if key not in grouped:
                order.append(key)
            grouped[key].append(item)
        for key in order:
            items = grouped[key]
            if key is None or len(items) == 1 or not hasattr(items[0][0][0], "model_copy"):
                for args, kwargs, future in items:
                    self._set_future(future, lambda a=args, k=kwargs: engine.sample(*a, **k))
                continue
            requests = [args[0] for args, _kwargs, _future in items]
            total = sum(int(request.num_samples) for request in requests)
            merged = requests[0].model_copy(update={"num_samples": total})
            first_kwargs = items[0][1]

            def _merged_call(
                merged_request: Any = merged, kwargs: dict[str, Any] = first_kwargs
            ) -> Any:
                return engine.sample(merged_request, **kwargs)

            try:
                response = _merged_call()
            except Exception as exc:
                for _args, _kwargs, future in items:
                    if not future.done():
                        future.set_exception(exc)
                continue
            samples = getattr(response, "samples", None)
            if samples is None:
                for _args, _kwargs, future in items:
                    self._set_future(future, lambda value=response: value)
                continue
            offset = 0
            try:
                for request, _kwargs, future in (
                    (args[0], kwargs, future) for args, kwargs, future in items
                ):
                    width = int(request.num_samples)
                    chunk = samples[offset : offset + width]
                    if len(chunk) != width:
                        raise RuntimeError("coalesced sample count did not match waiters")
                    future.set_result(response.model_copy(update={"samples": list(chunk)}))
                    offset += width
            except Exception as exc:
                for _args, _kwargs, future in items:
                    if not future.done():
                        future.set_exception(exc)

    @staticmethod
    def _set_future(future: Future[Any], fn: Callable[[], Any]) -> None:
        if future.done():
            return
        try:
            future.set_result(fn())
        except Exception as exc:
            future.set_exception(exc)

    def __getattr__(self, name: str) -> Any:
        if name == "sample":
            def call(*args: Any, **kwargs: Any) -> Any:
                return self._coalesced_sample(*args, **kwargs)

            call.__name__ = name
            return call
        attribute = getattr(self._get_engine(), name)
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
        self._closed = True
        with self._coalesce_lock:
            pending = self._pending_samples
            self._pending_samples = []
        for _args, _kwargs, future in pending:
            future.cancel()
        self._executor.shutdown(wait=True, cancel_futures=True)

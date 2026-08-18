"""Frozen policy snapshots (decision D2).

The seed prototype had no snapshot concept at all: its architecture note says
"after ``optim_step()`` the next ``/sample`` automatically sees the new policy
version". That is fine for a single-threaded SFT loop and wrong for on-policy RL,
where an episode's ratio denominator must refer to one fixed set of weights.

The resolution is *not* a second resident model. There is one resident base
model, and a pool of frozen LoRA adapter copies. At rank 8 an adapter is a few
megabytes, so a snapshot costs approximately nothing next to another 0.8B model.

Three rules this module enforces:

1. ``optim_step`` bumps the *training* version. It never mutates a published
   snapshot, so it cannot change what an in-flight sample sees.
2. A snapshot has an immutable id. ``/v1/sample`` resolves it once, at request
   start, and holds that reference for the whole completion.
3. Sampling against an evicted snapshot fails loudly. It never falls through to
   current weights -- that failure mode produces a plausible number computed
   against the wrong policy, which is worse than an error.

Nothing here imports MLX. The adapter arrays are an opaque ``payload``; the
engine supplies a deep copy and the pool only ever holds it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


class SnapshotError(RuntimeError):
    """Base class for snapshot refusals."""


class SnapshotNotFoundError(SnapshotError):
    """The id was never issued by this service."""


class SnapshotEvictedError(SnapshotError):
    """The id was issued, and its weights are gone. Never a fallback."""


@dataclass(frozen=True, slots=True)
class PolicySnapshot:
    """An immutable, frozen copy of the training adapter."""

    id: str
    training_version: int
    step: int
    created_at: float
    base_model: str
    lora_rank: int
    lora_scale: float
    tokenizer_digest: str
    template_digest: str
    payload: Any = field(repr=False, default=None)
    metadata: dict[str, Any] = field(default_factory=dict)

    def describe(self) -> dict[str, Any]:
        """The snapshot without its weights, safe to put on the wire."""

        return {
            "policy_snapshot_id": self.id,
            "training_version": self.training_version,
            "step": self.step,
            "created_at": self.created_at,
            "base_model": self.base_model,
            "lora_rank": self.lora_rank,
            "lora_scale": self.lora_scale,
            "tokenizer_digest": self.tokenizer_digest,
            "template_digest": self.template_digest,
            "metadata": dict(self.metadata),
        }


class SnapshotPool:
    """A bounded, ordered pool of frozen snapshots.

    Evicted ids are remembered as tombstones so that "you asked for weights that
    have been dropped" is distinguishable from "you made that id up". The two
    are different bugs and deserve different errors.
    """

    def __init__(self, *, capacity: int = 4, id_factory: Callable[[], str] | None = None):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._lock = threading.RLock()
        self._snapshots: "OrderedDict[str, PolicySnapshot]" = OrderedDict()
        self._evicted: set[str] = set()
        self._id_factory = id_factory or (lambda: f"snap_{uuid.uuid4().hex}")

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._snapshots)

    def publish(
        self,
        *,
        payload: Any,
        training_version: int,
        step: int,
        base_model: str,
        lora_rank: int,
        lora_scale: float,
        tokenizer_digest: str,
        template_digest: str,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot:
        """Freeze ``payload`` under a fresh immutable id."""

        with self._lock:
            new_id = snapshot_id or self._id_factory()
            if new_id in self._snapshots or new_id in self._evicted:
                raise SnapshotError(f"policy_snapshot_id {new_id!r} is already in use")
            snapshot = PolicySnapshot(
                id=new_id,
                training_version=training_version,
                step=step,
                created_at=time.time(),
                base_model=base_model,
                lora_rank=lora_rank,
                lora_scale=lora_scale,
                tokenizer_digest=tokenizer_digest,
                template_digest=template_digest,
                payload=payload,
                metadata=dict(metadata or {}),
            )
            self._snapshots[new_id] = snapshot
            self._prune()
            return snapshot

    def get(self, snapshot_id: str) -> PolicySnapshot:
        """Resolve an id, or refuse. There is no fallback path out of here."""

        with self._lock:
            snapshot = self._snapshots.get(snapshot_id)
            if snapshot is not None:
                return snapshot
            if snapshot_id in self._evicted:
                raise SnapshotEvictedError(
                    f"policy snapshot {snapshot_id!r} has been evicted from the "
                    f"resident pool (capacity {self._capacity}). Sampling will "
                    "not fall back to current weights: republish the snapshot or "
                    "raise max_snapshots."
                )
            raise SnapshotNotFoundError(
                f"unknown policy_snapshot_id {snapshot_id!r}"
            )

    def latest(self) -> PolicySnapshot | None:
        with self._lock:
            if not self._snapshots:
                return None
            return next(reversed(self._snapshots.values()))

    def resolve(self, snapshot_id: str | None) -> PolicySnapshot:
        """Resolve the pin for one request, once, at request start.

        ``None`` resolves to the newest published snapshot. It does not mean
        "read the live adapter as generation proceeds": with no snapshot
        published yet there is nothing frozen to sample from, and the caller is
        told to publish one rather than being handed mutable weights.
        """

        with self._lock:
            if snapshot_id is not None:
                return self.get(snapshot_id)
            latest = self.latest()
            if latest is None:
                raise SnapshotNotFoundError(
                    "no policy snapshot has been published yet; call "
                    "/v1/synth/snapshots (or save_weights_and_get_sampling_client) "
                    "before sampling"
                )
            return latest

    def list(self) -> list[PolicySnapshot]:
        with self._lock:
            return list(self._snapshots.values())

    def evict(self, snapshot_id: str) -> None:
        with self._lock:
            if self._snapshots.pop(snapshot_id, None) is None:
                raise SnapshotNotFoundError(
                    f"unknown policy_snapshot_id {snapshot_id!r}"
                )
            self._evicted.add(snapshot_id)

    def _prune(self) -> None:
        while len(self._snapshots) > self._capacity:
            evicted_id, _ = self._snapshots.popitem(last=False)
            self._evicted.add(evicted_id)

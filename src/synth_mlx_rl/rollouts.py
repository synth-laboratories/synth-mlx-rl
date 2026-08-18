"""Server-side rollout records: the training authority.

A container must never be trusted to relay a training record it does not own,
and ``/v1/chat/completions`` does not return token ids or log-probabilities
anyway. So the container carries exactly one joinable fact out of a rollout --
the proxy request id -- and this service holds everything else.

What a record must contain to be usable for on-policy training:

``token_ids``          prompt + completion, in order
``rollout_logprobs``   the sampler's raw distribution values, one per completion
                       token, read BEFORE top-p / top-k / min-p truncation
``finish_reason``      stop or length
``sampling_params``    verbatim, so a rerun is reproducible
``tokenizer_digest`` / ``template_digest`` / ``render_digest``
                       so a cross-family divergence is detectable
``policy_snapshot_id`` the frozen adapter this completion was generated against
``timing``             request start and duration

A missing record makes a trace invalid for on-policy training. It is never
zero-filled and never silently dropped: containers' ``TokenCaptureProvenance``
already has a name for this state (``unavailable``), and that is the honest
answer.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Literal

from pydantic import Field

from .schemas import SamplingParamsModel, StrictModel


class RolloutRecordNotFoundError(KeyError):
    """No record under that proxy request id, or it aged out of the store."""


def new_proxy_request_id() -> str:
    return f"prid_{uuid.uuid4().hex}"


class RolloutRecord(StrictModel):
    """One policy call, as the trainer needs to see it."""

    proxy_request_id: str
    policy_snapshot_id: str
    training_version: int
    api_family: Literal["chat_completions", "responses", "native"]
    model: str

    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    #: Raw next-token log-probabilities of the sampled tokens, one per
    #: completion token, BEFORE top-p / top-k / min-p truncation.
    rollout_logprobs: list[float]
    finish_reason: Literal["stop", "length"]

    sampling_params: SamplingParamsModel
    tokenizer_digest: str
    template_digest: str
    render_digest: str
    enable_thinking: bool

    created_at: float
    duration_ms: float
    #: Set when the caller supplied an Idempotency-Key on the responses family.
    idempotency_key: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def token_ids(self) -> list[int]:
        """Prompt and completion in one sequence, ready to pre-shift."""

        return [*self.prompt_token_ids, *self.completion_token_ids]

    def alignment_ok(self) -> bool:
        return len(self.rollout_logprobs) == len(self.completion_token_ids)


class RolloutRecordResponse(StrictModel):
    record: RolloutRecord


class RolloutRecordQuery(StrictModel):
    proxy_request_ids: list[str] = Field(min_length=1, max_length=1024)


class RolloutRecordQueryResponse(StrictModel):
    records: list[RolloutRecord]
    missing: list[str]


class RolloutStore:
    """A bounded, insertion-ordered store of rollout records.

    Bounded because this is a local service on a laptop, and unbounded token
    retention is how a long run runs the host out of memory. Eviction is FIFO
    and a dropped id reports as missing rather than as an empty record.
    """

    def __init__(self, *, capacity: int = 4096):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._lock = threading.RLock()
        self._records: "OrderedDict[str, RolloutRecord]" = OrderedDict()
        self._by_idempotency_key: dict[str, str] = {}

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def put(self, record: RolloutRecord) -> RolloutRecord:
        if not record.alignment_ok():
            raise ValueError(
                "rollout_logprobs must align one-to-one with "
                "completion_token_ids; a misaligned record is not a training "
                "record"
            )
        with self._lock:
            self._records[record.proxy_request_id] = record
            if record.idempotency_key is not None:
                self._by_idempotency_key[record.idempotency_key] = (
                    record.proxy_request_id
                )
            self._prune()
            return record

    def get(self, proxy_request_id: str) -> RolloutRecord:
        with self._lock:
            record = self._records.get(proxy_request_id)
            if record is None:
                raise RolloutRecordNotFoundError(
                    f"no rollout record for proxy_request_id "
                    f"{proxy_request_id!r}; it was never written or it aged out "
                    f"of the store (capacity {self._capacity}). Treat the trace "
                    "as unavailable for on-policy training rather than "
                    "substituting zeros."
                )
            return record

    def get_many(
        self, proxy_request_ids: list[str]
    ) -> tuple[list[RolloutRecord], list[str]]:
        found: list[RolloutRecord] = []
        missing: list[str] = []
        with self._lock:
            for proxy_request_id in proxy_request_ids:
                record = self._records.get(proxy_request_id)
                if record is None:
                    missing.append(proxy_request_id)
                else:
                    found.append(record)
        return found, missing

    def find_by_idempotency_key(self, key: str) -> RolloutRecord | None:
        with self._lock:
            proxy_request_id = self._by_idempotency_key.get(key)
            if proxy_request_id is None:
                return None
            return self._records.get(proxy_request_id)

    def _prune(self) -> None:
        while len(self._records) > self._capacity:
            evicted_id, evicted = self._records.popitem(last=False)
            if evicted.idempotency_key is not None:
                self._by_idempotency_key.pop(evicted.idempotency_key, None)


def now_ms() -> float:
    return time.perf_counter() * 1000.0

"""Idempotency for the Responses family.

Banking77's responses path demands ``responses_idempotency_key``, and the reason
is that a retried paid call must not become two calls. Locally there is no
money at stake, but there is something worse: a retry that samples again
produces a *second* rollout record and a second set of tokens, and the optimizer
has no way to tell which one the container actually acted on.

So a replay returns the first response, with the same ``proxy_request_id``. A
key reused with a different request body is refused -- the alternative is
returning someone else's completion under their key.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


def request_digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    ).hexdigest()


class IdempotencyConflict(ValueError):
    """The key is known, and it was first used for a different request."""


@dataclass(frozen=True, slots=True)
class IdempotentEntry:
    request_digest: str
    status_code: int
    body: dict[str, Any]
    headers: dict[str, str]


class IdempotencyCache:
    def __init__(self, *, capacity: int = 1024):
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._capacity = capacity
        self._lock = threading.RLock()
        self._entries: "OrderedDict[str, IdempotentEntry]" = OrderedDict()

    def lookup(self, key: str, digest: str) -> IdempotentEntry | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.request_digest != digest:
                raise IdempotencyConflict(
                    f"Idempotency-Key {key!r} was first used for a different "
                    "request body; reusing it here would return a completion "
                    "that was generated for another request"
                )
            self._entries.move_to_end(key)
            return entry

    def store(self, key: str, entry: IdempotentEntry) -> None:
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

from __future__ import annotations

from typing import Any, Protocol

from .schemas import (
    AdamParams,
    CheckpointResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    RenderChatRequest,
    RenderChatResponse,
    SampleRequest,
    SampleResponse,
    StateResponse,
)
from .snapshots import PolicySnapshot


class LearnerEngine(Protocol):
    """What the HTTP layer needs from an engine.

    :class:`synth_mlx_rl.engine.MLXEngine` is the implementation.
    """

    renderer: Any
    snapshots: Any
    rollouts: Any

    def state(self) -> StateResponse: ...

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False
    ) -> str: ...

    def render_chat(self, request: RenderChatRequest) -> RenderChatResponse: ...

    def sample(
        self, request: SampleRequest, *, idempotency_key: str | None = None
    ) -> SampleResponse: ...

    def score_logprobs(
        self, token_ids: list[int], *, policy_snapshot_id: str | None = None
    ) -> list[float | None]: ...

    def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse: ...

    def optim_step(self, params: AdamParams) -> object: ...

    def zero_grad(self) -> int: ...

    def publish_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot: ...

    def resolve_snapshot(self, snapshot_id: str | None) -> PolicySnapshot: ...

    def register_policy(
        self,
        *,
        policy_dir: str | Any,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot: ...

    def load_training_adapter(self, policy_dir: str | Any) -> None: ...

    def save_checkpoint(self, name: str) -> CheckpointResponse: ...

    def load_checkpoint(self, name: str) -> CheckpointResponse: ...

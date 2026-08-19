"""Everything about sampling that is not MLX.

Snapshot pinning and rollout-record writing are the two places where a quiet
mistake produces a plausible number computed against the wrong policy. Putting
them here -- above the MLX/fake split -- means the deterministic fake engine
exercises the *same* pinning and recording code the real engine runs, which is
the only way any of it is testable on a machine that cannot install MLX.

Subclasses supply three things: how to generate tokens, how to score a sequence,
and how to copy the training adapter.
"""

from __future__ import annotations

import logging

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from .config import Settings
from .renderer import RenderedPrompt, Renderer
from .rollouts import RolloutRecord, RolloutStore, new_proxy_request_id, now_ms
from .schemas import (
    RenderChatRequest,
    RenderChatResponse,
    Sample,
    SampleRequest,
    SampleResponse,
)
from .snapshots import PolicySnapshot, SnapshotPool


logger = logging.getLogger(__name__)


class EngineBase(ABC):
    """Shared sampling, pinning, and recording behavior."""


    _accumulation_reduction: str = "mean_tokens"

    def check_reduction(self, request: Any) -> str:
        """One reduction per accumulation window, enforced for every engine.

        `mean_tokens` weights each call by its token count and divides by the
        total at optim_step; `sum` accumulates as-is and divides by nothing.
        Mixing them composes the two normalizations into a scale that is
        neither convention, so it is refused rather than silently averaged.
        """
        reduction = getattr(request, "reduction", "mean_tokens")
        pending = getattr(self, "_accumulation_count", None)
        if pending is None:
            pending = getattr(self, "accumulation_count", 0)
        if pending and reduction != self._accumulation_reduction:
            raise ValueError(
                f"cannot mix reductions inside one accumulation window: "
                f"{self._accumulation_reduction!r} then {reduction!r}; "
                "call optim_step or zero_grad first"
            )
        self._accumulation_reduction = reduction
        return reduction

    def __init__(self, settings: Settings, renderer: Renderer):
        self.settings = settings.validated()
        self.renderer = renderer
        self._lock = threading.RLock()
        self.snapshots = SnapshotPool(capacity=settings.max_snapshots)
        self.rollouts = RolloutStore(capacity=settings.max_rollout_records)
        self._step = 0
        self._training_version = 0

    # -- subclass hooks --------------------------------------------------

    @abstractmethod
    def _capture_adapter(self) -> Any:
        """Return an immutable copy of the current training adapter."""

    @abstractmethod
    def _activate(self, snapshot: PolicySnapshot | None) -> None:
        """Make ``snapshot`` the weights used for generation.

        ``None`` restores the live training adapter. Called under the engine
        lock, never mid-completion.
        """

    @abstractmethod
    def _generate(
        self, prompt_token_ids: list[int], request: SampleRequest, sample_index: int
    ) -> tuple[list[int], list[float], str]:
        """Return ``(completion_token_ids, rollout_logprobs, finish_reason)``.

        ``rollout_logprobs`` must come from the raw next-token distribution,
        before top-p / top-k / min-p truncation.
        """

    @abstractmethod
    def _decode(self, token_ids: list[int]) -> str: ...

    # -- snapshots -------------------------------------------------------

    def publish_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PolicySnapshot:
        """Freeze the current training adapter under a new immutable id.

        This is the only way a new set of weights becomes samplable.
        ``optim_step`` deliberately does not call it: a step bumps the training
        version, and an in-flight sample must not notice.
        """

        with self._lock:
            self._activate(None)
            return self.snapshots.publish(
                payload=self._capture_adapter(),
                training_version=self._training_version,
                step=self._step,
                base_model=self.settings.model,
                lora_rank=self.settings.lora_rank,
                lora_scale=self.settings.lora_scale,
                tokenizer_digest=self.renderer.tokenizer_digest,
                template_digest=self.renderer.template_digest,
                snapshot_id=snapshot_id,
                metadata=metadata,
            )

    def resolve_snapshot(self, snapshot_id: str | None) -> PolicySnapshot:
        return self.snapshots.resolve(snapshot_id)

    # -- rendering -------------------------------------------------------

    def render_chat(self, request: RenderChatRequest) -> RenderChatResponse:
        rendered = self.renderer.render(
            request.messages,
            tools=request.tools,
            add_generation_prompt=request.add_generation_prompt,
            enable_thinking=request.enable_thinking,
            tokenize=request.tokenize,
        )
        return RenderChatResponse(
            token_ids=rendered.token_ids if request.tokenize else None,
            text=rendered.text if not request.tokenize else None,
            tokenizer_digest=rendered.tokenizer_digest,
            template_digest=rendered.template_digest,
            render_digest=rendered.render_digest,
            enable_thinking=rendered.enable_thinking,
        )

    def _render_prompt(self, request: SampleRequest) -> RenderedPrompt:
        if request.prompt_token_ids is not None:
            return self.renderer.render_tokens(request.prompt_token_ids)
        if request.prompt is not None:
            return self.renderer.render_text(request.prompt)
        assert request.messages is not None
        return self.renderer.render(
            request.messages,
            tools=request.tools,
            add_generation_prompt=request.add_generation_prompt,
            enable_thinking=request.enable_thinking,
        )

    # -- sampling --------------------------------------------------------

    def sample(
        self,
        request: SampleRequest,
        *,
        idempotency_key: str | None = None,
    ) -> SampleResponse:
        """Sample, pinned to one snapshot for the whole completion.

        The snapshot is resolved exactly once, here, before the first token. An
        ``optim_step`` that lands during generation changes the training version
        and nothing else; this request finishes against the weights it started
        with.
        """

        with self._lock:
            snapshot = self.resolve_snapshot(request.policy_snapshot_id)
            rendered = self._render_prompt(request)
            if not rendered.token_ids:
                raise ValueError("prompt tokenization produced an empty sequence")
            if len(rendered.token_ids) >= self.settings.max_seq_length:
                raise ValueError(
                    f"prompt has {len(rendered.token_ids)} tokens, but "
                    f"max_seq_length is {self.settings.max_seq_length}"
                )

            sampling_params = request.sampling_params()
            self._activate(snapshot)
            samples: list[Sample] = []
            # An engine that can decode a whole group in one batch does so:
            # every member shares this prompt, so the prefill happens once and
            # the decode is one wide step per token instead of N narrow ones.
            batch_generate = getattr(self, "generate_batch", None)
            if callable(batch_generate) and request.num_samples > 1:
                started = now_ms()
                try:
                    generated = batch_generate(rendered.token_ids, request)
                except Exception as exc:  # noqa: BLE001 - width is an optimization
                    # A batch wide enough to exhaust device memory must degrade
                    # to the sequential path rather than fail the whole request:
                    # the batch is a throughput choice, not a semantic one, and
                    # both paths produce the same tokens.
                    logger.warning(
                        "batched sampling failed (%s: %s); falling back to sequential",
                        type(exc).__name__, exc,
                    )
                    generated = None
                # One wall-clock measurement covers the batch; splitting it
                # evenly is a report of cost per sample, not a claim that each
                # was produced independently.
                per_sample_ms = (
                    (now_ms() - started) / max(len(generated), 1)
                    if generated is not None else None
                )
            else:
                generated = None
                per_sample_ms = None

            try:
                for sample_index in range(request.num_samples):
                    started = now_ms()
                    if generated is not None:
                        completion, rollout_logprobs, finish_reason = generated[sample_index]
                        duration_ms = per_sample_ms
                    else:
                        completion, rollout_logprobs, finish_reason = self._generate(
                            rendered.token_ids, request, sample_index
                        )
                        duration_ms = now_ms() - started
                    record = RolloutRecord(
                        proxy_request_id=new_proxy_request_id(),
                        policy_snapshot_id=snapshot.id,
                        training_version=snapshot.training_version,
                        api_family=request.api_family,
                        model=self.settings.model,
                        prompt_token_ids=list(rendered.token_ids),
                        completion_token_ids=list(completion),
                        rollout_logprobs=list(rollout_logprobs),
                        finish_reason=finish_reason,  # type: ignore[arg-type]
                        sampling_params=sampling_params,
                        tokenizer_digest=rendered.tokenizer_digest,
                        template_digest=rendered.template_digest,
                        render_digest=rendered.render_digest,
                        enable_thinking=rendered.enable_thinking,
                        created_at=time.time(),
                        duration_ms=duration_ms,
                        idempotency_key=(
                            idempotency_key if sample_index == 0 else None
                        ),
                    )
                    self.rollouts.put(record)
                    samples.append(
                        Sample(
                            text=self._decode(completion),
                            prompt_token_ids=record.prompt_token_ids,
                            completion_token_ids=record.completion_token_ids,
                            rollout_logprobs=record.rollout_logprobs,
                            finish_reason=record.finish_reason,
                            policy_snapshot_id=snapshot.id,
                            training_version=snapshot.training_version,
                            proxy_request_id=record.proxy_request_id,
                        )
                    )
            finally:
                self._activate(None)

            return SampleResponse(samples=samples)

"""A deterministic, CPU-only engine.

Derived from the MIT-licensed `mlx-local-rl` prototype's fake engine (see
NOTICE) and extended to carry the snapshot pool and the rollout record store.

It does not pretend to be a language model. Its job is to make the HTTP
protocol, snapshot pinning, rollout records, and the shared renderer testable on
a host with no MLX -- which is every host this package is developed on
(finalized plan, correction C11).

The fake tokenizer's "chat template" is a real string, rendered by real string
formatting, so the renderer's digests are exercised the same way a Jinja chat
template would exercise them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from ..config import Settings
from ..engine_base import EngineBase
from ..kernel import policy_terms, sft_terms
from ..backends import NumpyOps
from ..renderer import Renderer
from ..schemas import (
    AdamParams,
    CheckpointResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepResponse,
    SampleRequest,
    StateResponse,
)
from ..snapshots import PolicySnapshot

FAKE_CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{m.role}}|>{{m.content}}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


class FakeTokenizer:
    """A reversible byte-ish tokenizer with a deterministic chat template."""

    vocab_size = 1024
    eos_token_id = 0
    pad_token_id = 0
    bos_token_id = 1
    chat_template = FAKE_CHAT_TEMPLATE

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [ord(character) + 10 for character in text]

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        values = list(token_ids)
        if skip_special_tokens:
            values = [value for value in values if value not in {0, 1}]
        return "".join(chr(value - 10) for value in values if value >= 10)

    def apply_chat_template(
        self,
        conversation: Any,
        *,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        tools: Any = None,
    ) -> Any:
        parts: list[str] = []
        if tools:
            parts.append("<|tools|>" + json.dumps(tools, sort_keys=True) + "<|end|>")
        if enable_thinking:
            parts.append("<|thinking|>")
        for message in conversation:
            role = message.get("role", "user")
            content = message.get("content") or ""
            if message.get("tool_calls"):
                content += "<|tool_calls|>" + json.dumps(
                    message["tool_calls"], sort_keys=True
                )
            if message.get("tool_call_id"):
                content += "<|for|>" + str(message["tool_call_id"])
            parts.append(f"<|{role}|>{content}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        text = "".join(parts)
        if not tokenize:
            return text
        return self.encode(text)


class FakeEngine(EngineBase):
    """The engine the portable test suite runs against."""

    def __init__(
        self,
        checkpoint_dir: Path | None = None,
        settings: Settings | None = None,
    ):
        resolved = (settings or Settings()).with_overrides(
            model="fake/Qwen3.5-0.8B",
            checkpoint_dir=checkpoint_dir or Path("./fake-checkpoints"),
        )
        tokenizer = FakeTokenizer()
        super().__init__(
            resolved,
            Renderer(
                tokenizer,
                model=resolved.model,
                default_enable_thinking=resolved.enable_thinking,
            ),
        )
        self.tokenizer = tokenizer
        self.accumulation_count = 0
        self.accumulation_weight = 0.0
        self.optimizer_initialized = False
        #: A stand-in for the LoRA arrays. Real enough to prove a snapshot is a
        #: frozen copy and not an alias of the live adapter.
        self._adapter: dict[str, float] = {"lora_a": 0.0, "lora_b": 0.0}
        self._resident: str | None = None
        self.publish_snapshot(metadata={"reason": "initial"})

    # -- EngineBase hooks ------------------------------------------------

    def _capture_adapter(self) -> Any:
        return dict(self._adapter)

    def _activate(self, snapshot: PolicySnapshot | None) -> None:
        self._resident = None if snapshot is None else snapshot.id

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _generate(
        self, prompt_token_ids: list[int], request: SampleRequest, sample_index: int
    ) -> tuple[list[int], list[float], str]:
        text = f"fake-{sample_index}"
        completion = self.tokenizer.encode(text, add_special_tokens=False) + [0]
        completion = completion[: request.max_tokens]
        finish_reason = "stop" if completion and completion[-1] == 0 else "length"
        return completion, [-0.25] * len(completion), finish_reason

    # -- learner surface -------------------------------------------------

    def state(self) -> StateResponse:
        latest = self.snapshots.latest()
        return StateResponse(
            ready=True,
            model=self.settings.model,
            device="cpu-fake",
            lora_rank=self.settings.lora_rank,
            lora_alpha=self.settings.lora_alpha,
            lora_scale=self.settings.lora_scale,
            lora_dropout=self.settings.lora_dropout,
            num_layers=self.settings.num_layers,
            trainable_parameters=1024,
            total_parameters=800_000_000,
            step=self._step,
            training_version=self._training_version,
            accumulation_count=self.accumulation_count,
            optimizer_initialized=self.optimizer_initialized,
            max_seq_length=self.settings.max_seq_length,
            enable_thinking=self.settings.enable_thinking,
            tokenizer_digest=self.renderer.tokenizer_digest,
            template_digest=self.renderer.template_digest,
            latest_policy_snapshot_id=None if latest is None else latest.id,
            resident_snapshots=len(self.snapshots),
        )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self.tokenizer.decode(
            token_ids, skip_special_tokens=skip_special_tokens
        )

    def score_logprobs(
        self, token_ids: list[int], *, policy_snapshot_id: str | None = None
    ) -> list[float | None]:
        with self._lock:
            snapshot = (
                None
                if policy_snapshot_id is None
                else self.resolve_snapshot(policy_snapshot_id)
            )
            self._activate(snapshot)
            try:
                return [None] + [-0.25] * (len(token_ids) - 1)
            finally:
                self._activate(None)

    def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse:
        """Run the real objective kernel over fake log-probabilities.

        The loss values are meaningless -- the "model" is a constant -- but the
        objective plumbing, the token-weighted accumulation, and the metric
        names are the production ones.
        """
        reduction = self.check_reduction(request)

        import numpy as np

        with self._lock:
            self._activate(None)
            spec = request.objective()
            ops = NumpyOps()
            token_weight = 0.0
            total_loss = 0.0
            metrics: dict[str, float] = {}
            for datum in request.data:
                weights = np.asarray(datum.weights, dtype=np.float64)
                current = np.full(weights.shape, -0.25, dtype=np.float64)
                if spec.name == "cross_entropy":
                    loss, terms = sft_terms(ops, current, weights, reduction)
                else:
                    behavior = np.asarray(
                        datum.behavior_logprobs, dtype=np.float64
                    )
                    advantages = (
                        np.full(weights.shape, float(datum.advantages))
                        if not isinstance(datum.advantages, list)
                        else np.asarray(datum.advantages, dtype=np.float64)
                    )
                    reference = (
                        None
                        if datum.reference_logprobs is None
                        else np.asarray(datum.reference_logprobs, dtype=np.float64)
                    )
                    loss, terms = policy_terms(
                        ops,
                        spec,
                        current_logprobs=current,
                        behavior_logprobs=behavior,
                        advantages=advantages,
                        weights=weights,
                        reference_logprobs=reference,
                        reference_mask=(
                            None if reference is None else np.ones_like(weights)
                        ),
                    )
                batch_weight = float(terms["token_count"])
                token_weight += batch_weight
                total_loss += float(loss) * batch_weight
                metrics = {key: float(value) for key, value in terms.items()}

            self.accumulation_count += 1
            self.accumulation_weight += token_weight
            metrics["accumulation_token_weight"] = self.accumulation_weight
            return ForwardBackwardResponse(
                loss=total_loss / token_weight,
                metrics=metrics,
                accumulation_count=self.accumulation_count,
                training_version=self._training_version,
            )

    def optim_step(self, params: AdamParams) -> OptimStepResponse:
        with self._lock:
            if self.accumulation_count == 0:
                raise ValueError(
                    "no accumulated gradients; call forward_backward first"
                )
            count = self.accumulation_count
            self.accumulation_count = 0
            self.accumulation_weight = 0.0
            self.optimizer_initialized = True
            self._step += 1
            # A step advances the *training* adapter and the training version.
            # It publishes nothing: no in-flight sample changes policy here.
            self._training_version += 1
            self._adapter = {
                key: value + params.learning_rate
                for key, value in self._adapter.items()
            }
            return OptimStepResponse(
                step=self._step,
                training_version=self._training_version,
                grad_norm=0.5,
                applied_accumulations=count,
                learning_rate=params.learning_rate,
            )

    def zero_grad(self) -> int:
        with self._lock:
            count = self.accumulation_count
            self.accumulation_count = 0
            self.accumulation_weight = 0.0
            return count

    def save_checkpoint(self, name: str) -> CheckpointResponse:
        path = self.settings.checkpoint_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return CheckpointResponse(
            path=str(path), step=self._step, training_version=self._training_version
        )

    def load_checkpoint(self, name: str) -> CheckpointResponse:
        path = self.settings.checkpoint_dir / name
        if not path.exists():
            raise FileNotFoundError(str(path))
        return CheckpointResponse(
            path=str(path), step=self._step, training_version=self._training_version
        )

"""The resident MLX engine.

Derived from the MIT-licensed `mlx-local-rl` prototype (see NOTICE). What the
prototype already had right is kept verbatim in behavior and called out below;
what it lacked -- snapshots, the CISPO family, one shared renderer, rollout
records -- comes from :mod:`engine_base`, :mod:`kernel`, and :mod:`renderer`, so
this file stays the MLX-specific part and nothing else.

Kept from the prototype, deliberately unchanged:

* **Global unmasked-token reduction across gradient accumulation.**
  ``forward_backward`` multiplies each call's gradients by that call's token
  count, and ``optim_step`` divides the accumulated tree by the summed weight.
  The result is one mean over every unmasked token in the whole accumulation
  window, not a mean of per-microbatch means -- which is a different objective
  whenever microbatches have different lengths.
* **The indexed target-logprob path.** ``nn.losses.cross_entropy(reduction=
  "none")`` yields the target score and log-normalizer without materializing a
  vocabulary-sized log-softmax. The forward still materializes ``[B,T,V]``
  logits; the saving is real but not unlimited.
* **Entropy gated behind ``entropy_coef > 0``**, precisely because a positive
  coefficient forces full-vocab probabilities.
* Qwen3.5 stays in training mode for every backward pass (its gated-delta
  implementation branches on the module training flag); only LoRA dropout is
  disabled for policy objectives, so ratios stay deterministic.

Nothing in this file has ever executed: MLX cannot be installed on the
development host (finalized plan, correction C11). ``scripts/real_mlx_smoke.sh``
is milestone 0, not a regression check.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .backends import MlxOps
from .config import Settings
from .engine_base import EngineBase
from .kernel import policy_terms, sft_terms
from .objective_spec import ObjectiveSpec
from .renderer import Renderer
from .schemas import (
    AdamParams,
    CheckpointResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    OptimStepResponse,
    SampleRequest,
    StateResponse,
)
from .snapshots import (
    PolicySnapshot,
    SnapshotError,
    SnapshotEvictedError,
    SnapshotNotFoundError,
)
from .storage import sha256_path


class MLXUnavailableError(RuntimeError):
    pass


def _grad_checkpoint(layer: Any, mx: Any) -> None:
    """Apply the same class-level checkpoint wrapper MLX-LM's trainer uses."""

    original_call = type(layer).__call__
    if getattr(original_call, "_synth_mlx_rl_checkpointed", False):
        return

    def checkpointed_call(model: Any, *args: Any, **kwargs: Any) -> Any:
        def inner_fn(params: Any, *inner_args: Any, **inner_kwargs: Any) -> Any:
            model.update(params)
            return original_call(model, *inner_args, **inner_kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    checkpointed_call._synth_mlx_rl_checkpointed = True  # type: ignore[attr-defined]
    type(layer).__call__ = checkpointed_call


def _float32_logits_processor(mx: Any) -> Any:
    """Build a logits processor that upcasts to float32 before the log-softmax.

    A factory rather than a module-level function because this module imports
    MLX lazily -- the portable test suite must keep importing it on a machine
    with no MLX at all.

    mlx-lm computes `logprobs = logits - logsumexp(logits)` in the *model*
    dtype. For a bfloat16 model that is not merely coarse, it is destructive:
    bf16 has an 8-bit mantissa, so its spacing near p=1.0 is about 2^-8. Any
    token the model is confident about -- p greater than roughly 0.996 --
    rounds to exactly 1.0 and reports a log-probability of exactly 0.0.

    Measured on Qwen3.5-0.8B, every sampled token of a confident completion came
    back as 0.0. That does not look like a bug downstream; it looks like a
    perfectly deterministic policy, and it silently zeroes the numerator of
    every importance ratio and makes the sampler/trainer mismatch check compare
    real numbers against zeros.

    float32 has a 24-bit mantissa and resolves those probabilities correctly.
    The cast costs one vocabulary-sized array per generated token.
    """

    def processor(_tokens: Any, logits: Any) -> Any:
        return logits.astype(mx.float32)

    return processor


class MLXEngine(EngineBase):
    """One resident model, one lock, a training adapter, and frozen snapshots.

    Every public operation holds one re-entrant lock. Generation mutates KV
    caches, backward passes build lazy graphs, and optimizer updates mutate the
    adapter; a single process with a single Uvicorn worker is the honest
    topology for that, and pretending otherwise would make the snapshot pin
    meaningless.
    """

    supports_cispo = True

    def __init__(self, settings: Settings):
        try:
            import mlx.core as mx
            import mlx.nn as nn
            import mlx.optimizers as optim
            from mlx.utils import tree_flatten, tree_map, tree_unflatten
            from mlx_lm import load
            from mlx_lm.generate import generate_step
            from mlx_lm.sample_utils import make_sampler
            from mlx_lm.tuner.utils import linear_to_lora_layers
        except Exception as exc:  # pragma: no cover - only on MLX hosts
            raise MLXUnavailableError(
                "MLX runtime imports failed. Run this service on Apple Silicon "
                "and install the MLX extras with: pip install -e '.[mlx]'"
            ) from exc

        self.mx = mx
        self.nn = nn
        self.optim = optim
        self.tree_flatten = tree_flatten
        self.tree_map = tree_map
        self.tree_unflatten = tree_unflatten
        self.generate_step = generate_step
        self.make_sampler = make_sampler
        self.ops = MlxOps(mx)

        settings = settings.validated()
        if hasattr(mx, "random"):
            mx.random.seed(settings.seed)

        # Never hand a Hub id to mlx-lm here: doing so can turn engine startup
        # into an unannounced network download. Workshop supplies a managed
        # snapshot, with an already-populated HF cache as the only fallback.
        self.model, self.tokenizer = load(str(settings.require_local_model_path()))
        self.model.freeze()

        if not hasattr(self.model, "layers") or not self.model.layers:
            raise ValueError("loaded model does not expose transformer layers")
        if settings.num_layers > len(self.model.layers):
            raise ValueError(
                f"num_layers={settings.num_layers} exceeds model depth "
                f"{len(self.model.layers)}"
            )

        lora_config: dict[str, Any] = {
            "rank": settings.lora_rank,
            "scale": settings.lora_scale,
            "dropout": settings.lora_dropout,
        }
        if settings.lora_keys is not None:
            lora_config["keys"] = set(settings.lora_keys)
        linear_to_lora_layers(
            self.model, settings.num_layers, lora_config, use_dora=False
        )
        self._lora_dropouts = [
            module.dropout
            for _, module in self.model.named_modules()
            if hasattr(module, "lora_a")
            and hasattr(module, "lora_b")
            and hasattr(module, "dropout")
        ]

        if settings.adapter_path is not None:
            self.model.load_weights(
                str(self._adapter_file(settings.adapter_path)), strict=False
            )

        if settings.grad_checkpoint:
            _grad_checkpoint(self.model.layers[0], mx)

        self.model.train()
        mx.eval(self.model.parameters())

        flat_total = tree_flatten(self.model.parameters())
        flat_trainable = tree_flatten(self.model.trainable_parameters())
        self._total_parameters = sum(int(value.size) for _, value in flat_total)
        self._trainable_parameters = sum(
            int(value.size) for _, value in flat_trainable
        )
        if self._trainable_parameters <= 0:
            raise RuntimeError("LoRA injection produced no trainable parameters")

        eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
        if eos_ids is None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            eos_ids = [] if eos_id is None else [eos_id]
        self._eos_token_ids = {int(token_id) for token_id in eos_ids}
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = next(iter(self._eos_token_ids), 0)
        self._pad_token_id = int(pad_id)

        super().__init__(
            settings,
            Renderer(
                self.tokenizer,
                model=settings.model,
                default_enable_thinking=settings.enable_thinking,
            ),
        )

        self._grad_accum: Any | None = None
        self._accumulation_count = 0
        self._accumulation_weight = 0.0
        self._accumulation_reduction = "mean_tokens"
        self._operation_count = 0
        self._optimizer: Any | None = None
        self._optimizer_shape: tuple[float, float, float, float, bool] | None = None
        self._last_learning_rate: float | None = None
        #: id of the frozen snapshot currently loaded into the resident model,
        #: or None when the live training adapter is loaded.
        self._resident_snapshot_id: str | None = None
        self._training_adapter: dict[str, Any] | None = None

        # There must be something frozen to sample from before the first
        # request arrives, or the first /v1/sample would have to read mutable
        # weights -- which is the failure D2 exists to prevent.
        self.publish_snapshot(metadata={"reason": "initial"})

    # -- adapters and snapshots ------------------------------------------

    @staticmethod
    def _adapter_file(path: Path) -> Path:
        candidate = path / "adapters.safetensors" if path.is_dir() else path
        if not candidate.exists():
            raise FileNotFoundError(f"adapter weights not found: {candidate}")
        return candidate

    def _capture_adapter(self) -> dict[str, Any]:
        """A frozen copy of the training adapter.

        ``mx.array(value)`` copies rather than aliasing. A rank-8 adapter is a
        few megabytes, which is what makes a pool of snapshots affordable next
        to one resident base model.
        """

        return {
            name: self.mx.array(value)
            for name, value in self.tree_flatten(self.model.trainable_parameters())
        }

    def _load_adapter(self, payload: dict[str, Any]) -> None:
        self.model.update(self.tree_unflatten(list(payload.items())))

    def _activate(self, snapshot: PolicySnapshot | None) -> None:
        """Swap the adapter the resident model is holding.

        Called under the engine lock and never mid-completion. The live training
        adapter is stashed on the way out and restored on the way back, so a
        snapshot read can never clobber training state.
        """

        target_id = None if snapshot is None else snapshot.id
        if target_id == self._resident_snapshot_id:
            return
        if self._resident_snapshot_id is None:
            self._training_adapter = self._capture_adapter()
        if snapshot is None:
            assert self._training_adapter is not None
            self._load_adapter(self._training_adapter)
            self._training_adapter = None
            self._resident_snapshot_id = None
        else:
            self._load_adapter(snapshot.payload)
            self._resident_snapshot_id = snapshot.id
        self.mx.eval(self.model.parameters())

    # -- housekeeping ----------------------------------------------------

    def _set_training_mode(self, *, lora_dropout: bool) -> None:
        self.model.train()
        for dropout in self._lora_dropouts:
            dropout.train(lora_dropout)

    def _maybe_clear_cache(self) -> None:
        self._operation_count += 1
        every = self.settings.clear_cache_every
        if every > 0 and self._operation_count % every == 0:
            self.mx.clear_cache()

    def _device_name(self) -> str:
        if hasattr(self.mx, "metal") and self.mx.metal.is_available():
            return "metal"
        try:
            return str(self.mx.default_device())
        except Exception:
            return "mlx"

    def state(self) -> StateResponse:
        with self._lock:
            latest = self.snapshots.latest()
            return StateResponse(
                ready=True,
                model=self.settings.model,
                device=self._device_name(),
                lora_rank=self.settings.lora_rank,
                lora_alpha=self.settings.lora_alpha,
                lora_scale=self.settings.lora_scale,
                lora_dropout=self.settings.lora_dropout,
                num_layers=self.settings.num_layers,
                trainable_parameters=self._trainable_parameters,
                total_parameters=self._total_parameters,
                step=self._step,
                training_version=self._training_version,
                accumulation_count=self._accumulation_count,
                optimizer_initialized=self._optimizer is not None,
                max_seq_length=self.settings.max_seq_length,
                enable_thinking=self.settings.enable_thinking,
                tokenizer_digest=self.renderer.tokenizer_digest,
                template_digest=self.renderer.template_digest,
                latest_policy_snapshot_id=None if latest is None else latest.id,
                resident_snapshots=len(self.snapshots),
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        with self._lock:
            return [
                int(token_id)
                for token_id in self.tokenizer.encode(
                    text, add_special_tokens=add_special_tokens
                )
            ]

    def decode(
        self, token_ids: list[int], *, skip_special_tokens: bool = False
    ) -> str:
        with self._lock:
            return str(
                self.tokenizer.decode(
                    token_ids, skip_special_tokens=skip_special_tokens
                )
            )

    def _decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    # -- generation ------------------------------------------------------

    def generate_batch(
        self, prompt_token_ids: list[int], request: SampleRequest
    ) -> list[tuple[list[int], list[float], str]]:
        """Decode `num_samples` completions of one prompt in a single batch.

        Group sampling is the ideal batching case: every member shares an
        identical prompt, so the prefill is done once and the decode runs as one
        [B, 1] step per token instead of B separate [1, 1] steps. On Apple
        Silicon the decode is memory-bandwidth bound -- the weights are read
        once per step regardless of B -- so widening the batch is close to free
        until B gets large.

        mlx-lm ships `batch_generate`, but it returns only text and stats. The
        rollout log-probabilities are the training authority here, so this
        decodes directly and records the log-probability of each sampled token
        from the same float32 distribution the sampler drew from.
        """

        mx = self.mx
        from mlx_lm.models.cache import make_prompt_cache

        count = int(request.num_samples)
        available = self.settings.max_seq_length - len(prompt_token_ids)
        max_tokens = min(request.max_tokens, available)
        if max_tokens <= 0:
            raise ValueError("no generation room remains under max_seq_length")

        stop_ids = set(self._eos_token_ids)
        if request.stop_token_ids:
            stop_ids.update(int(token_id) for token_id in request.stop_token_ids)
        stop_sequences = self._stop_sequences(request)

        sampler = self.make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
        )
        if request.seed is not None:
            mx.random.seed(int(request.seed))

        self.model.eval()
        try:
            cache = make_prompt_cache(self.model)
            tokens = mx.array([list(prompt_token_ids)] * count)
            logits = self.model(tokens, cache=cache)[:, -1, :]

            completions: list[list[int]] = [[] for _ in range(count)]
            logprob_rows: list[list[float]] = [[] for _ in range(count)]
            finish: list[str] = ["length"] * count
            live = [True] * count

            for _ in range(max_tokens):
                # float32 before the log-softmax. In the model dtype (bf16) any
                # token with p > ~0.996 rounds to exactly 1.0 and reports a
                # log-probability of 0.0; see `_float32_logits_processor`.
                logprobs = logits.astype(mx.float32)
                logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
                sampled = sampler(logprobs)
                chosen = mx.take_along_axis(
                    logprobs, sampled.reshape(count, 1), axis=-1
                ).reshape(count)
                mx.eval(sampled, chosen)

                sampled_ids = [int(value) for value in sampled.tolist()]
                chosen_values = [float(value) for value in chosen.tolist()]
                for row in range(count):
                    if not live[row]:
                        continue
                    token_id = sampled_ids[row]
                    completions[row].append(token_id)
                    logprob_rows[row].append(chosen_values[row])
                    matched = any(
                        len(completions[row]) >= len(sequence)
                        and completions[row][-len(sequence):] == sequence
                        for sequence in stop_sequences
                    )
                    if token_id in stop_ids or matched:
                        finish[row] = "stop"
                        live[row] = False
                if not any(live):
                    break
                # Finished rows keep stepping so the batch stays rectangular;
                # their output is discarded by the `live` check above. Dropping
                # them mid-flight would mean rebuilding the KV cache.
                logits = self.model(sampled.reshape(count, 1), cache=cache)[:, -1, :]

            return [
                (completions[row], logprob_rows[row], finish[row])
                for row in range(count)
            ]
        finally:
            self._set_training_mode(lora_dropout=True)
            self._maybe_clear_cache()

    def _stop_sequences(self, request: SampleRequest) -> list[list[int]]:
        raw_stops = (
            [request.stop] if isinstance(request.stop, str) else (request.stop or [])
        )
        return [
            sequence
            for sequence in (
                [
                    int(token_id)
                    for token_id in self.tokenizer.encode(
                        stop_text, add_special_tokens=False
                    )
                ]
                for stop_text in raw_stops
                if stop_text
            )
            if sequence
        ]

    def _generate(
        self, prompt_token_ids: list[int], request: SampleRequest, sample_index: int
    ) -> tuple[list[int], list[float], str]:
        available = self.settings.max_seq_length - len(prompt_token_ids)
        max_tokens = min(request.max_tokens, available)
        if max_tokens <= 0:
            raise ValueError("no generation room remains under max_seq_length")

        stop_ids = set(self._eos_token_ids)
        if request.stop_token_ids:
            stop_ids.update(int(token_id) for token_id in request.stop_token_ids)
        raw_stops = (
            [request.stop] if isinstance(request.stop, str) else (request.stop or [])
        )
        stop_sequences = [
            sequence
            for sequence in (
                [
                    int(token_id)
                    for token_id in self.tokenizer.encode(
                        stop_text, add_special_tokens=False
                    )
                ]
                for stop_text in raw_stops
                if stop_text
            )
            if sequence
        ]

        sampler = self.make_sampler(
            temp=request.temperature,
            top_p=request.top_p,
            min_p=request.min_p,
            top_k=request.top_k,
        )
        prompt_array = self.mx.array(prompt_token_ids)

        self.model.eval()
        try:
            # The engine is seeded once at startup. An explicit request seed
            # makes the call reproducible and gives each member of a sample
            # group an independent stream.
            if request.seed is not None:
                self.mx.random.seed(int(request.seed) + sample_index)
            completion: list[int] = []
            rollout_logprobs: list[float] = []
            finish_reason = "length"
            for token, logprobs in self.generate_step(
                prompt_array,
                self.model,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=[_float32_logits_processor(self.mx)],
            ):
                token_id = int(token)
                self.mx.eval(logprobs)
                completion.append(token_id)
                # `logprobs` here is the model's full next-token distribution,
                # log-softmaxed, BEFORE the sampler applied top-p / top-k /
                # min-p. That is the population the mismatch check needs; a
                # truncated value would understate the true sampling density.
                # It is float32 because `_float32_logits` upcast the logits --
                # see that function for why bf16 here silently destroys the
                # signal rather than merely coarsening it.
                rollout_logprobs.append(float(logprobs[token_id].item()))
                matched_text_stop = any(
                    len(completion) >= len(sequence)
                    and completion[-len(sequence) :] == sequence
                    for sequence in stop_sequences
                )
                if token_id in stop_ids or matched_text_stop:
                    finish_reason = "stop"
                    break
            return completion, rollout_logprobs, finish_reason
        finally:
            self._set_training_mode(lora_dropout=True)
            self._maybe_clear_cache()

    def score_logprobs(
        self, token_ids: list[int], *, policy_snapshot_id: str | None = None
    ) -> list[float | None]:
        """Score a sequence under a pinned snapshot, or under live weights.

        This is how ``behavior_logprobs`` are produced: a trainer forward pass
        at the snapshot the rollout was generated against, which is a different
        population from the sampler's ``rollout_logprobs``.
        """

        with self._lock:
            tokens = [int(token_id) for token_id in token_ids]
            if len(tokens) < 2:
                raise ValueError("at least two tokens are required")
            if len(tokens) > self.settings.max_seq_length:
                raise ValueError(
                    f"sequence has {len(tokens)} tokens, but max_seq_length is "
                    f"{self.settings.max_seq_length}"
                )
            snapshot = (
                None
                if policy_snapshot_id is None
                else self.resolve_snapshot(policy_snapshot_id)
            )
            self._activate(snapshot)
            self.model.eval()
            try:
                inputs = self.mx.array([tokens[:-1]])
                targets = self.mx.array([tokens[1:]])
                logits = self.model(inputs)
                selected = self._selected_logprobs(logits, targets)
                self.mx.eval(selected)
                return [None, *[float(value) for value in selected.tolist()[0]]]
            finally:
                self._set_training_mode(lora_dropout=True)
                self._activate(None)
                self._maybe_clear_cache()

    # -- training --------------------------------------------------------

    def _make_batch(self, request: ForwardBackwardRequest) -> dict[str, Any]:
        max_len = max(len(datum.input_ids) for datum in request.data)
        if max_len > self.settings.max_seq_length:
            raise ValueError(
                f"batch sequence length {max_len} exceeds max_seq_length "
                f"{self.settings.max_seq_length}"
            )

        input_rows: list[list[int]] = []
        target_rows: list[list[int]] = []
        weight_rows: list[list[float]] = []
        behavior_rows: list[list[float]] = []
        advantage_rows: list[list[float]] = []
        reference_rows: list[list[float]] = []
        reference_mask_rows: list[list[float]] = []

        for datum in request.data:
            n = len(datum.input_ids)
            pad = max_len - n
            input_rows.append(datum.input_ids + [self._pad_token_id] * pad)
            target_rows.append(datum.target_ids + [self._pad_token_id] * pad)
            weight_rows.append(datum.weights + [0.0] * pad)

            if datum.behavior_logprobs is None:
                behavior_rows.append([0.0] * max_len)
            else:
                behavior_rows.append(datum.behavior_logprobs + [0.0] * pad)

            if datum.advantages is None:
                advantages = [0.0] * n
            elif isinstance(datum.advantages, list):
                advantages = datum.advantages
            else:
                advantages = [float(datum.advantages)] * n
            advantage_rows.append(advantages + [0.0] * pad)

            if datum.reference_logprobs is None:
                reference_rows.append([0.0] * max_len)
                reference_mask_rows.append([0.0] * max_len)
            else:
                reference_rows.append(datum.reference_logprobs + [0.0] * pad)
                reference_mask_rows.append([1.0] * n + [0.0] * pad)

        mx = self.mx
        return {
            "inputs": mx.array(input_rows),
            "targets": mx.array(target_rows),
            "weights": mx.array(weight_rows, dtype=mx.float32),
            "behavior_logprobs": mx.array(behavior_rows, dtype=mx.float32),
            "advantages": mx.array(advantage_rows, dtype=mx.float32),
            "reference_logprobs": mx.array(reference_rows, dtype=mx.float32),
            "reference_mask": mx.array(reference_mask_rows, dtype=mx.float32),
        }

    def _selected_logprobs(self, logits: Any, targets: Any) -> Any:
        """Target-token log-probabilities without a full-vocab log-softmax.

        MLX's indexed cross-entropy computes the target score and the log
        normalizer directly, preserving the same gradient while keeping the
        default SFT/RL path substantially smaller than a ``[B,T,V]`` float32
        log-probability tensor would be.
        """

        return (-self.nn.losses.cross_entropy(logits, targets, reduction="none")).astype(
            self.mx.float32
        )

    def _entropy(self, logits: Any, weights: Any, token_count: Any) -> Any:
        """Mean token entropy. Only ever called when ``entropy_coef > 0``."""

        mx = self.mx
        logits_fp32 = logits.astype(mx.float32)
        full_logprobs = logits_fp32 - mx.logsumexp(logits_fp32, axis=-1, keepdims=True)
        probabilities = mx.exp(full_logprobs)
        token_entropy = -(probabilities * full_logprobs).sum(axis=-1)
        return (token_entropy * weights).sum() / token_count

    def _sft_loss(
        self, model: Any, inputs: Any, targets: Any, weights: Any, reduction: str
    ) -> tuple[Any, dict[str, Any]]:
        logits = model(inputs)
        current = self._selected_logprobs(logits, targets)
        return sft_terms(self.ops, current, weights, reduction)

    def _policy_loss(
        self,
        model: Any,
        inputs: Any,
        targets: Any,
        weights: Any,
        behavior_logprobs: Any,
        advantages: Any,
        reference_logprobs: Any,
        reference_mask: Any,
        spec: ObjectiveSpec,
    ) -> tuple[Any, dict[str, Any]]:
        logits = model(inputs)
        current = self._selected_logprobs(logits, targets)
        entropy = None
        if spec.entropy_coef > 0.0:
            entropy = self._entropy(logits, weights, self.mx.sum(weights))
        return policy_terms(
            self.ops,
            spec,
            current_logprobs=current,
            behavior_logprobs=behavior_logprobs,
            advantages=advantages,
            weights=weights,
            reference_logprobs=reference_logprobs,
            reference_mask=reference_mask,
            entropy=entropy,
        )

    def _evaluate_tree(self, *items: Any) -> None:
        leaves: list[Any] = []
        for item in items:
            if isinstance(item, dict):
                leaves.extend(item.values())
            elif isinstance(item, (list, tuple)):
                leaves.extend(value for _, value in self.tree_flatten(item))
            else:
                leaves.append(item)
        self.mx.eval(*leaves)

    def forward_backward(
        self, request: ForwardBackwardRequest
    ) -> ForwardBackwardResponse:
        with self._lock:
            spec = request.objective()
            self._activate(None)
            batch = self._make_batch(request)
            # Qwen3.5's eval mode selects an inference-only gated-delta kernel,
            # so the model stays in training mode for every backward pass; only
            # LoRA dropout is disabled for policy objectives, which keeps the
            # ratio between two forward passes deterministic.
            self._set_training_mode(lora_dropout=spec.name == "cross_entropy")

            try:
                if spec.name == "cross_entropy":
                    value_and_grad = self.nn.value_and_grad(self.model, self._sft_loss)
                    (loss, terms), gradients = value_and_grad(
                        self.model,
                        batch["inputs"],
                        batch["targets"],
                        batch["weights"],
                        getattr(request, "reduction", "mean_tokens"),
                    )
                else:
                    value_and_grad = self.nn.value_and_grad(
                        self.model, self._policy_loss
                    )
                    (loss, terms), gradients = value_and_grad(
                        self.model,
                        batch["inputs"],
                        batch["targets"],
                        batch["weights"],
                        batch["behavior_logprobs"],
                        batch["advantages"],
                        batch["reference_logprobs"],
                        batch["reference_mask"],
                        spec,
                    )
                self._evaluate_tree(loss, terms, gradients)
                metrics = {key: float(value.item()) for key, value in terms.items()}
            finally:
                self._set_training_mode(lora_dropout=True)

            # --- global unmasked-token reduction, part 1 of 2 ---
            # Each accumulation contributes gradients weighted by its own token
            # count. `optim_step` divides by the summed weight. Together they
            # produce one mean over every unmasked token in the window.
            # The accumulation scaling has to match the reduction, or the two
            # normalizations compose into something that is neither convention.
            #   mean_tokens: weight each call by its tokens, divide by the total
            #                at optim_step -> one global mean over the window.
            #   sum:         accumulate as-is, divide by nothing -> a sum of
            #                sums, which is what `sum` means.
            reduction = self.check_reduction(request)
            batch_weight = (
                float(metrics["token_count"]) if reduction == "mean_tokens" else 1.0
            )
            weighted_gradients = self.tree_map(
                lambda gradient: gradient * batch_weight, gradients
            )
            if self._grad_accum is None:
                self._grad_accum = weighted_gradients
            else:
                self._grad_accum = self.tree_map(
                    lambda accumulated, new: accumulated + new,
                    self._grad_accum,
                    weighted_gradients,
                )
            self._accumulation_count += 1
            self._accumulation_weight += batch_weight
            metrics["accumulation_token_weight"] = self._accumulation_weight
            self._maybe_clear_cache()

            return ForwardBackwardResponse(
                loss=float(loss.item()),
                metrics=metrics,
                accumulation_count=self._accumulation_count,
                training_version=self._training_version,
            )

    @staticmethod
    def _optimizer_structural_key(
        params: AdamParams,
    ) -> tuple[float, float, float, float, bool]:
        return (
            params.beta1,
            params.beta2,
            params.eps,
            params.weight_decay,
            params.bias_correction,
        )

    def _ensure_optimizer(self, params: AdamParams) -> None:
        key = self._optimizer_structural_key(params)
        if self._optimizer is None:
            self._optimizer = self.optim.AdamW(
                learning_rate=params.learning_rate,
                betas=[params.beta1, params.beta2],
                eps=params.eps,
                weight_decay=params.weight_decay,
                bias_correction=params.bias_correction,
            )
            self._optimizer.init(self.model.trainable_parameters())
            self.mx.eval(self._optimizer.state)
            self._optimizer_shape = key
        elif key != self._optimizer_shape:
            raise ValueError(
                "beta1, beta2, eps, weight_decay, and bias_correction cannot be "
                "changed after optimizer initialization; save a checkpoint and "
                "restart the service to change them"
            )
        self._optimizer.learning_rate = params.learning_rate
        self._last_learning_rate = params.learning_rate

    def _gradient_norm(self, gradients: Any) -> Any:
        total = self.mx.array(0.0)
        for _, gradient in self.tree_flatten(gradients):
            g = gradient.astype(self.mx.float32)
            total = total + self.mx.sum(g * g)
        return self.mx.sqrt(total)

    def optim_step(self, params: AdamParams) -> OptimStepResponse:
        """Apply the accumulated gradient. Publishes no snapshot.

        The training version advances here. Nothing an in-flight ``/v1/sample``
        pinned changes, because a pinned snapshot is a separate frozen copy of
        the adapter (decision D2).
        """

        with self._lock:
            if self._grad_accum is None or self._accumulation_count == 0:
                raise ValueError(
                    "no accumulated gradients; call forward_backward first"
                )
            self._activate(None)
            self._ensure_optimizer(params)
            assert self._optimizer is not None

            count = self._accumulation_count
            token_weight = (
                self._accumulation_weight
                if self._accumulation_reduction == "mean_tokens"
                else 1.0
            )
            if token_weight <= 0.0:
                raise RuntimeError("accumulated gradient token weight is zero")
            # --- global unmasked-token reduction, part 2 of 2 ---
            gradients = self.tree_map(
                lambda gradient: gradient / token_weight, self._grad_accum
            )
            grad_norm = self._gradient_norm(gradients)
            if params.max_grad_norm is not None:
                scale = self.mx.minimum(
                    self.mx.array(1.0), params.max_grad_norm / (grad_norm + 1e-6)
                )
                gradients = self.tree_map(lambda gradient: gradient * scale, gradients)

            self._optimizer.update(self.model, gradients)
            self._evaluate_tree(
                self.model.parameters(), self._optimizer.state, grad_norm
            )

            self._grad_accum = None
            self._accumulation_count = 0
            self._accumulation_weight = 0.0
            self._step += 1
            self._training_version += 1
            self._maybe_clear_cache()

            return OptimStepResponse(
                step=self._step,
                training_version=self._training_version,
                grad_norm=float(grad_norm.item()),
                applied_accumulations=count,
                learning_rate=params.learning_rate,
            )

    def zero_grad(self) -> int:
        with self._lock:
            cleared = self._accumulation_count
            self._grad_accum = None
            self._accumulation_count = 0
            self._accumulation_weight = 0.0
            self._maybe_clear_cache()
            return cleared

    # -- checkpoints -----------------------------------------------------

    def _checkpoint_path(self, name: str) -> Path:
        root = self.settings.checkpoint_dir.expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        destination = (root / name).resolve()
        if root != destination and root not in destination.parents:
            raise ValueError("checkpoint name escapes checkpoint_dir")
        return destination

    def _adapter_config(self) -> dict[str, Any]:
        lora_parameters: dict[str, Any] = {
            "rank": self.settings.lora_rank,
            "dropout": self.settings.lora_dropout,
            "scale": self.settings.lora_scale,
        }
        if self.settings.lora_keys is not None:
            lora_parameters["keys"] = list(self.settings.lora_keys)
        return {
            "model": self.settings.model,
            "fine_tune_type": "lora",
            "num_layers": self.settings.num_layers,
            "lora_parameters": lora_parameters,
        }

    def save_checkpoint(self, name: str) -> CheckpointResponse:
        with self._lock:
            self._activate(None)
            path = self._checkpoint_path(name)
            path.mkdir(parents=True, exist_ok=True)

            adapter_weights = dict(
                self.tree_flatten(self.model.trainable_parameters())
            )
            self.mx.save_safetensors(
                str(path / "adapters.safetensors"), adapter_weights
            )
            (path / "adapter_config.json").write_text(
                json.dumps(self._adapter_config(), indent=2) + "\n", encoding="utf-8"
            )

            optimizer_params: dict[str, Any] | None = None
            if self._optimizer is not None:
                optimizer_weights = dict(self.tree_flatten(self._optimizer.state))
                self.mx.save_safetensors(
                    str(path / "optimizer.safetensors"), optimizer_weights
                )
                assert self._optimizer_shape is not None
                beta1, beta2, eps, weight_decay, bias_correction = (
                    self._optimizer_shape
                )
                optimizer_params = {
                    "learning_rate": self._last_learning_rate,
                    "beta1": beta1,
                    "beta2": beta2,
                    "eps": eps,
                    "weight_decay": weight_decay,
                    "bias_correction": bias_correction,
                }

            state = {
                "format_version": 2,
                "step": self._step,
                "training_version": self._training_version,
                "tokenizer_digest": self.renderer.tokenizer_digest,
                "template_digest": self.renderer.template_digest,
                "optimizer": optimizer_params,
                "settings": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in asdict(self.settings).items()
                },
            }
            (path / "state.json").write_text(
                json.dumps(state, indent=2) + "\n", encoding="utf-8"
            )
            return CheckpointResponse(
                path=str(path),
                step=self._step,
                training_version=self._training_version,
            )

    def _validate_checkpoint_adapter(self, config: dict[str, Any]) -> None:
        expected = self._adapter_config()
        checks = [
            ("model", config.get("model"), expected["model"]),
            ("num_layers", config.get("num_layers"), expected["num_layers"]),
            (
                "lora rank",
                config.get("lora_parameters", {}).get("rank"),
                expected["lora_parameters"]["rank"],
            ),
            (
                "lora scale",
                config.get("lora_parameters", {}).get("scale"),
                expected["lora_parameters"]["scale"],
            ),
        ]
        mismatches = [
            f"{label}: checkpoint={actual!r}, service={wanted!r}"
            for label, actual, wanted in checks
            if actual != wanted
        ]
        if mismatches:
            raise ValueError("checkpoint adapter mismatch: " + "; ".join(mismatches))

    def load_checkpoint(self, name: str) -> CheckpointResponse:
        with self._lock:
            path = self._checkpoint_path(name)
            if not path.is_dir():
                raise FileNotFoundError(f"checkpoint does not exist: {path}")
            adapter_file = path / "adapters.safetensors"
            config_file = path / "adapter_config.json"
            state_file = path / "state.json"
            if not (
                adapter_file.exists()
                and config_file.exists()
                and state_file.exists()
            ):
                raise FileNotFoundError(
                    "checkpoint requires adapters.safetensors, "
                    "adapter_config.json, and state.json"
                )

            self._activate(None)
            adapter_config = json.loads(config_file.read_text(encoding="utf-8"))
            self._validate_checkpoint_adapter(adapter_config)
            state = json.loads(state_file.read_text(encoding="utf-8"))
            self.model.load_weights(str(adapter_file), strict=False)

            optimizer_meta = state.get("optimizer")
            optimizer_file = path / "optimizer.safetensors"
            if optimizer_meta is not None and optimizer_file.exists():
                self._ensure_optimizer(AdamParams(**optimizer_meta))
                assert self._optimizer is not None
                flat_state = list(self.mx.load(str(optimizer_file)).items())
                self._optimizer.state = self.tree_unflatten(flat_state)
                self.mx.eval(self._optimizer.state)
            else:
                self._optimizer = None
                self._optimizer_shape = None
                self._last_learning_rate = None

            self._step = int(state.get("step", 0))
            self._training_version = int(
                state.get("training_version", state.get("policy_version", self._step))
            )
            self._grad_accum = None
            self._accumulation_count = 0
            self._accumulation_weight = 0.0
            self.mx.eval(self.model.parameters())
            self._maybe_clear_cache()
            digest, _ = sha256_path(path)
            snapshot_id = f"sha256:{digest}"
            try:
                self.snapshots.get(snapshot_id)
            except (SnapshotNotFoundError, SnapshotEvictedError):
                try:
                    self.publish_snapshot(
                        snapshot_id=snapshot_id,
                        metadata={
                            "reason": "checkpoint_load",
                            "name": name,
                            "sha256": digest,
                        },
                    )
                except SnapshotError:
                    pass
            return CheckpointResponse(
                path=str(path),
                step=self._step,
                training_version=self._training_version,
            )

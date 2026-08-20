"""The local training backend. One backend: a real Qwen LoRA fine-tune."""

from __future__ import annotations

import json
import random
import threading
import uuid
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Callable, Protocol

from synth_mlx_rl.models import Checkpoint, Job, JobStatus
from synth_mlx_rl.rewards import has_learning_signal, normalize_group_rewards
from synth_mlx_rl.storage import JobStore, sha256_file, sha256_path, utc_now


class TrainingEngine(Protocol):
    def render_chat(self, request): ...
    def forward_backward(self, request): ...
    def optim_step(self, params): ...
    def save_checkpoint(self, name: str): ...
    def score_logprobs(self, token_ids: list[int], *, policy_snapshot_id: str | None = None): ...


class _MinibatchStream:
    """Shuffled minibatches that wrap around the dataset, counting epochs.

    A batch may straddle an epoch boundary; the tail of one shuffle is followed
    by the head of the next rather than being dropped or padded, so every row
    is seen the same number of times regardless of how the batch size divides
    the dataset.
    """

    def __init__(self, *, count: int, batch_size: int, shuffle: bool, seed: int) -> None:
        if count <= 0:
            raise ValueError("cannot build minibatches from an empty dataset")
        self._count = count
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._random = random.Random(seed)
        self._order = list(range(count))
        self.epoch = 1
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        if self._shuffle:
            self._random.shuffle(self._order)

    def next_batch(self) -> list[int]:
        batch: list[int] = []
        while len(batch) < self._batch_size:
            if self._cursor >= self._count:
                self.epoch += 1
                self._cursor = 0
                self._reshuffle()
            take = min(self._batch_size - len(batch), self._count - self._cursor)
            batch.extend(self._order[self._cursor : self._cursor + take])
            self._cursor += take
        return batch


def _datum_from_action(datum_cls, action, advantage: float, *, sequence_cap: int):
    """One sampled turn as a training example, laid out as the hosted lane lays it.

    The observation is scored but carries no advantage: only the tokens the
    policy actually chose are credited or blamed for the reward, so weights and
    advantages are zero across the prompt and live over the completion. The
    behaviour log-probabilities the container returned occupy the same
    positions, which is what makes the importance ratio a ratio of the same
    tokens under two policy versions rather than of two different spans.
    """

    prompt = list(action.prompt_token_ids)
    completion = list(action.token_ids)
    behavior = list(action.log_probs)
    tokens = [*prompt, *completion]
    if len(tokens) > sequence_cap:
        raise RuntimeError("rollout_sequence_cap_exceeded")
    inputs = tokens[:-1]
    targets = tokens[1:]
    observation_length = len(prompt) - 1
    logged = [0.0] * observation_length + behavior
    advantages = [0.0] * observation_length + [advantage] * len(completion)
    weights = [0.0] * observation_length + [1.0] * len(completion)
    if not (len(inputs) == len(targets) == len(logged) == len(advantages) == len(weights)):
        raise RuntimeError("rollout_training_tensor_unaligned")
    return (
        datum_cls(
            input_ids=inputs,
            target_ids=targets,
            weights=weights,
            behavior_logprobs=logged,
            advantages=advantages,
        ),
        len(completion),
    )


def _mlx_peak_memory() -> int | None:
    """Peak MLX allocation so far, or None when MLX is not loaded.

    Recorded per step because memory is the binding constraint on this
    hardware: the same 768-token datum peaks at 7.30 GB with gradient
    checkpointing and 33.40 GB without, and above Metal's recommended working
    set the allocator thrashes and step time goes superlinear. A run that does
    not record it cannot explain why it got slow.
    """

    try:
        import mlx.core as mx
    except ImportError:
        return None
    return int(mx.get_peak_memory())


class Cancelled(Exception):
    """Raised when a user-cancelled job reaches a safe step boundary."""


class TrainingRunner:
    """Runs bounded jobs in service-owned threads.

    The service intentionally offers no claimed auto-resume: an MLX optimizer
    state is not durable in this v0.6 slice. A restarted active job becomes
    `interrupted`; completed state and terminal checkpoints remain reopenable.
    """

    def __init__(
        self,
        store: JobStore,
        engine_provider: Callable[[], TrainingEngine] | None = None,
        sampler_url: str = "http://127.0.0.1:8787/v1/chat/completions",
        sampler_token: str = "local",
    ) -> None:
        self.store = store
        self._engine_provider = engine_provider
        # Where the task container calls back to sample. Loopback in both
        # directions: the sampler and the trainer are the same resident model.
        self._sampler_url = sampler_url
        self._sampler_token = sampler_token
        self._threads: dict[str, threading.Thread] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def launch(self, job_id: str) -> Job:
        with self._lock:
            job = self.store.load_job(job_id)
            if job.status != JobStatus.CONFIGURED:
                raise ValueError(f"job {job_id} is {job.status}; only configured jobs can launch")
            cancel = threading.Event()
            self._cancel[job_id] = cancel
            job.status = JobStatus.QUEUED
            job.updated_at = utc_now()
            self.store.save_job(job)
            self.store.append_event(job_id, "job.queued")
            thread = threading.Thread(
                target=self._run,
                args=(job_id, cancel),
                daemon=True,
                name=f"mlx-job-{job_id}",
            )
            self._threads[job_id] = thread
            thread.start()
            return job

    def resume(self, job_id: str) -> Job:
        """Continue an interrupted or cancelled job from its last checkpoint.

        The engine restores the adapter weights, the Adam moments and the step
        counter together -- resuming weights without the optimizer state would
        silently restart the moment estimates and change the trajectory while
        looking like a continuation.
        """

        with self._lock:
            job = self.store.load_job(job_id)
            if job.status not in {JobStatus.INTERRUPTED, JobStatus.CANCELLED}:
                raise ValueError(
                    f"job {job_id} is {job.status}; only interrupted or cancelled jobs resume"
                )
            if not job.checkpoints:
                raise ValueError("job has no checkpoint to resume from")
            if job.config.lora_dropout > 0:
                raise ValueError(
                    "resume with lora_dropout>0 is refused; the MLX RNG stream "
                    "is not persisted and a resumed run would diverge"
                )
            if job.current_step >= job.config.max_steps:
                raise ValueError("job already reached max_steps; there is nothing to resume")
            cancel = threading.Event()
            self._cancel[job_id] = cancel
            job.status = JobStatus.QUEUED
            job.error_code = None
            job.error_detail = None
            job.finished_at = None
            job.updated_at = utc_now()
            self.store.save_job(job)
            self.store.append_event(
                job_id,
                "job.resumed",
                {"from_step": job.current_step, "checkpoint": job.checkpoints[-1].checkpoint_id},
            )
            thread = threading.Thread(
                target=self._run,
                args=(job_id, cancel, True),
                daemon=True,
                name=f"mlx-job-{job_id}",
            )
            self._threads[job_id] = thread
            thread.start()
            return job

    def cancel(self, job_id: str) -> Job:
        with self._lock:
            job = self.store.load_job(job_id)
            if job.status in {
                JobStatus.SUCCEEDED,
                JobStatus.CANCELLED,
                JobStatus.FAILED,
                JobStatus.INTERRUPTED,
            }:
                return job
            event = self._cancel.get(job_id)
            if event is None:
                raise ValueError("job is not owned by this service process and cannot be cancelled")
            event.set()
            job.status = JobStatus.CANCELLING
            job.updated_at = utc_now()
            self.store.save_job(job)
            self.store.append_event(job_id, "job.cancellation_requested")
            return job

    def _run(self, job_id: str, cancelled: threading.Event, resuming: bool = False) -> None:
        job = self.store.load_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.updated_at = job.started_at
        self.store.save_job(job)
        self.store.append_event(
            job_id, "job.started", {"backend": job.config.backend, "resumed": resuming}
        )
        try:
            if job.config.backend == "qwen_lora":
                self._run_qwen_lora(job, cancelled, resuming)
            elif job.config.backend == "cispo":
                self._run_cispo(job, cancelled, resuming)
            else:  # pydantic validates before persistence
                raise RuntimeError(f"unsupported backend {job.config.backend}")
            job = self.store.load_job(job_id)
            job.status = JobStatus.SUCCEEDED
            job.finished_at = utc_now()
            job.updated_at = job.finished_at
            self.store.save_job(job)
            self.store.append_event(
                job_id,
                "job.succeeded",
                {"terminal_checkpoint": job.checkpoints[-1].checkpoint_id},
            )
        except Cancelled:
            job = self.store.load_job(job_id)
            job.status = JobStatus.CANCELLED
            job.finished_at = utc_now()
            job.updated_at = job.finished_at
            self.store.save_job(job)
            self.store.append_event(job_id, "job.cancelled", {"step": job.current_step})
        except Exception as exc:  # Errors are structured and persisted, not only logged.
            job = self.store.load_job(job_id)
            job.status = JobStatus.FAILED
            job.error_code = type(exc).__name__.lower()
            job.error_detail = str(exc)[:1000]
            job.finished_at = utc_now()
            job.updated_at = job.finished_at
            self.store.save_job(job)
            self.store.append_event(job_id, "job.failed", {"error_code": job.error_code})

    def _run_qwen_lora(
        self, job: Job, cancelled: threading.Event, resuming: bool = False
    ) -> None:
        """Bounded real Qwen LoRA SFT through the resident MLX engine."""
        if self._engine_provider is None:
            raise RuntimeError("the resident Qwen MLX engine is not available")
        from synth_mlx_rl.schemas import (
            AdamParams,
            ChatMessage,
            Datum,
            ForwardBackwardRequest,
            RenderChatRequest,
        )

        engine = self._engine_provider()

        def render_dataset(path: Path, label: str):
            rows: list[list[ChatMessage]] = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                if isinstance(raw.get("messages"), list):
                    messages = [ChatMessage.model_validate(item) for item in raw["messages"]]
                elif isinstance(raw.get("prompt"), str) and isinstance(raw.get("completion"), str):
                    messages = [
                        ChatMessage(role="user", content=raw["prompt"]),
                        ChatMessage(role="assistant", content=raw["completion"]),
                    ]
                else:
                    raise ValueError(
                        f"{label} line {line_number} needs messages or prompt/completion"
                    )
                if not messages or messages[-1].role != "assistant":
                    raise ValueError(f"{label} line {line_number} must end with assistant")
                rows.append(messages)
            if not rows:
                raise ValueError(f"{label} contains no rows")
            datums: list[Datum] = []
            template_digest: str | None = None
            render_digests: list[str] = []
            for messages in rows:
                full = engine.render_chat(
                    RenderChatRequest(
                        messages=messages,
                        tokenize=True,
                        add_generation_prompt=False,
                        enable_thinking=job.config.enable_thinking,
                    )
                )
                prompt = engine.render_chat(
                    RenderChatRequest(
                        messages=messages[:-1],
                        tokenize=True,
                        add_generation_prompt=True,
                        enable_thinking=job.config.enable_thinking,
                    )
                )
                if full.token_ids is None or prompt.token_ids is None:
                    raise RuntimeError("Qwen renderer did not return token IDs")
                if (
                    full.template_digest != prompt.template_digest
                    or full.enable_thinking != prompt.enable_thinking
                ):
                    raise RuntimeError("SFT/eval prompt and completion render contracts differ")
                boundary = 0
                for left, right in zip(prompt.token_ids, full.token_ids):
                    if left != right:
                        break
                    boundary += 1
                if boundary < max(1, len(prompt.token_ids) - 4):
                    raise ValueError("generation prompt and SFT completion lack a stable prefix")
                weights = [
                    1.0 if index >= boundary else 0.0 for index in range(1, len(full.token_ids))
                ]
                datums.append(
                    Datum(
                        input_ids=full.token_ids[:-1],
                        target_ids=full.token_ids[1:],
                        weights=weights,
                        metadata={
                            "render_digest": full.render_digest,
                            "prompt_render_digest": prompt.render_digest,
                            "template_digest": full.template_digest,
                            "enable_thinking": full.enable_thinking,
                        },
                    )
                )
                template_digest = full.template_digest
                render_digests.append(full.render_digest)
            return datums, template_digest, render_digests

        datums, template_digest, render_digests = render_dataset(
            Path(job.config.dataset.path), "training dataset"
        )
        eval_datums = []
        eval_render_digests: list[str] = []
        if job.config.evaluation_dataset is not None:
            eval_datums, eval_template_digest, eval_render_digests = render_dataset(
                Path(job.config.evaluation_dataset.path), "evaluation dataset"
            )
            if eval_template_digest != template_digest:
                raise RuntimeError("SFT and evaluation template digests differ")

        job.render_contract = {
            "template_digest": template_digest,
            "render_digests": render_digests,
            "evaluation_render_digests": eval_render_digests,
            "enable_thinking": job.config.enable_thinking,
        }
        self.store.save_job(job)

        if resuming:
            # Weights, Adam moments and the step counter come back together.
            checkpoint = Path(job.checkpoints[-1].path)
            engine.load_checkpoint(checkpoint.name)
            baseline_losses = [float(value) for value in job.evaluation.get("baseline_losses", [])]
            start_step = job.current_step + 1
        else:
            # The baseline is scored before the first update and persisted, so a
            # resumed run compares against the same untrained model rather than
            # re-scoring one that has already moved.
            baseline_losses = self._qwen_losses(engine, eval_datums) if eval_datums else []
            job = self.store.load_job(job.job_id)
            job.evaluation = {**job.evaluation, "baseline_losses": baseline_losses}
            self.store.save_job(job)
            start_step = 1

        optimizer = AdamParams(
            learning_rate=job.config.learning_rate,
            weight_decay=0.0,
            max_grad_norm=1.0,
        )
        batches = _MinibatchStream(
            count=len(datums),
            batch_size=job.config.batch_size,
            shuffle=job.config.shuffle,
            seed=job.config.seed,
        )
        # The stream is deterministic in `seed`, so replaying it to the resume
        # point reproduces the exact batch order rather than persisting it.
        for _ in range(start_step - 1):
            batches.next_batch()
        for step in range(start_step, job.config.max_steps + 1):
            self._cancel_boundary(cancelled)
            started = time.monotonic()
            indices = batches.next_batch()
            batch = [datums[index] for index in indices]
            # One optimizer step, however many forward passes it takes. The
            # engine accumulates each pass weighted by its own unmasked-token
            # count and divides by the summed weight at optim_step, so the
            # result is one mean over every token in the batch -- splitting for
            # memory does not change the update.
            weighted_loss = 0.0
            tokens = 0.0
            for start in range(0, len(batch), job.config.micro_batch_size):
                self._cancel_boundary(cancelled)
                micro = batch[start : start + job.config.micro_batch_size]
                forward = engine.forward_backward(
                    ForwardBackwardRequest(data=micro, loss_fn="cross_entropy")
                )
                micro_tokens = float(forward.metrics.get("token_count", 0.0))
                weighted_loss += float(forward.loss) * micro_tokens
                tokens += micro_tokens
            engine.optim_step(optimizer)
            self._record_step(
                job.job_id,
                step,
                loss=weighted_loss / tokens if tokens else 0.0,
                tokens=tokens,
                seconds=time.monotonic() - started,
                epoch=batches.epoch,
                memory=_mlx_peak_memory(),
            )
            if step % job.config.checkpoint_every == 0 or step == job.config.max_steps:
                saved = engine.save_checkpoint(f"{job.job_id}-step-{step:06d}")
                self._adapter_checkpoint(job.job_id, step, Path(saved.path))
        if eval_datums:
            trained_losses = self._qwen_losses(engine, eval_datums)
            self._record_qwen_evaluation(job.job_id, baseline_losses, trained_losses)

    def _run_cispo(
        self, job: Job, cancelled: threading.Event, resuming: bool = False
    ) -> None:
        """The on-policy lane, mirroring the hosted Tinker CISPO runner.

        One step is: collect `groups_per_step` groups of `group_size` rollouts at
        a pinned policy version, drop and resample any group whose rewards are
        all equal, normalize each surviving group into advantages, build one
        datum per sampled turn, and apply a single CISPO update.

        The structure is the hosted one on purpose. Same slime normalization,
        same filtered-group retry, same advantage placement over completion
        tokens only, same refusal to invent a signal where the rewards show
        none. What differs is the transport and the sampler: the container calls
        back into this service, so the behaviour log-probabilities it returns
        were produced by the very model being trained, at the policy version the
        step pinned. Hosted has to reason about two populations there; locally
        they are one, and the ratio denominator means what it says.
        """

        if self._engine_provider is None:
            raise RuntimeError("the resident Qwen MLX engine is not available")
        from synth_mlx_rl.rollout_client import ContainerRolloutClient
        from synth_mlx_rl.schemas import AdamParams, Datum, ForwardBackwardRequest

        engine = self._engine_provider()
        target = job.config.rollout
        assert target is not None  # config validation guarantees this

        client = ContainerRolloutClient(
            base_url=target.url,
            task_id=target.task_id,
            sampler_url=self._sampler_url,
            sampler_token=self._sampler_token,
            bearer_token=target.bearer_token,
            max_tokens=target.max_tokens,
            temperature=target.temperature,
            connection_mode=target.connection_mode,
        )
        # Rollout identity is the container's idempotency key, so it has to be
        # unique per launch. A stable id means a relaunched job replays the
        # previous attempt's cached result -- including a cached failure.
        launch = uuid.uuid4().hex[:8]
        capabilities = client.capabilities()
        if capabilities.get("max_concurrency", 1) < 1:
            raise RuntimeError("rollout_container_advertises_no_capacity")
        job = self.store.load_job(job.job_id)
        job.render_contract = {
            "rollout_container_id": capabilities.get("container_id"),
            "rollout_container_digest": capabilities.get("container_digest"),
            "rollout_capability_hash": capabilities.get("capability_hash"),
            "task_id": target.task_id,
            "objective": job.config.objective,
            "eps_low": job.config.eps_low,
            "eps_high": job.config.eps_high,
        }
        self.store.save_job(job)
        self.store.append_event(
            job.job_id, "rollout.container_admitted", dict(job.render_contract)
        )

        if resuming:
            engine.load_checkpoint(Path(job.checkpoints[-1].path).name)
            start_step = job.current_step + 1
        else:
            start_step = 1

        instances = _MinibatchStream(
            count=target.train_instances,
            batch_size=1,
            shuffle=True,
            seed=job.config.seed,
        )
        for _ in range((start_step - 1) * job.config.groups_per_step):
            instances.next_batch()

        baseline = job.evaluation.get("baseline")
        if not resuming:
            baseline = self._heldout_reward(job, client, cancelled, "baseline", launch)
            job = self.store.load_job(job.job_id)
            job.evaluation = {**job.evaluation, "baseline": baseline}
            self.store.save_job(job)

        optimizer = AdamParams(
            learning_rate=job.config.learning_rate,
            weight_decay=0.0,
            max_grad_norm=1.0,
        )
        for step in range(start_step, job.config.max_steps + 1):
            self._cancel_boundary(cancelled)
            started = time.monotonic()
            # Pin the policy for the whole step: every rollout in it must be
            # sampled from the same weights the update is computed against.
            snapshot = engine.publish_snapshot(
                metadata={"reason": "cispo_step", "job_id": job.job_id, "step": step}
            )
            policy_version = snapshot.id

            # Timed separately from the update. Without the split a step is one
            # opaque number and the first question anyone asks of an on-policy
            # lane -- is this environment-bound or compute-bound? -- cannot be
            # answered from the run's own record. On a slow or paid environment
            # that distinction is the whole cost model.
            rollout_started = time.monotonic()
            groups: list[tuple[list, list[float]]] = []
            attempts_used = 0
            for group_index in range(job.config.groups_per_step):
                summaries, rewards, attempts = self._collect_group(
                    job, client, cancelled, step, group_index, policy_version, instances, launch
                )
                attempts_used += attempts
                groups.append((summaries, rewards))
            rollout_seconds = time.monotonic() - rollout_started
            rollouts_collected = attempts_used * job.config.group_size
            rollouts_used = len(groups) * job.config.group_size

            batch: list = []
            advantage_values: list[float] = []
            reward_values: list[float] = []
            completion_tokens = 0
            for summaries, rewards in groups:
                advantages = normalize_group_rewards(rewards)
                advantage_values.extend(advantages)
                reward_values.extend(rewards)
                for summary, advantage in zip(summaries, advantages, strict=True):
                    for action in summary.actions:
                        datum, completions = _datum_from_action(
                            Datum, action, advantage, sequence_cap=job.config.sequence_cap
                        )
                        batch.append(datum)
                        completion_tokens += completions
            if not batch or completion_tokens == 0:
                raise RuntimeError("cispo_empty_training_batch")

            update_started = time.monotonic()
            weighted_loss = 0.0
            tokens = 0.0
            metrics: dict[str, object] = {}
            for start in range(0, len(batch), job.config.micro_batch_size):
                self._cancel_boundary(cancelled)
                micro = batch[start : start + job.config.micro_batch_size]
                forward = engine.forward_backward(
                    ForwardBackwardRequest(
                        data=micro,
                        loss_fn=job.config.objective,
                        eps_low=job.config.eps_low,
                        eps_high=job.config.eps_high,
                    )
                )
                micro_tokens = float(forward.metrics.get("token_count", 0.0))
                weighted_loss += float(forward.loss) * micro_tokens
                tokens += micro_tokens
                metrics = dict(forward.metrics)
            engine.optim_step(optimizer)
            update_seconds = time.monotonic() - update_started

            self._record_step(
                job.job_id,
                step,
                loss=weighted_loss / tokens if tokens else 0.0,
                tokens=tokens,
                seconds=time.monotonic() - started,
                epoch=1,
                memory=_mlx_peak_memory(),
                extra={
                    "objective": job.config.objective,
                    "policy_version": policy_version,
                    "rollouts": len(reward_values),
                    "reward_mean": fmean(reward_values),
                    "reward_std": pstdev(reward_values),
                    "advantage_mean": fmean(advantage_values),
                    "advantage_std": pstdev(advantage_values),
                    "clip_fraction": metrics.get("clip_fraction"),
                    "mean_ratio": metrics.get("mean_ratio"),
                    "completion_tokens": completion_tokens,
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    # Collected includes the groups thrown away for having no
                    # reward variance; used is what reached the optimizer. The
                    # gap between them is the real sample cost of this lane.
                    "rollouts_collected": rollouts_collected,
                    "rollouts_used": rollouts_used,
                    "groups_filtered": attempts_used - len(groups),
                },
            )
            if step % job.config.checkpoint_every == 0 or step == job.config.max_steps:
                saved = engine.save_checkpoint(f"{job.job_id}-step-{step:06d}")
                self._adapter_checkpoint(job.job_id, step, Path(saved.path))

        trained = self._heldout_reward(job, client, cancelled, "trained", launch)
        self._record_policy_evaluation(job.job_id, baseline, trained)

    def _heldout_reward(
        self, job: Job, client, cancelled: threading.Event, phase: str, launch: str
    ) -> dict:
        """Mean reward over the frozen held-out instances, greedily sampled.

        Greedy, because this is a measurement and not a search: temperature is
        what makes training groups differ, and letting it vary here would report
        sampling noise as a change in the policy. Same instances before and
        after, so the comparison is paired.
        """

        target = job.config.rollout
        assert target is not None
        rewards: list[float] = []
        for index in range(target.heldout_instances):
            self._cancel_boundary(cancelled)
            summary = client.rollout(
                job_id=job.job_id,
                attempt_id=f"{job.job_id}-eval-{phase}",
                policy_version=f"{phase}",
                rollout_id=f"{job.job_id}-{launch}-eval-{phase}-{index}",
                task_instance_id=f"seed:{index}",
                world_ref=target.heldout_world_ref,
                temperature=0.01,
            )
            rewards.append(summary.reward)
        outcome = {
            "phase": phase,
            "instances": len(rewards),
            "mean_reward": fmean(rewards) if rewards else 0.0,
            "rewards": rewards,
        }
        self.store.append_event(job.job_id, "heldout_eval.completed", outcome)
        return outcome

    def _record_policy_evaluation(self, job_id: str, baseline, trained) -> None:
        job = self.store.load_job(job_id)
        before = float((baseline or {}).get("mean_reward") or 0.0)
        after = float((trained or {}).get("mean_reward") or 0.0)
        wins = sum(
            1
            for b, a in zip((baseline or {}).get("rewards", []), trained.get("rewards", []))
            if a > b
        )
        losses = sum(
            1
            for b, a in zip((baseline or {}).get("rewards", []), trained.get("rewards", []))
            if a < b
        )
        job.evaluation = {
            **job.evaluation,
            "status": "completed",
            "baseline": baseline,
            "trained": trained,
            "mean_reward_before": before,
            "mean_reward_after": after,
            "uplift": after - before,
            "instances_improved": wins,
            "instances_regressed": losses,
        }
        job.updated_at = utc_now()
        self.store.save_job(job)
        self.store.append_event(
            job_id,
            "evaluation.completed",
            {
                "mean_reward_before": before,
                "mean_reward_after": after,
                "uplift": after - before,
                "instances_improved": wins,
                "instances_regressed": losses,
            },
        )

    def _collect_group(
        self,
        job: Job,
        client,
        cancelled: threading.Event,
        step: int,
        group_index: int,
        policy_version: str,
        instances: "_MinibatchStream",
        launch: str,
    ):
        """One group with a usable signal, or a failure that says why.

        A group shares one task instance, which is what makes its advantages
        group-*relative*. A group whose rewards are all equal defines no
        preference and is filtered rather than trained on -- and the retry draws
        a *different* instance, because retrying the same one mostly reproduces
        the same answer. An instance the policy fails uniformly is not a
        transient failure to sample through; it is a problem this policy cannot
        yet distinguish, and there is nothing to learn from four identical
        wrong answers to it.

        Hosted emits `rollout.group_filtered` with reason `zero_advantage` and
        eventually raises `cispo_no_learning_signal`; so does this.
        """

        for attempt in range(1, job.config.signal_attempts + 1):
            self._cancel_boundary(cancelled)
            instance = instances.next_batch()[0]
            summaries = []
            for member in range(job.config.group_size):
                summary = client.rollout(
                    job_id=job.job_id,
                    attempt_id=f"{job.job_id}-s{step:04d}-g{group_index}-a{attempt}",
                    policy_version=policy_version,
                    rollout_id=f"{job.job_id}-{launch}-s{step:04d}-g{group_index}-a{attempt}-r{member}",
                    task_instance_id=f"seed:{instance}",
                    world_ref=job.config.rollout.train_world_ref,
                )
                summaries.append(summary)
                self.store.append_event(
                    job.job_id,
                    "rollout.completed",
                    {
                        "step": step,
                        "rollout_id": summary.rollout_id,
                        "policy_version": policy_version,
                        "reward": summary.reward,
                        "container_digest": summary.container_digest,
                    },
                )
            rewards = [summary.reward for summary in summaries]
            if has_learning_signal(rewards):
                return summaries, rewards, attempt
            self.store.append_event(
                job.job_id,
                "rollout.group_filtered",
                {
                    "step": step,
                    "group": group_index,
                    "reason": "zero_advantage",
                    "signal_attempt": attempt,
                    "instance": instance,
                    "rewards": rewards,
                },
            )
        raise RuntimeError("cispo_no_learning_signal")

    @staticmethod
    def _qwen_losses(engine: TrainingEngine, datums: list) -> list[float]:
        losses = []
        for datum in datums:
            token_ids = [datum.input_ids[0], *datum.target_ids]
            logprobs = engine.score_logprobs(token_ids, policy_snapshot_id=None)[1:]
            weighted = [
                (-float(value), weight)
                for value, weight in zip(logprobs, datum.weights)
                if value is not None and weight > 0
            ]
            if not weighted:
                raise RuntimeError("evaluation item has no weighted assistant tokens")
            losses.append(
                sum(value * weight for value, weight in weighted)
                / sum(weight for _, weight in weighted)
            )
        return losses

    def _record_qwen_evaluation(self, job_id: str, before: list[float], after: list[float]) -> None:
        if len(before) != len(after) or not before:
            raise RuntimeError("paired evaluation outcomes are incomplete")
        outcomes = [
            {
                "item": index,
                "before_loss": left,
                "after_loss": right,
                "delta": right - left,
                "improved": right < left,
            }
            for index, (left, right) in enumerate(zip(before, after))
        ]
        evaluation = {
            "status": "completed",
            "schema_version": "synth_mlx_rl.paired_evaluation.v1",
            "items": outcomes,
            "item_count": len(outcomes),
            "mean_before_loss": sum(before) / len(before),
            "mean_after_loss": sum(after) / len(after),
            "mean_paired_delta": sum(right - left for left, right in zip(before, after))
            / len(before),
            "improved_items": sum(item["improved"] for item in outcomes),
            "baseline_policy": "resident_policy_at_job_start",
            "mcnemar": {
                "applicable": False,
                "reason": (
                    "held-out outcome is continuous token loss, not paired binary correctness"
                ),
            },
        }
        job = self.store.load_job(job_id)
        if job.config.evaluation_dataset is not None:
            evaluation["dataset_sha256"] = sha256_file(Path(job.config.evaluation_dataset.path))
        output = Path(job.config.output_dir) / "evaluation.json"
        output.write_text(json.dumps(evaluation, indent=2, sort_keys=True))
        evaluation["path"] = str(output)
        evaluation["sha256"] = sha256_file(output)
        job.evaluation = evaluation
        job.updated_at = utc_now()
        self.store.save_job(job)
        self.store.append_event(
            job_id,
            "evaluation.completed",
            {"item_count": len(outcomes), "sha256": evaluation["sha256"]},
        )

    def _adapter_checkpoint(self, job_id: str, step: int, path: Path) -> Checkpoint:
        required = [
            path / "adapter_config.json",
            path / "adapters.safetensors",
            path / "state.json",
        ]
        if not all(item.is_file() for item in required):
            raise RuntimeError("Qwen checkpoint is missing adapter lineage files")
        digest, total_bytes = sha256_path(path)
        checkpoint = Checkpoint(
            checkpoint_id=f"{job_id}:step-{step}",
            step=step,
            path=str(path),
            sha256=digest,
            bytes=total_bytes,
            created_at=utc_now(),
        )
        job = self.store.load_job(job_id)
        job.checkpoints.append(checkpoint)
        job.resume_supported = True
        job.recovery = "resume_from_checkpoint"
        job.updated_at = utc_now()
        self.store.save_job(job)
        self.store.append_event(
            job_id,
            "checkpoint.created",
            {**checkpoint.model_dump(), "kind": "mlx-lora.v1"},
        )
        return checkpoint

    @staticmethod
    def _cancel_boundary(cancelled: threading.Event) -> None:
        if cancelled.is_set():
            raise Cancelled()

    def _record_step(
        self,
        job_id: str,
        step: int,
        *,
        loss: float,
        tokens: float,
        seconds: float,
        epoch: int,
        memory: int | None,
        extra: dict[str, object] | None = None,
    ) -> None:
        job = self.store.load_job(job_id)
        job.current_step = step
        job.updated_at = utc_now()
        self.store.save_job(job)
        metric: dict[str, object] = {
            "step": step,
            "epoch": epoch,
            "loss": loss,
            "learning_rate": job.config.learning_rate,
            # Tokens and seconds, then the rate derived from them. The previous
            # field was named `throughput_steps_per_second` and carried a token
            # count, which is the kind of instrument that reads plausibly and
            # says nothing true.
            "tokens": tokens,
            "step_seconds": seconds,
            "tokens_per_second": tokens / seconds if seconds > 0 else 0.0,
            "memory_bytes": memory,
            "timestamp": utc_now(),
            **(extra or {}),
        }
        self.store.append_metric(job_id, metric)
        self.store.append_event(job_id, "training.metric", metric)

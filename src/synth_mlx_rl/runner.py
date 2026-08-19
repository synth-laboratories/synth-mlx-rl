"""The local training backend. One backend: a real Qwen LoRA fine-tune."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Protocol

from synth_mlx_rl.models import Checkpoint, Job, JobStatus
from synth_mlx_rl.storage import JobStore, sha256_file, sha256_path, utc_now


class TrainingEngine(Protocol):
    def render_chat(self, request): ...
    def forward_backward(self, request): ...
    def optim_step(self, params): ...
    def save_checkpoint(self, name: str): ...
    def score_logprobs(self, token_ids: list[int], *, policy_snapshot_id: str | None = None): ...


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
    ) -> None:
        self.store = store
        self._engine_provider = engine_provider
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

    def _run(self, job_id: str, cancelled: threading.Event) -> None:
        job = self.store.load_job(job_id)
        job.status = JobStatus.RUNNING
        job.started_at = utc_now()
        job.updated_at = job.started_at
        self.store.save_job(job)
        self.store.append_event(job_id, "job.started", {"backend": job.config.backend})
        try:
            if job.config.backend != "qwen_lora":  # pydantic validates before persistence
                raise RuntimeError(f"unsupported backend {job.config.backend}")
            self._run_qwen_lora(job, cancelled)
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

    def _run_qwen_lora(self, job: Job, cancelled: threading.Event) -> None:
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
        baseline_losses = self._qwen_losses(engine, eval_datums) if eval_datums else []
        optimizer = AdamParams(
            learning_rate=job.config.learning_rate,
            weight_decay=0.0,
            max_grad_norm=1.0,
        )
        for step in range(1, job.config.max_steps + 1):
            self._cancel_boundary(cancelled)
            forward = engine.forward_backward(
                ForwardBackwardRequest(data=datums, loss_fn="cross_entropy")
            )
            engine.optim_step(optimizer)
            self._record_step(
                job.job_id,
                step,
                loss=forward.loss,
                throughput=float(forward.metrics.get("token_count", 0.0)),
                memory=_mlx_peak_memory(),
            )
            if step % job.config.checkpoint_every == 0 or step == job.config.max_steps:
                saved = engine.save_checkpoint(f"{job.job_id}-step-{step:06d}")
                self._adapter_checkpoint(job.job_id, step, Path(saved.path))
        if eval_datums:
            trained_losses = self._qwen_losses(engine, eval_datums)
            self._record_qwen_evaluation(job.job_id, baseline_losses, trained_losses)

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
        throughput: float,
        memory: int | None,
    ) -> None:
        job = self.store.load_job(job_id)
        job.current_step = step
        job.updated_at = utc_now()
        self.store.save_job(job)
        metric: dict[str, object] = {
            "step": step,
            "loss": loss,
            "learning_rate": job.config.learning_rate,
            "throughput_steps_per_second": throughput,
            "memory_bytes": memory,
            "timestamp": utc_now(),
        }
        self.store.append_metric(job_id, metric)
        self.store.append_event(job_id, "training.metric", metric)

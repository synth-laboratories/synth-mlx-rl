"""Bounded local backends for the first Workshop-compatible training path."""

from __future__ import annotations

import json
import threading
import time

from synth_mlx_rl.models import Checkpoint, Job, JobStatus
from synth_mlx_rl.storage import JobStore, sha256_file, utc_now


class Cancelled(Exception):
    """Raised when a user-cancelled job reaches a safe step boundary."""


class TrainingRunner:
    """Runs bounded jobs in service-owned threads.

    The service intentionally offers no claimed auto-resume: an MLX optimizer
    state is not durable in this v0.6 slice. A restarted active job becomes
    `interrupted`; completed state and terminal checkpoints remain reopenable.
    """

    def __init__(self, store: JobStore) -> None:
        self.store = store
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
            if job.config.backend == "fixture":
                self._run_fixture(job, cancelled)
            elif job.config.backend == "mlx_scalar_smoke":
                self._run_mlx_scalar(job, cancelled)
            else:  # Defensive: pydantic validates this before persistence.
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

    def _run_fixture(self, job: Job, cancelled: threading.Event) -> None:
        """Deterministic protocol fixture; it never calls itself MLX training."""
        parameter = 1.0
        for step in range(1, job.config.max_steps + 1):
            self._cancel_boundary(cancelled)
            parameter -= job.config.learning_rate * parameter
            self._record_step(
                job.job_id,
                step,
                loss=parameter * parameter,
                throughput=1000.0,
                memory=None,
            )
            if step % job.config.checkpoint_every == 0 or step == job.config.max_steps:
                self._checkpoint(job.job_id, step, {"backend": "fixture", "parameter": parameter})
            time.sleep(0.01)

    def _run_mlx_scalar(self, job: Job, cancelled: threading.Event) -> None:
        """Actual MLX compute smoke, explicitly not a language-model fine-tune."""
        try:
            import mlx.core as mx
        except ImportError as exc:
            raise RuntimeError("mlx is not installed; use `uv sync --extra mlx`") from exc

        mx.random.seed(job.config.seed)
        parameter = mx.array(1.0)

        def loss_fn(value):
            return value * value

        grad_fn = mx.grad(loss_fn)
        for step in range(1, job.config.max_steps + 1):
            self._cancel_boundary(cancelled)
            gradient = grad_fn(parameter)
            parameter = parameter - job.config.learning_rate * gradient
            loss = loss_fn(parameter)
            mx.eval(parameter, loss)
            self._record_step(
                job.job_id,
                step,
                loss=float(loss.item()),
                throughput=1.0,
                memory=int(mx.get_peak_memory()),
            )
            if step % job.config.checkpoint_every == 0 or step == job.config.max_steps:
                self._checkpoint(
                    job.job_id,
                    step,
                    {"backend": "mlx_scalar_smoke", "parameter": float(parameter.item())},
                )

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

    def _checkpoint(self, job_id: str, step: int, payload: dict[str, object]) -> Checkpoint:
        job = self.store.load_job(job_id)
        directory = self.store.job_dir(job_id) / "checkpoints"
        directory.mkdir(exist_ok=True)
        path = directory / f"step-{step:06d}.json"
        content = {
            "schema_version": "synth_mlx_rl.checkpoint.v1",
            "step": step,
            **payload,
        }
        path.write_text(json.dumps(content, sort_keys=True))
        checkpoint = Checkpoint(
            checkpoint_id=f"{job_id}:step-{step}",
            step=step,
            path=str(path),
            sha256=sha256_file(path),
            bytes=path.stat().st_size,
            created_at=utc_now(),
        )
        job.checkpoints.append(checkpoint)
        job.updated_at = utc_now()
        self.store.save_job(job)
        self.store.append_event(job_id, "checkpoint.created", checkpoint.model_dump())
        return checkpoint

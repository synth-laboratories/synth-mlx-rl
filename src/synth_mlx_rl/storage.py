"""Atomic local job storage with append-only events and digested artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from synth_mlx_rl.models import Event, Job, JobStatus


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> tuple[str, int]:
    """Content digest and byte count for one file or a deterministic directory tree."""
    if path.is_file():
        return sha256_file(path), path.stat().st_size
    digest = hashlib.sha256()
    total = 0
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix().encode()
        content = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        total += len(content)
    return digest.hexdigest(), total


class JobStore:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        # Sequence allocation and append are one operation. Training and HTTP
        # cancellation run on different threads and must never race here.
        self._events_lock = threading.RLock()

    def job_dir(self, job_id: str) -> Path:
        if not job_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("job_id must contain only letters, digits, hyphens, and underscores")
        return self.root / job_id

    def configure(self, job: Job) -> None:
        directory = self.job_dir(job.job_id)
        directory.mkdir(parents=True, exist_ok=False)
        self.save_job(job)
        self.append_event(job.job_id, "job.configured", {"config_sha256": job.config_sha256})

    def save_job(self, job: Job) -> None:
        path = self.job_dir(job.job_id) / "job.json"
        self._atomic_write(path, job.model_dump(mode="json"))

    def load_job(self, job_id: str) -> Job:
        path = self.job_dir(job_id) / "job.json"
        return Job.model_validate_json(path.read_text())

    def list_jobs(self) -> list[Job]:
        jobs: list[Job] = []
        for path in sorted(self.root.glob("*/job.json")):
            jobs.append(Job.model_validate_json(path.read_text()))
        return jobs

    def append_event(
        self,
        job_id: str,
        type_: str,
        payload: dict[str, object] | None = None,
    ) -> Event:
        with self._events_lock:
            directory = self.job_dir(job_id)
            events_path = directory / "events.jsonl"
            sequence = 1
            if events_path.exists():
                with events_path.open("rb") as handle:
                    for sequence, _line in enumerate(handle, start=1):
                        pass
                sequence += 1
            now = utc_now()
            event = Event(
                sequence=sequence,
                type=type_,
                kind=type_,
                timestamp=now,
                occurred_at=now,
                payload=payload or {},
                schema_version="training.event.v1",
                event_id=f"{job_id}:{sequence}",
                job_id=job_id,
                attempt_id="attempt-1",
                producer={
                    "service": "synth-mlx-rl",
                    "version": "0.6.0",
                    "commit": "synth-mlx-rl",
                },
            )
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(event.model_dump_json() + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event

    def events_after(self, job_id: str, after: int = 0) -> list[Event]:
        path = self.job_dir(job_id) / "events.jsonl"
        if not path.exists():
            return []
        return [
            event
            for event in (
                Event.model_validate_json(line) for line in path.read_text().splitlines() if line
            )
            if event.sequence > after
        ]

    def append_metric(self, job_id: str, metric: dict[str, Any]) -> None:
        path = self.job_dir(job_id) / "metrics.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metric, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def recover_interrupted(self) -> list[str]:
        recovered: list[str] = []
        for job in self.list_jobs():
            if job.status in {JobStatus.QUEUED, JobStatus.RUNNING, JobStatus.CANCELLING}:
                job.status = JobStatus.INTERRUPTED
                job.error_code = "service_restarted"
                job.error_detail = (
                    "The service restarted mid-run. Resume from the last checkpoint with "
                    "POST /v1/jobs/{job_id}/resume, or leave it terminal."
                    if job.checkpoints
                    else "The service restarted before any checkpoint was written; there is "
                    "nothing to resume from."
                )
                job.finished_at = utc_now()
                job.updated_at = job.finished_at
                self.save_job(job)
                self.append_event(job.job_id, "job.interrupted", {"reason": "service_restarted"})
                recovered.append(job.job_id)
        return recovered

    @staticmethod
    def _atomic_write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", delete=False, dir=path.parent
        ) as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            temporary = Path(handle.name)
        temporary.replace(path)

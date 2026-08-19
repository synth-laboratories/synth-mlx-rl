"""Public wire models for the v0.6 local-training vertical slice.

The endpoints deliberately model a job as a durable artifact.  The Tinker
training subset can be layered on this state machine without teaching Workshop
two incompatible notions of a local run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class JobStatus(StrEnum):
    CONFIGURED = "configured"
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATUSES = {
    JobStatus.CANCELLED,
    JobStatus.SUCCEEDED,
    JobStatus.FAILED,
    JobStatus.INTERRUPTED,
}


class DatasetSpec(BaseModel):
    path: str
    sha256: str | None = None


class TrainingConfig(BaseModel):
    """A bounded, local-only training request.

    `fixture` is intentional for contract testing.  `mlx_scalar_smoke` runs
    actual MLX operations but is only an infrastructure smoke, never model
    fine-tuning.  Unsupported model backends fail preflight rather than falling
    back silently.
    """

    backend: Literal["fixture", "mlx_scalar_smoke"] = "fixture"
    base_model: str = "Qwen/Qwen3.5-0.8B"
    dataset: DatasetSpec
    output_dir: str
    max_steps: int = Field(default=4, ge=1, le=10_000)
    checkpoint_every: int = Field(default=1, ge=1, le=10_000)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    seed: int = 0
    max_disk_bytes: int = Field(default=8 * 1024**3, ge=1)

    @field_validator("output_dir")
    @classmethod
    def output_dir_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output_dir must not be empty")
        return value


class ConfigureRequest(BaseModel):
    job_id: str | None = None
    config: TrainingConfig


class Event(BaseModel):
    sequence: int
    type: str
    timestamp: str
    payload: dict[str, object] = Field(default_factory=dict)


class Checkpoint(BaseModel):
    checkpoint_id: str
    step: int
    path: str
    sha256: str
    bytes: int
    created_at: str


class Job(BaseModel):
    job_id: str
    status: JobStatus
    config: TrainingConfig
    config_sha256: str
    dataset_sha256: str
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    current_step: int = 0
    error_code: str | None = None
    error_detail: str | None = None
    checkpoints: list[Checkpoint] = Field(default_factory=list)
    metrics_path: str
    events_path: str
    manifest_path: str
    resume_supported: bool = False
    recovery: Literal["reopen", "restart_from_checkpoint_unsupported"] = "reopen"


class Capability(BaseModel):
    supported: bool
    reason: str | None = None


class Capabilities(BaseModel):
    schema_version: Literal["synth_mlx_rl.capabilities.v1"] = "synth_mlx_rl.capabilities.v1"
    service_version: str
    platform: str
    architecture: str
    memory_bytes: int | None
    available_disk_bytes: int
    capabilities: dict[str, Capability]


class Preflight(BaseModel):
    accepted: bool
    checks: dict[str, Capability]
    estimated_disk_bytes: int
    config_sha256: str
    dataset_sha256: str | None = None


class Handoff(BaseModel):
    job_id: str
    checkpoint: Checkpoint
    inference: dict[str, str]
    provenance: dict[str, str]
    evaluation: dict[str, str]

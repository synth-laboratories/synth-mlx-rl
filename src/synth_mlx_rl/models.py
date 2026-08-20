"""Public wire models for the v0.6 local-training vertical slice.

The endpoints deliberately model a job as a durable artifact.  The Tinker
training subset can be layered on this state machine without teaching Workshop
two incompatible notions of a local run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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


class RolloutTarget(BaseModel):
    """A task container speaking the hosted `training.rollout.request.v1`."""

    url: str
    task_id: str
    bearer_token: str | None = None
    max_tokens: int = Field(default=512, ge=1, le=16_384)
    #: Sampling temperature for training rollouts. Zero is a mistake here: a
    #: group of identical greedy samples has no reward variance and the step is
    #: filtered, so the lane would never find a signal.
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    connection_mode: Literal["close", "keep_alive"] = "close"
    #: Task instances to draw training groups from, as `seed:N` for N in range.
    train_instances: int = Field(default=64, ge=1, le=100_000)
    train_world_ref: str | None = None
    #: The frozen slice the before/after comparison is measured on. Never
    #: trained against.
    heldout_world_ref: str | None = None
    heldout_instances: int = Field(default=16, ge=1, le=10_000)


class TrainingConfig(BaseModel):
    """A bounded, local-only training request.

    Two lanes, both real. `qwen_lora` is offline supervised fine-tuning from a
    dataset. `cispo` is the on-policy lane: it collects grouped rollouts from a
    task container, turns their rewards into group advantages, and applies the
    CISPO objective. Both produce a deployable `mlx-lora.v1` adapter.

    The `cispo` lane mirrors the hosted Tinker runner deliberately -- the same
    group normalization, the same refusal to train on a group with no reward
    variance, the same advantage placement over completion tokens only. A local
    result computed from a different advantage definition would not be
    comparable to a hosted one, which is most of the point of having both.
    """

    backend: Literal["qwen_lora", "cispo"] = "qwen_lora"
    base_model: str = "Qwen/Qwen3.5-0.8B"
    #: Required by `qwen_lora`; unused by `cispo`, whose data is its rollouts.
    dataset: DatasetSpec | None = None
    evaluation_dataset: DatasetSpec | None = None

    # --- on-policy lane -------------------------------------------------
    rollout: RolloutTarget | None = None
    #: Rollouts per group. Two is the minimum that can define an advantage.
    group_size: int = Field(default=4, ge=2, le=64)
    groups_per_step: int = Field(default=1, ge=1, le=64)
    #: How many times to resample a group that came back with no reward
    #: variance before failing the step. Hosted calls this a filtered group.
    signal_attempts: int = Field(default=6, ge=1, le=32)
    objective: Literal["cispo_minimax", "cispo_two_sided"] = "cispo_minimax"
    #: Clip bounds are `[1 - eps_low, 1 + eps_high]`. `cispo_minimax` is
    #: single-sided by definition and refuses an active lower bound.
    eps_low: float = Field(default=1.0, ge=0.0, le=1.0)
    eps_high: float = Field(default=4.0, gt=0.0, le=64.0)
    sequence_cap: int = Field(default=8192, ge=16, le=131_072)
    output_dir: str
    #: One step is one optimizer update over `batch_size` rows. The rows are
    #: fed to the engine `micro_batch_size` at a time and their gradients are
    #: accumulated, so batch size is a *statistical* choice and micro-batch size
    #: is a *memory* one. They are separate knobs because peak memory scales
    #: with the tokens in a single forward pass, not with the update size.
    max_steps: int = Field(default=4, ge=1, le=10_000)
    batch_size: int = Field(default=8, ge=1, le=8192)
    micro_batch_size: int = Field(default=1, ge=1, le=1024)
    #: Reshuffled every epoch from `seed`. Off means the dataset order is the
    #: gradient order, which correlates consecutive updates.
    shuffle: bool = True
    checkpoint_every: int = Field(default=1, ge=1, le=10_000)
    learning_rate: float = Field(default=0.01, gt=0, le=1)
    lora_rank: int = Field(default=8, ge=1, le=256)
    lora_alpha: float = Field(default=16.0, gt=0)
    max_seq_length: int = Field(default=1024, ge=8, le=4096)
    enable_thinking: bool = False
    seed: int = 0
    max_disk_bytes: int = Field(default=8 * 1024**3, ge=1)

    @field_validator("output_dir")
    @classmethod
    def output_dir_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output_dir must not be empty")
        return value

    @model_validator(mode="after")
    def lane_requirements_are_met(self) -> "TrainingConfig":
        if self.backend == "qwen_lora" and self.dataset is None:
            raise ValueError("qwen_lora needs a dataset")
        if self.backend == "cispo":
            if self.rollout is None:
                raise ValueError(
                    "cispo needs a rollout target: an on-policy lane without an "
                    "environment has no reward to learn from"
                )
            if self.objective == "cispo_minimax" and self.eps_low < 1.0:
                raise ValueError(
                    "cispo_minimax is single-sided by definition: eps_low must be "
                    "1.0 so the lower clip bound is inactive. Ask for "
                    "cispo_two_sided if an active lower bound is what you want."
                )
        return self

    @model_validator(mode="after")
    def micro_batch_fits_in_batch(self) -> "TrainingConfig":
        if self.micro_batch_size > self.batch_size:
            raise ValueError(
                "micro_batch_size must not exceed batch_size: a micro-batch is a "
                "slice of one update, not a larger one"
            )
        return self


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
    #: True once a checkpoint exists: the engine persists adapter weights, the
    #: Adam moments and the step, so a resumed run continues rather than
    #: restarting. It is not bit-exact when `lora_dropout > 0`, because the MLX
    #: RNG stream is not persisted -- with the default dropout of 0.0 that has
    #: no effect on the update.
    resume_supported: bool = False
    recovery: Literal["reopen", "resume_from_checkpoint"] = "reopen"
    render_contract: dict[str, object] = Field(default_factory=dict)
    evaluation: dict[str, object] = Field(default_factory=dict)


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
    qwen_lora_contract: dict[str, object]


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
    evaluation: dict[str, object]

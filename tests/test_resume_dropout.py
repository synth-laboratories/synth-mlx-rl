"""Resume is refused when LoRA dropout would make the run diverge."""

from __future__ import annotations

import pytest

from synth_mlx_rl.models import Checkpoint, DatasetSpec, Job, JobStatus, TrainingConfig
from synth_mlx_rl.runner import TrainingRunner
from synth_mlx_rl.storage import JobStore, utc_now


def test_resume_refuses_dropout(tmp_path) -> None:
    store = JobStore(tmp_path / "jobs")
    job_dir = store.job_dir("job-drop")
    job_dir.mkdir()
    now = utc_now()
    config = TrainingConfig(
        dataset=DatasetSpec(path=str(tmp_path / "train.jsonl")),
        output_dir=str(tmp_path / "out"),
        lora_dropout=0.1,
        max_steps=4,
    )
    job = Job(
        job_id="job-drop",
        status=JobStatus.INTERRUPTED,
        config=config,
        config_sha256="abc",
        dataset_sha256="def",
        created_at=now,
        updated_at=now,
        checkpoints=[
            Checkpoint(
                checkpoint_id="ckpt-1",
                step=1,
                path=str(job_dir / "ckpt"),
                sha256="aa",
                bytes=1,
                created_at=now,
            )
        ],
        current_step=1,
        metrics_path=str(job_dir / "metrics.jsonl"),
        events_path=str(job_dir / "events.jsonl"),
        manifest_path=str(job_dir / "job.json"),
        resume_supported=True,
    )
    store.save_job(job)
    runner = TrainingRunner(store=store)
    with pytest.raises(ValueError, match="lora_dropout"):
        runner.resume("job-drop")

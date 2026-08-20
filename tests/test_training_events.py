"""Append-only job events carry training.event.v1 identity."""

from __future__ import annotations

from pathlib import Path

from synth_mlx_rl.storage import JobStore


def test_appended_events_carry_training_event_identity(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    (store.root / "job-1").mkdir()
    event = store.append_event("job-1", "training.metric", {"step": 1, "loss": 0.4})
    assert event.schema_version == "training.event.v1"
    assert event.event_id == "job-1:1"
    assert event.job_id == "job-1"
    assert event.attempt_id == "attempt-1"
    assert event.kind == "training.metric"
    assert event.occurred_at == event.timestamp
    assert event.producer["service"] == "synth-mlx-rl"
    replayed = store.events_after("job-1", after=0)
    assert replayed[0].event_id == "job-1:1"

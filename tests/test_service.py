"""Wire and durability tests that run without MLX or a model download."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from synth_mlx_rl.config import Settings
from synth_mlx_rl.service import create_app
from synth_mlx_rl.testing import FakeEngine


def _payload(tmp_path: Path, *, job_id: str = "fixture-run", steps: int = 3) -> dict[str, object]:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text('{"prompt":"2+2","completion":"4"}\n')
    return {
        "job_id": job_id,
        "config": {
            "backend": "fixture",
            "dataset": {"path": str(dataset)},
            "output_dir": str(tmp_path / f"output-{job_id}"),
            "max_steps": steps,
            "checkpoint_every": 2,
            "learning_rate": 0.1,
            "seed": 3,
        },
    }


def _wait_for_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(100):
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "cancelled", "failed", "interrupted"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not become terminal")


def test_preflight_rejects_missing_dataset(tmp_path: Path) -> None:
    client = TestClient(create_app(tmp_path / "service"))
    payload = _payload(tmp_path)
    payload["config"]["dataset"]["path"] = str(tmp_path / "missing.jsonl")  # type: ignore[index]

    response = client.post("/v1/jobs/preflight", json=payload)

    assert response.status_code == 200
    assert response.json()["accepted"] is False
    assert response.json()["checks"]["dataset"]["supported"] is False
    assert client.post("/v1/jobs", json=payload).status_code == 422


def test_fixture_job_has_live_metrics_terminal_digest_and_durable_handoff(tmp_path: Path) -> None:
    root = tmp_path / "service"
    client = TestClient(create_app(root))
    payload = _payload(tmp_path)

    preflight = client.post("/v1/jobs/preflight", json=payload).json()
    assert preflight["accepted"] is True
    configured = client.post("/v1/jobs", json=payload)
    assert configured.status_code == 201
    assert client.post("/v1/jobs/fixture-run/launch").status_code == 202

    job = _wait_for_terminal(client, "fixture-run")
    assert job["status"] == "succeeded"
    assert job["current_step"] == 3
    assert len(job["checkpoints"]) == 2
    terminal = job["checkpoints"][-1]
    checkpoint = Path(terminal["path"])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == terminal["sha256"]

    events = client.get("/v1/jobs/fixture-run/events").json()["events"]
    assert any(event["type"] == "training.metric" for event in events)
    assert events[-1]["type"] == "job.succeeded"
    assert len((root / "fixture-run" / "metrics.jsonl").read_text().splitlines()) == 3

    handoff = client.get("/v1/jobs/fixture-run/handoff")
    assert handoff.status_code == 200
    assert handoff.json()["checkpoint"]["sha256"] == terminal["sha256"]
    assert handoff.json()["inference"]["kind"] == "non_inference_smoke_checkpoint"
    assert handoff.json()["evaluation"]["status"] == "not_run"

    reopened = TestClient(create_app(root))
    reopened_job = reopened.get("/v1/jobs/fixture-run").json()
    assert reopened_job["status"] == "succeeded"
    assert reopened.get("/v1/jobs/fixture-run/handoff").json()["checkpoint"] == terminal


def test_restart_marks_active_job_interrupted_without_claiming_resume(tmp_path: Path) -> None:
    root = tmp_path / "service"
    client = TestClient(create_app(root))
    payload = _payload(tmp_path, job_id="recover-me")
    assert client.post("/v1/jobs", json=payload).status_code == 201

    job_path = root / "recover-me" / "job.json"
    job = json.loads(job_path.read_text())
    job["status"] = "running"
    job_path.write_text(json.dumps(job))

    reopened = TestClient(create_app(root))
    recovered = reopened.get("/v1/jobs/recover-me").json()
    assert recovered["status"] == "interrupted"
    assert recovered["resume_supported"] is False
    assert recovered["error_code"] == "service_restarted"
    assert reopened.get("/healthz").json()["recovered_jobs"] == ["recover-me"]


def test_cancellation_is_terminal_and_keeps_completed_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "service"
    client = TestClient(create_app(root))
    payload = _payload(tmp_path, job_id="cancel-me", steps=1_000)
    assert client.post("/v1/jobs", json=payload).status_code == 201
    assert client.post("/v1/jobs/cancel-me/launch").status_code == 202

    for _ in range(100):
        status = client.get("/v1/jobs/cancel-me").json()["status"]
        if status in {"running", "cancelling"}:
            break
        time.sleep(0.01)
    assert client.post("/v1/jobs/cancel-me/cancel").status_code == 202
    job = _wait_for_terminal(client, "cancel-me")
    assert job["status"] == "cancelled"
    assert (root / "cancel-me" / "events.jsonl").exists()


@pytest.mark.mlx
def test_mlx_compute_smoke_records_real_peak_memory(tmp_path: Path) -> None:
    pytest.importorskip("mlx.core")
    root = tmp_path / "service"
    client = TestClient(create_app(root))
    payload = _payload(tmp_path, job_id="mlx-smoke", steps=2)
    payload["config"]["backend"] = "mlx_scalar_smoke"  # type: ignore[index]
    assert client.post("/v1/jobs/preflight", json=payload).json()["accepted"] is True
    assert client.post("/v1/jobs", json=payload).status_code == 201
    assert client.post("/v1/jobs/mlx-smoke/launch").status_code == 202

    job = _wait_for_terminal(client, "mlx-smoke")
    assert job["status"] == "succeeded"
    metrics = [
        json.loads(line) for line in (root / "mlx-smoke" / "metrics.jsonl").read_text().splitlines()
    ]
    assert len(metrics) == 2
    assert all(isinstance(metric["memory_bytes"], int) for metric in metrics)


def test_qwen_lora_job_persists_real_adapter_contract_and_render_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "service"
    settings = Settings(
        model="Qwen/Qwen3.5-0.8B",
        checkpoint_dir=root / "adapters",
        lora_rank=8,
        lora_alpha=16.0,
        max_seq_length=1024,
        enable_thinking=False,
    )
    engine = FakeEngine(checkpoint_dir=root / "adapters")
    dataset = tmp_path / "qwen.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"Say ready"},'
        '{"role":"assistant","content":"READY"}]}\n'
    )
    payload = {
        "job_id": "qwen-lora",
        "config": {
            "backend": "qwen_lora",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "dataset": {"path": str(dataset)},
            "output_dir": str(tmp_path / "qwen-output"),
            "max_steps": 2,
            "checkpoint_every": 2,
            "learning_rate": 5e-5,
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "max_seq_length": 1024,
            "enable_thinking": False,
        },
    }
    with TestClient(create_app(root, settings=settings, engine=engine)) as client:
        capabilities = client.get("/v1/capabilities").json()
        assert capabilities["qwen_lora_contract"] == {
            "backend": "qwen_lora",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "lora_rank": 8,
            "lora_alpha": 16.0,
            "max_seq_length": 1024,
            "enable_thinking": False,
            "adapter_kind": "mlx-lora.v1",
            "renderer": "qwen-chat-template.v1",
        }
        preflight = client.post("/v1/jobs/preflight", json=payload).json()
        assert preflight["accepted"] is True
        assert client.post("/v1/jobs", json=payload).status_code == 201
        assert client.post("/v1/jobs/qwen-lora/launch").status_code == 202
        job = _wait_for_terminal(client, "qwen-lora")
        assert job["status"] == "succeeded"
        assert job["render_contract"]["enable_thinking"] is False
        assert job["render_contract"]["template_digest"]
        checkpoint = Path(job["checkpoints"][-1]["path"])
        assert (checkpoint / "adapter_config.json").is_file()
        assert (checkpoint / "adapters.safetensors").is_file()
        handoff = client.get("/v1/jobs/qwen-lora/handoff").json()
        assert handoff["inference"]["kind"] == "mlx-lora.v1"
        assert handoff["checkpoint"]["sha256"] == job["checkpoints"][-1]["sha256"]


def test_qwen_preflight_fails_closed_on_resident_render_contract_mismatch(
    tmp_path: Path,
) -> None:
    settings = Settings(model="Qwen/Qwen3.5-0.8B", lora_rank=8, max_seq_length=1024)
    engine = FakeEngine(checkpoint_dir=tmp_path / "adapters")
    payload = _payload(tmp_path, job_id="mismatch")
    payload["config"].update(  # type: ignore[union-attr]
        {
            "backend": "qwen_lora",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "lora_rank": 16,
            "lora_alpha": 16.0,
            "max_seq_length": 1024,
            "enable_thinking": False,
        }
    )
    with TestClient(
        create_app(tmp_path / "service", settings=settings, engine=engine)
    ) as client:
        preflight = client.post("/v1/jobs/preflight", json=payload).json()
    assert preflight["accepted"] is False
    assert "do not match the resident service" in preflight["checks"]["backend"]["reason"]

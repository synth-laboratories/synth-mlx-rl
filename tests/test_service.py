"""Wire and durability tests for the one real backend.

There is no fixture or smoke backend to test against: the service offers
`qwen_lora` and nothing else, and these tests drive it with the real resident
MLX engine. There is no fake engine to fall back to: the whole job state machine
-- preflight, configure, launch, metrics, cancel, checkpoint, handoff, reopen --
runs against a model that is actually loaded.

Every client is entered as a context manager: the engine binds in the app
lifespan, so a client that never starts has no engine and every job fails.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

from synth_mlx_rl.config import Settings
from synth_mlx_rl.service import LocalTrainingService, create_app
from synth_mlx_rl.storage import sha256_path

RESIDENT = dict(
    model="Qwen/Qwen3.5-0.8B",
    lora_rank=8,
    lora_alpha=16.0,
    max_seq_length=1024,
    enable_thinking=False,
)


@contextmanager
def _client(root: Path, tmp_path: Path) -> Iterator[TestClient]:
    """A service whose only backend is real, driven by a fake engine."""
    settings = Settings(checkpoint_dir=tmp_path / "adapters", **RESIDENT)
    _require_real_backend(settings)
    with TestClient(create_app(root, settings=settings)) as client:
        yield client


def _require_real_backend(settings: Settings) -> None:
    pytest.importorskip("mlx.core")
    pytest.importorskip("mlx_lm")
    if settings.local_model_path() is None:
        pytest.skip("Qwen/Qwen3.5-0.8B is not downloaded locally")


def _payload(tmp_path: Path, *, job_id: str = "local-run", steps: int = 3) -> dict[str, object]:
    dataset = tmp_path / f"data-{job_id}.jsonl"
    dataset.write_text('{"prompt":"2+2","completion":"4"}\n')
    return {
        "job_id": job_id,
        "config": {
            "backend": "qwen_lora",
            "base_model": RESIDENT["model"],
            "dataset": {"path": str(dataset)},
            "output_dir": str(tmp_path / f"output-{job_id}"),
            "max_steps": steps,
            "checkpoint_every": 2,
            "learning_rate": 5e-5,
            "lora_rank": RESIDENT["lora_rank"],
            "lora_alpha": RESIDENT["lora_alpha"],
            "max_seq_length": RESIDENT["max_seq_length"],
            "enable_thinking": RESIDENT["enable_thinking"],
            "seed": 3,
        },
    }


def _wait_for_terminal(client: TestClient, job_id: str) -> dict[str, object]:
    for _ in range(400):
        response = client.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "cancelled", "failed", "interrupted"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not become terminal")


def test_the_service_offers_exactly_one_backend(tmp_path: Path) -> None:
    with _client(tmp_path / "service", tmp_path) as client:
        capabilities = client.get("/v1/capabilities").json()["capabilities"]
    # A backend that cannot produce a deployable model is not offered at all.
    assert "fixture_training" not in capabilities
    assert "mlx_scalar_smoke" not in capabilities
    assert capabilities["qwen_lora_training"]["supported"] is True


def test_cispo_capability_is_independent_of_sft_runtime_availability(
    tmp_path: Path,
) -> None:
    service = LocalTrainingService(
        tmp_path / "service",
        Settings(checkpoint_dir=tmp_path / "adapters"),
        qwen_available_override=True,
        cispo_available_override=False,
    )
    capabilities = service.capabilities().capabilities
    assert capabilities["qwen_lora_training"].supported is True
    assert capabilities["cispo_training"].supported is False


def test_preflight_refuses_when_no_managed_path_or_cache_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    settings = Settings(
        model_path=tmp_path / "missing-managed-snapshot",
        checkpoint_dir=tmp_path / "adapters",
        lora_rank=8,
        lora_alpha=16.0,
        max_seq_length=1024,
        enable_thinking=False,
    )
    with TestClient(create_app(tmp_path / "service", settings=settings)) as client:
        preflight = client.post(
            "/v1/jobs/preflight", json=_payload(tmp_path, job_id="missing-model")
        ).json()
    assert preflight["accepted"] is False
    assert preflight["checks"]["model"]["supported"] is False
    assert "Workshop Settings" in preflight["checks"]["model"]["reason"]


def test_managed_model_snapshot_is_preferred_over_hf_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    managed = tmp_path / "managed"
    managed.mkdir()
    for name in ("config.json", "tokenizer_config.json", "model.safetensors"):
        (managed / name).write_text("{}")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "other-cache"))
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    assert Settings(model_path=managed).local_model_path() == managed.resolve()


def test_an_unknown_backend_is_refused_at_the_wire(tmp_path: Path) -> None:
    with _client(tmp_path / "service", tmp_path) as client:
        payload = _payload(tmp_path)
        payload["config"]["backend"] = "fixture"  # type: ignore[index]
        assert client.post("/v1/jobs/preflight", json=payload).status_code == 422
        assert client.post("/v1/jobs", json=payload).status_code == 422


def test_preflight_rejects_missing_dataset(tmp_path: Path) -> None:
    with _client(tmp_path / "service", tmp_path) as client:
        payload = _payload(tmp_path)
        payload["config"]["dataset"]["path"] = str(tmp_path / "missing.jsonl")  # type: ignore[index]

        response = client.post("/v1/jobs/preflight", json=payload)

        assert response.status_code == 200
        assert response.json()["accepted"] is False
        assert response.json()["checks"]["dataset"]["supported"] is False
        assert client.post("/v1/jobs", json=payload).status_code == 422


def test_preflight_rejects_dataset_digest_mismatch(tmp_path: Path) -> None:
    with _client(tmp_path / "service", tmp_path) as client:
        payload = _payload(tmp_path)
        payload["config"]["dataset"]["sha256"] = "0" * 64  # type: ignore[index]

        preflight = client.post("/v1/jobs/preflight", json=payload).json()

    assert preflight["accepted"] is False
    assert "does not match" in preflight["checks"]["dataset"]["reason"]


def test_job_has_live_metrics_terminal_digest_and_durable_handoff(tmp_path: Path) -> None:
    root = tmp_path / "service"
    payload = _payload(tmp_path)
    with _client(root, tmp_path) as client:
        preflight = client.post("/v1/jobs/preflight", json=payload).json()
        assert preflight["accepted"] is True
        assert client.post("/v1/jobs", json=payload).status_code == 201
        assert client.post("/v1/jobs/local-run/launch").status_code == 202

        job = _wait_for_terminal(client, "local-run")
        assert job["status"] == "succeeded"
        assert job["current_step"] == 3
        assert len(job["checkpoints"]) == 2
        terminal = job["checkpoints"][-1]
        checkpoint = Path(terminal["path"])
        # The checkpoint is an adapter directory; the digest covers the tree.
        assert (checkpoint / "adapters.safetensors").is_file()
        assert sha256_path(checkpoint)[0] == terminal["sha256"]

        events = client.get("/v1/jobs/local-run/events").json()["events"]
        assert any(event["type"] == "training.metric" for event in events)
        assert events[-1]["type"] == "job.succeeded"
        assert len((root / "local-run" / "metrics.jsonl").read_text().splitlines()) == 3

        handoff = client.get("/v1/jobs/local-run/handoff")
        assert handoff.status_code == 200
        assert handoff.json()["checkpoint"]["sha256"] == terminal["sha256"]
        assert handoff.json()["inference"]["kind"] == "mlx-lora.v1"
        assert handoff.json()["evaluation"]["status"] == "not_run"

    with _client(root, tmp_path) as reopened:
        reopened_job = reopened.get("/v1/jobs/local-run").json()
        assert reopened_job["status"] == "succeeded"
        assert reopened.get("/v1/jobs/local-run/handoff").json()["checkpoint"] == terminal


def test_restart_marks_active_job_interrupted_without_claiming_resume(tmp_path: Path) -> None:
    root = tmp_path / "service"
    payload = _payload(tmp_path, job_id="recover-me")
    with _client(root, tmp_path) as client:
        assert client.post("/v1/jobs", json=payload).status_code == 201

    job_path = root / "recover-me" / "job.json"
    job = json.loads(job_path.read_text())
    job["status"] = "running"
    job_path.write_text(json.dumps(job))

    with _client(root, tmp_path) as reopened:
        recovered = reopened.get("/v1/jobs/recover-me").json()
        assert recovered["status"] == "interrupted"
        assert recovered["resume_supported"] is False
        assert recovered["error_code"] == "service_restarted"
        assert reopened.get("/healthz").json()["recovered_jobs"] == ["recover-me"]


def test_cancellation_is_terminal_and_keeps_completed_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "service"
    payload = _payload(tmp_path, job_id="cancel-me", steps=10_000)
    with _client(root, tmp_path) as client:
        assert client.post("/v1/jobs", json=payload).status_code == 201
        assert client.post("/v1/jobs/cancel-me/launch").status_code == 202

        for _ in range(200):
            status = client.get("/v1/jobs/cancel-me").json()["status"]
            if status in {"running", "cancelling"}:
                break
            time.sleep(0.01)
        assert client.post("/v1/jobs/cancel-me/cancel").status_code == 202
        job = _wait_for_terminal(client, "cancel-me")
        assert job["status"] == "cancelled"
        assert (root / "cancel-me" / "events.jsonl").exists()


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
    _require_real_backend(settings)
    dataset = tmp_path / "qwen.jsonl"
    dataset.write_text(
        '{"messages":[{"role":"user","content":"Say ready"},'
        '{"role":"assistant","content":"READY"}]}\n'
    )
    evaluation_dataset = tmp_path / "qwen-eval.jsonl"
    evaluation_dataset.write_text(
        '{"prompt":"Reply yes","completion":"yes"}\n{"prompt":"Reply no","completion":"no"}\n'
    )
    payload = {
        "job_id": "qwen-lora",
        "config": {
            "backend": "qwen_lora",
            "base_model": "Qwen/Qwen3.5-0.8B",
            "dataset": {"path": str(dataset)},
            "evaluation_dataset": {"path": str(evaluation_dataset)},
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
    with TestClient(create_app(root, settings=settings)) as client:
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
        assert len(job["render_contract"]["evaluation_render_digests"]) == 2
        assert job["evaluation"]["status"] == "completed"
        assert job["evaluation"]["item_count"] == 2
        assert len(job["evaluation"]["items"]) == 2
        assert job["evaluation"]["mcnemar"]["applicable"] is False
        checkpoint = Path(job["checkpoints"][-1]["path"])
        assert (checkpoint / "adapter_config.json").is_file()
        assert (checkpoint / "adapters.safetensors").is_file()
        handoff = client.get("/v1/jobs/qwen-lora/handoff").json()
        assert handoff["inference"]["kind"] == "mlx-lora.v1"
        assert handoff["checkpoint"]["sha256"] == job["checkpoints"][-1]["sha256"]
        assert handoff["evaluation"]["status"] == "completed"


def test_qwen_preflight_fails_closed_on_resident_render_contract_mismatch(
    tmp_path: Path,
) -> None:
    settings = Settings(model="Qwen/Qwen3.5-0.8B", lora_rank=8, max_seq_length=1024)
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
    with TestClient(create_app(tmp_path / "service", settings=settings)) as client:
        preflight = client.post("/v1/jobs/preflight", json=payload).json()
    assert preflight["accepted"] is False
    assert "do not match the resident service" in preflight["checks"]["backend"]["reason"]

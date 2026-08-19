"""FastAPI service for durable local MLX training jobs."""

from __future__ import annotations

import os
import platform
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from synth_mlx_rl import __version__
from synth_mlx_rl.models import Capabilities, Capability, ConfigureRequest, Handoff, Job, Preflight
from synth_mlx_rl.runner import TrainingRunner
from synth_mlx_rl.storage import JobStore, canonical_json, sha256_bytes, sha256_file, utc_now


def _memory_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None
    try:
        return int(os.popen("sysctl -n hw.memsize").read().strip())
    except (ValueError, OSError):
        return None


class LocalTrainingService:
    def __init__(self, root: Path) -> None:
        self.store = JobStore(root)
        self.runner = TrainingRunner(self.store)
        self.recovered_jobs = self.store.recover_interrupted()

    def capabilities(self) -> Capabilities:
        disk = shutil.disk_usage(self.store.root)
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        try:
            import mlx  # noqa: F401

            mlx_available = True
        except ImportError:
            mlx_available = False
        return Capabilities(
            service_version=__version__,
            platform=platform.system().lower(),
            architecture=platform.machine(),
            memory_bytes=_memory_bytes(),
            available_disk_bytes=disk.free,
            capabilities={
                "local_training": Capability(
                    supported=apple_silicon,
                    reason=None if apple_silicon else "requires macOS arm64",
                ),
                "fixture_training": Capability(supported=True),
                "mlx_scalar_smoke": Capability(
                    supported=apple_silicon and mlx_available,
                    reason=(
                        None
                        if apple_silicon and mlx_available
                        else "MLX unavailable or unsupported platform"
                    ),
                ),
                "qwen_lora_training": Capability(
                    supported=False,
                    reason="not implemented in the v0.6 closure slice",
                ),
                "automatic_resume": Capability(
                    supported=False,
                    reason="optimizer state is not yet durable",
                ),
                "tinker_training_subset": Capability(
                    supported=False,
                    reason="local job API only in this closure slice",
                ),
            },
        )

    def preflight(self, request: ConfigureRequest) -> Preflight:
        config = request.config
        checks: dict[str, Capability] = {}
        dataset = Path(config.dataset.path).expanduser()
        if dataset.is_file():
            dataset_digest = sha256_file(dataset)
            checks["dataset"] = Capability(supported=True)
        else:
            dataset_digest = None
            checks["dataset"] = Capability(supported=False, reason="dataset path is not a file")
        output = Path(config.output_dir).expanduser()
        output_parent = output if output.is_dir() else output.parent
        while not output_parent.exists() and output_parent != output_parent.parent:
            output_parent = output_parent.parent
        free = shutil.disk_usage(output_parent).free
        estimated = max(1024 * 1024, dataset.stat().st_size * 3 if dataset.is_file() else 0)
        disk_needed = min(config.max_disk_bytes, estimated)
        checks["disk"] = Capability(
            supported=free >= disk_needed,
            reason=None if free >= disk_needed else f"need {disk_needed} bytes; have {free}",
        )
        capability_name = {
            "fixture": "fixture_training",
            "mlx_scalar_smoke": "mlx_scalar_smoke",
        }[config.backend]
        capability = self.capabilities().capabilities[capability_name]
        checks["backend"] = capability
        checks["output"] = Capability(
            supported=output_parent.is_dir(),
            reason=None
            if output_parent.is_dir()
            else "no existing parent directory for output_dir",
        )
        config_digest = sha256_bytes(canonical_json(config.model_dump(mode="json")))
        return Preflight(
            accepted=all(check.supported for check in checks.values()),
            checks=checks,
            estimated_disk_bytes=estimated,
            config_sha256=config_digest,
            dataset_sha256=dataset_digest,
        )

    def configure(self, request: ConfigureRequest) -> Job:
        preflight = self.preflight(request)
        if not preflight.accepted:
            raise ValueError("preflight failed")
        job_id = request.job_id or f"mlx-{uuid.uuid4().hex[:12]}"
        output = Path(request.config.output_dir).expanduser().resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("output_dir must be empty for a new job")
        output.mkdir(parents=True, exist_ok=True)
        now = utc_now()
        service_dir = self.store.job_dir(job_id)
        job = Job(
            job_id=job_id,
            status="configured",
            config=request.config,
            config_sha256=preflight.config_sha256,
            dataset_sha256=preflight.dataset_sha256 or "",
            created_at=now,
            updated_at=now,
            metrics_path=str(service_dir / "metrics.jsonl"),
            events_path=str(service_dir / "events.jsonl"),
            manifest_path=str(service_dir / "job.json"),
        )
        self.store.configure(job)
        # A reproducible public-facing copy remains with output artifacts.
        (output / "manifest.json").write_text(job.model_dump_json(indent=2))
        return job


def create_app(root: str | Path = ".synth-mlx-rl") -> FastAPI:
    service = LocalTrainingService(Path(root))
    app = FastAPI(title="synth-mlx-rl", version=__version__)
    app.state.service = service

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "recovered_jobs": service.recovered_jobs}

    @app.get("/v1/capabilities", response_model=Capabilities)
    def capabilities() -> Capabilities:
        return service.capabilities()

    @app.post("/v1/jobs/preflight", response_model=Preflight)
    def preflight(request: ConfigureRequest) -> Preflight:
        return service.preflight(request)

    @app.post("/v1/jobs", response_model=Job, status_code=201)
    def configure(request: ConfigureRequest) -> Job:
        try:
            return service.configure(request)
        except (ValueError, FileExistsError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/jobs", response_model=list[Job])
    def list_jobs() -> list[Job]:
        return service.store.list_jobs()

    @app.get("/v1/jobs/{job_id}", response_model=Job)
    def get_job(job_id: str) -> Job:
        try:
            return service.store.load_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.post("/v1/jobs/{job_id}/launch", response_model=Job, status_code=202)
    def launch(job_id: str) -> Job:
        try:
            return service.runner.launch(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/jobs/{job_id}/cancel", response_model=Job, status_code=202)
    def cancel(job_id: str) -> Job:
        try:
            return service.runner.cancel(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/jobs/{job_id}/events")
    def events(job_id: str, after: int = Query(default=0, ge=0)) -> dict[str, object]:
        try:
            items = service.store.events_after(job_id, after)
            return {"events": [event.model_dump(mode="json") for event in items]}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/v1/jobs/{job_id}/events/stream")
    def event_stream(job_id: str, after: int = Query(default=0, ge=0)) -> StreamingResponse:
        try:
            service.store.load_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

        def generate():
            cursor = after
            while True:
                events = service.store.events_after(job_id, cursor)
                for event in events:
                    cursor = event.sequence
                    yield (
                        f"id: {event.sequence}\nevent: {event.type}\n"
                        f"data: {event.model_dump_json()}\n\n"
                    )
                job = service.store.load_job(job_id)
                if job.status.value in {"succeeded", "cancelled", "failed", "interrupted"}:
                    break
                yield ": keepalive\n\n"
                import time

                time.sleep(0.2)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/v1/jobs/{job_id}/handoff", response_model=Handoff)
    def handoff(job_id: str) -> Handoff:
        try:
            job = service.store.load_job(job_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if not job.checkpoints:
            raise HTTPException(status_code=409, detail="no checkpoint is available")
        checkpoint = job.checkpoints[-1]
        return Handoff(
            job_id=job_id,
            checkpoint=checkpoint,
            inference={
                "kind": "non_inference_smoke_checkpoint",
                "path": checkpoint.path,
                "requested_base_model": job.config.base_model,
            },
            provenance={"config_sha256": job.config_sha256, "dataset_sha256": job.dataset_sha256},
            evaluation={
                "status": "not_run",
                "reason": "v0.6 smoke backends do not create a deployable model",
            },
        )

    return app

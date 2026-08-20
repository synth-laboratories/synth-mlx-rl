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
from synth_mlx_rl.api.app import create_app as create_learning_app
from synth_mlx_rl.config import Settings
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


def _available_memory_bytes() -> int | None:
    """Memory the allocator could actually get, not total installed."""

    if platform.system() != "Darwin":
        return None
    try:
        out = os.popen("vm_stat").read()
    except OSError:
        return None
    page = 4096
    reclaimable = 0
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split("bytes")[0].strip())
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if key.strip() in {"Pages free", "Pages inactive", "Pages speculative"} and value.isdigit():
            reclaimable += int(value)
    return reclaimable * page or None


#: Peak MLX allocation is approximately linear in the tokens of a single
#: forward/backward pass. Fitted on this backend with gradient checkpointing on,
#: Qwen3.5-0.8B rank-8: 758 tokens -> 15.54 GB, 3090 tokens -> 57.61 GB. That is
#: 18.0 MB per token over a 1.9 GB resident floor. Without checkpointing the
#: slope is roughly 42 MB/token, which is why it is not optional here.
PEAK_BYTES_PER_TOKEN = 18_000_000
RESIDENT_FLOOR_BYTES = 1_900_000_000
#: Leave the machine able to do something other than swap.
MEMORY_HEADROOM_BYTES = 4 * 1024**3
#: Rough bytes-per-token for chat text. Only used to size a refusal, and the
#: estimate is reported so a caller can see what it was refused on.
BYTES_PER_TOKEN_ESTIMATE = 3.5


def estimated_peak_bytes(*, dataset_bytes: int, rows: int, micro_batch_size: int,
                         max_seq_length: int) -> int:
    """What one forward/backward over a micro-batch is expected to peak at."""

    if rows <= 0:
        return RESIDENT_FLOOR_BYTES
    tokens_per_row = min(
        float(max_seq_length), (dataset_bytes / rows) / BYTES_PER_TOKEN_ESTIMATE
    )
    tokens = tokens_per_row * min(micro_batch_size, rows)
    return int(RESIDENT_FLOOR_BYTES + PEAK_BYTES_PER_TOKEN * tokens)


class LocalTrainingService:
    def __init__(
        self,
        root: Path,
        settings: Settings,
        engine_provider=None,
        qwen_available_override: bool | None = None,
        sampler_base_url: str = "http://127.0.0.1:8787",
    ) -> None:
        self.store = JobStore(root)
        self.settings = settings
        self.qwen_available_override = qwen_available_override
        self.sampler_base_url = sampler_base_url.rstrip("/")
        self.runner = TrainingRunner(
            self.store,
            engine_provider,
            sampler_url=f"{sampler_base_url.rstrip('/')}/v1/training/sample",
        )
        self.recovered_jobs = self.store.recover_interrupted()

    def capabilities(self) -> Capabilities:
        disk = shutil.disk_usage(self.store.root)
        apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
        try:
            import mlx  # noqa: F401
            import mlx_lm  # noqa: F401

            mlx_available = True
        except ImportError:
            mlx_available = False
        qwen_available = (
            self.qwen_available_override
            if self.qwen_available_override is not None
            else apple_silicon and mlx_available
        )
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
                "qwen_lora_training": Capability(
                    supported=qwen_available,
                    reason=(
                        None
                        if qwen_available
                        else "Qwen LoRA requires Apple Silicon, mlx, and mlx-lm"
                    ),
                ),
                "cispo_training": Capability(
                    supported=qwen_available,
                    reason=(
                        None
                        if qwen_available
                        else "the on-policy lane needs the same MLX stack as SFT"
                    ),
                ),
                "resume_from_checkpoint": Capability(
                    supported=qwen_available,
                    reason=(
                        None
                        if qwen_available
                        else "resume needs the same engine that wrote the checkpoint"
                    ),
                ),
                "tinker_training_subset": Capability(
                    supported=False,
                    reason="local job API only in this closure slice",
                ),
            },
            qwen_lora_contract={
                "backend": "qwen_lora",
                "base_model": self.settings.model,
                "lora_rank": self.settings.lora_rank,
                "lora_alpha": self.settings.lora_alpha,
                "max_seq_length": self.settings.max_seq_length,
                "enable_thinking": self.settings.enable_thinking,
                "adapter_kind": "mlx-lora.v1",
                "renderer": "qwen-chat-template.v1",
            },
        )

    def preflight(self, request: ConfigureRequest) -> Preflight:
        config = request.config
        checks: dict[str, Capability] = {}
        # The on-policy lane has no dataset: its data is the rollouts it
        # collects, so a dataset check there would refuse a valid config.
        dataset = Path(config.dataset.path).expanduser() if config.dataset else None
        dataset_digest = None
        if dataset is not None:
            if dataset.is_file():
                dataset_digest = sha256_file(dataset)
                digest_matches = config.dataset.sha256 in {None, dataset_digest}
                checks["dataset"] = Capability(
                    supported=digest_matches,
                    reason=None if digest_matches else "dataset sha256 does not match file bytes",
                )
            else:
                checks["dataset"] = Capability(
                    supported=False, reason="dataset path is not a file"
                )
        if config.evaluation_dataset is not None:
            evaluation_dataset = Path(config.evaluation_dataset.path).expanduser()
            evaluation_digest = (
                sha256_file(evaluation_dataset) if evaluation_dataset.is_file() else None
            )
            evaluation_matches = config.evaluation_dataset.sha256 in {
                None,
                evaluation_digest,
            }
            checks["evaluation_dataset"] = Capability(
                supported=evaluation_dataset.is_file() and evaluation_matches,
                reason=(
                    None
                    if evaluation_dataset.is_file() and evaluation_matches
                    else (
                        "evaluation dataset sha256 does not match file bytes"
                        if evaluation_dataset.is_file()
                        else "evaluation dataset path is not a file"
                    )
                ),
            )
        output = Path(config.output_dir).expanduser()
        output_parent = output if output.is_dir() else output.parent
        while not output_parent.exists() and output_parent != output_parent.parent:
            output_parent = output_parent.parent
        free = shutil.disk_usage(output_parent).free
        dataset_on_disk = dataset is not None and dataset.is_file()
        estimated = max(1024 * 1024, dataset.stat().st_size * 3 if dataset_on_disk else 0)
        disk_needed = min(config.max_disk_bytes, estimated)
        checks["disk"] = Capability(
            supported=free >= disk_needed,
            reason=None if free >= disk_needed else f"need {disk_needed} bytes; have {free}",
        )
        # Memory, not disk, is what actually ends a run on this hardware, and it
        # scales with the tokens in one forward pass -- so it is a function of
        # micro_batch_size, never of dataset size. A full-batch config that would
        # peak past the machine is refused here rather than discovered by
        # watching the allocator thrash.
        rows = 0
        dataset_bytes = 0
        if dataset_on_disk:
            dataset_bytes = dataset.stat().st_size
            rows = sum(1 for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip())
        if config.backend == "cispo":
            # A rollout's length is bounded by the prompt plus `max_tokens`, and
            # a micro-batch holds that many sequences.
            rows = max(1, config.micro_batch_size)
            dataset_bytes = int(
                rows * config.rollout.max_tokens * 2 * BYTES_PER_TOKEN_ESTIMATE
            )
        estimated = estimated_peak_bytes(
            dataset_bytes=dataset_bytes,
            rows=rows,
            micro_batch_size=config.micro_batch_size,
            max_seq_length=config.max_seq_length,
        )
        available = _available_memory_bytes()
        budget = None if available is None else available - MEMORY_HEADROOM_BYTES
        checks["memory"] = Capability(
            supported=budget is None or estimated <= budget,
            reason=(
                None
                if budget is None or estimated <= budget
                else (
                    f"a micro-batch of {config.micro_batch_size} row(s) is estimated to peak at "
                    f"{estimated / 1024**3:.1f} GB, and only {available / 1024**3:.1f} GB is "
                    f"available. Lower micro_batch_size -- batch_size can stay where it is, "
                    f"because accumulation makes them independent."
                )
            ),
        )
        if config.backend == "cispo":
            # Spend-free: ask the container what it can do before any rollout
            # runs. An on-policy step that discovers its environment is
            # unreachable has already published a snapshot and burned a step.
            from synth_mlx_rl.rollout_client import ContainerRolloutClient, RolloutError

            assert config.rollout is not None
            probe = ContainerRolloutClient(
                base_url=config.rollout.url,
                task_id=config.rollout.task_id,
                sampler_url=f"{self.sampler_base_url}/v1/training/sample",
                sampler_token="preflight",
                bearer_token=config.rollout.bearer_token,
                timeout_seconds=15.0,
            )
            try:
                advertised = probe.capabilities()
            except RolloutError as exc:
                checks["rollout"] = Capability(supported=False, reason=str(exc))
            else:
                room = int(advertised.get("max_concurrency") or 0)
                checks["rollout"] = Capability(
                    supported=room >= 1,
                    reason=(
                        None
                        if room >= 1
                        else "container advertises no rollout concurrency"
                    ),
                )

        capability = self.capabilities().capabilities[
            "cispo_training" if config.backend == "cispo" else "qwen_lora_training"
        ]
        if config.base_model != "Qwen/Qwen3.5-0.8B":
            capability = Capability(
                supported=False,
                reason="v0.6 local SFT supports exactly Qwen/Qwen3.5-0.8B",
            )
        if (
            config.base_model != self.settings.model
            or config.lora_rank != self.settings.lora_rank
            or config.lora_alpha != self.settings.lora_alpha
            or config.max_seq_length != self.settings.max_seq_length
            or config.enable_thinking != self.settings.enable_thinking
        ):
            capability = Capability(
                supported=False,
                reason=(
                    "requested Qwen render/LoRA settings do not match the resident service; "
                    "restart the service with the requested model, rank, alpha, sequence "
                    "length, and thinking mode"
                ),
            )
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


def create_app(
    root: str | Path = ".synth-mlx-rl",
    *,
    settings: Settings | None = None,
    engine=None,
    sampler_base_url: str = "http://127.0.0.1:8787",
) -> FastAPI:
    root_path = Path(root).expanduser().resolve()
    configured = (settings or Settings.from_env()).with_overrides(
        checkpoint_dir=root_path / "adapters"
    )
    app = create_learning_app(engine=engine, settings=configured)
    # Replace the inference-only health route with a service health response
    # that remains useful while the resident model is still loading.
    app.router.routes = [
        route for route in app.router.routes if getattr(route, "path", None) != "/healthz"
    ]
    service = LocalTrainingService(
        root_path,
        configured,
        lambda: app.state.engine,
        qwen_available_override=True if engine is not None else None,
        sampler_base_url=sampler_base_url,
    )
    app.state.service = service

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "recovered_jobs": service.recovered_jobs,
            "model": configured.model,
        }

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

    @app.post("/v1/training/sample")
    def training_sample(body: dict) -> dict:
        """The sampler leg of the on-policy lane, in the hosted shape.

        A task container does not speak this service's OpenAI surface; it speaks
        the hosted sampler contract, because that is what the cloud lane hands
        it. Serving the same shape locally is what lets one unmodified container
        drive either lane.

        Sampling runs with truncation disabled -- no top-p, no top-k, no min-p.
        That is deliberate: `rollout_logprobs` are read from the full next-token
        distribution, so with truncation off they *are* log pi(a|s) under the
        pinned policy, which is the ratio denominator CISPO needs. Truncated
        sampling would return values from a distribution the trainer never had,
        and the importance ratio would quietly compare two different policies.
        """

        from synth_mlx_rl.schemas import ChatMessage, SampleRequest

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(status_code=422, detail="messages are required")
        try:
            request = SampleRequest(
                messages=[ChatMessage.model_validate(item) for item in messages],
                max_tokens=int(body.get("max_tokens") or 128),
                temperature=float(body.get("temperature") or 0.0),
                top_p=1.0,
                top_k=0,
                min_p=0.0,
                num_samples=1,
                policy_snapshot_id=body.get("policy_snapshot_id"),
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        sample = app.state.engine.sample(request).samples[0]
        return {
            "schema_version": "training.rollout.action.v1",
            "text": sample.text,
            "prompt_token_ids": sample.prompt_token_ids,
            "token_ids": sample.completion_token_ids,
            "log_probs": sample.rollout_logprobs,
            "policy_version": sample.policy_snapshot_id,
            "usage": {
                "prompt_tokens": len(sample.prompt_token_ids),
                "completion_tokens": len(sample.completion_token_ids),
                "total_tokens": len(sample.prompt_token_ids)
                + len(sample.completion_token_ids),
            },
        }

    @app.post("/v1/jobs/{job_id}/resume", response_model=Job, status_code=202)
    def resume(job_id: str) -> Job:
        try:
            return service.runner.resume(job_id)
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
        evaluation = dict(job.evaluation)
        if "status" not in evaluation:
            evaluation.update(
                status="not_run",
                reason="evaluation dataset was not configured",
            )
        return Handoff(
            job_id=job_id,
            checkpoint=checkpoint,
            inference={
                "kind": "mlx-lora.v1",
                "path": checkpoint.path,
                "requested_base_model": job.config.base_model,
            },
            provenance={
                "config_sha256": job.config_sha256,
                "dataset_sha256": job.dataset_sha256,
                "render_contract": canonical_json(job.render_contract).decode(),
            },
            evaluation=evaluation,
        )

    return app

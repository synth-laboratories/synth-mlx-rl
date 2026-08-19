# synth-mlx-rl

The v0.6 closure slice is a durable, local Apple-Silicon training-job service.
Its intentionally narrow path is:

```text
capabilities/preflight → configure → launch → metrics/events → checkpoint → handoff → reopen
```

It is a Workshop-facing local-job protocol, not yet a claim of Tinker training
API support.  In particular, Qwen LoRA/QLoRA, Tinker client creation, and
automatic checkpoint resume are explicitly reported as unsupported by
`GET /v1/capabilities`; callers must not infer them from the project name.

## Quick verification (Apple Silicon)

```bash
uv sync --extra dev --extra mlx
uv run pytest
uv run ruff check src tests
```

Start a local service with a service-owned artifact root:

```bash
uv run synth-mlx-rl serve --root .synth-mlx-rl --port 8787
curl http://127.0.0.1:8787/v1/capabilities
```

The supported real-compute smoke backend is `mlx_scalar_smoke`. It executes a
bounded MLX gradient update and records MLX peak memory, but is *not* a model
fine-tune and its checkpoint is not inference deployable. `fixture` is only a
deterministic protocol test backend.

## Local-job API for Workshop

1. `POST /v1/jobs/preflight` validates dataset identity, output directory,
   local disk, platform, and requested backend without configuring a run.
2. `POST /v1/jobs` persists an immutable job manifest. The requested
   `output_dir` must be empty.
3. `POST /v1/jobs/{job_id}/launch` starts the bounded job.
4. Read live history from `GET /v1/jobs/{job_id}/events?after=N` or the SSE
   endpoint `/v1/jobs/{job_id}/events/stream`; metrics are additionally
   persisted in `metrics.jsonl`.
5. `POST /v1/jobs/{job_id}/cancel` takes effect at a safe step boundary.
6. `GET /v1/jobs/{job_id}/handoff` returns the terminal checkpoint ID, SHA-256,
   provenance, and an honest non-deployable/evaluation status.

Service artifacts are append-only events/metrics plus atomically written job
state. On process restart, an active job becomes `interrupted`; completed jobs,
checkpoints, and handoff information reopen durably. No endpoint will represent
an interrupted run as resumed.

## Boundaries for v0.6

- This release contains no Public Cookbook packaging and no model/backend
  expansion.
- It does not download or train `Qwen/Qwen3.5-0.8B` yet.
- A future first-class Tinker bridge must advertise only operations proven
  against an MLX LoRA backend; it must not map unsupported operations to this
  smoke backend.

## License

Apache-2.0

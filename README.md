# synth-mlx-rl

A durable local Apple-Silicon training-job service. It runs on the user's
machine, on the user's hardware, against the user's files, and nothing it
produces leaves that machine.

```text
capabilities/preflight → configure → launch → metrics/events → checkpoint → handoff → reopen
```

## Two lanes, both real

`qwen_lora` is offline supervised fine-tuning from a dataset. `cispo` is the
on-policy lane: it collects grouped rollouts from a task container, turns their
rewards into group advantages, and applies the CISPO objective. Both fine-tune
`Qwen/Qwen3.5-0.8B` through the resident MLX engine and write an `mlx-lora.v1`
adapter you can serve.

There is no fixture backend and no compute-smoke backend: a job that cannot
produce a usable model has no place on a product surface. There is no fake
engine either -- the test suite runs against a resident model.

The `cispo` lane mirrors the hosted Tinker runner on purpose: the same slime
group normalization, the same refusal to train on a group whose rewards are all
equal, the same advantage placement over completion tokens only. A local number
computed from a different advantage definition would not be comparable to a
hosted one, which is most of why both lanes exist. It talks to the task
container over the hosted `training.rollout.request.v1` contract and serves the
hosted sampler shape at `/v1/training/sample`, so one unmodified container can
drive either lane.

`GET /v1/capabilities` is the authority on what this host can do. It reports
`qwen_lora_training` against the actual platform, MLX install, and resident
model, and it reports honestly on what is still missing:

```text
qwen_lora_training       offline SFT, when Apple Silicon + mlx + mlx-lm are present
cispo_training           the on-policy lane, same stack
resume_from_checkpoint   caller-initiated, restores weights and Adam moments
local_training           requires macOS arm64
tinker_training_subset   false — local job API only
```

Measured on Qwen3.5-0.8B, one 768-token datum: gradient checkpointing off →
33.40 GB peak / 9.8 train tok/s; on → 7.30 GB / 35.0 tok/s. It is faster *and*
smaller, so it is on by default. Retained activations across 28 layers cost
~42 MB/token, and above Metal's recommended working set the allocator thrashes
and time goes superlinear — which is also why the base model is pinned rather
than configurable. Preflight refuses a config whose model, rank, alpha,
sequence length, or thinking mode differs from the resident service.

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

## Local-job API for Workshop

1. `POST /v1/jobs/preflight` validates dataset identity, output directory,
   local disk, platform, and the resident model contract without configuring a
   run.
2. `POST /v1/jobs` persists an immutable job manifest. The requested
   `output_dir` must be empty.
3. `POST /v1/jobs/{job_id}/launch` starts the bounded job.
4. Read live history from `GET /v1/jobs/{job_id}/events?after=N` or the SSE
   endpoint `/v1/jobs/{job_id}/events/stream`; metrics are additionally
   persisted in `metrics.jsonl`.
5. `POST /v1/jobs/{job_id}/cancel` takes effect at a safe step boundary.
6. `GET /v1/jobs/{job_id}/handoff` returns the terminal checkpoint id, its
   SHA-256 over the adapter tree, provenance, and the paired held-out
   evaluation result — or an explicit `not_run` when no evaluation dataset was
   configured.

Supply an `evaluation_dataset` and the run scores the adapter against it before
and after training on the same rows. A number without that pairing is not a
result.

Service artifacts are append-only events and metrics plus atomically written
job state. On process restart an active job becomes `interrupted`; completed
jobs, checkpoints, and handoff information reopen durably. No endpoint will
represent an interrupted run as resumed.

## The on-policy lane

```bash
# a task container speaking the hosted rollout contract, on loopback
SYNTH_CONTAINERS_ALLOW_LOOPBACK_SAMPLER=1 SYNTH_BANKING77_SOURCE=hf \
  python -c "import uvicorn; from synth_containers.platform import create_compat_app; \
             uvicorn.run(create_compat_app('banking77_classify'), port=8114)"

python scripts/real_cispo_run.py --steps 12 --group-size 4
```

RL needs a policy that already succeeds sometimes. Asked to classify a
card-fraud query, the base model answers `fraud` -- reasonable, and not one of
the 77 labels, so exact match scores it zero, every rollout in the group scores
zero, and the lane correctly refuses a group with no reward variance. Warm-start
with the SFT lane first (`--warmstart-train`, and
`scripts/banking77_sft_dataset.py` builds the set from the container's own
prompt so there is no train/serve skew), then let CISPO improve it.

## Serving a trained adapter

The same process serves inference. Adapters are content-addressed by the SHA-256
of their directory, and a snapshot is pinned per request, so one resident base
model serves any number of adapters:

```bash
curl -X POST http://127.0.0.1:8787/v1/chat/completions -d '{
  "model": "Qwen/Qwen3.5-0.8B",
  "policy_snapshot_id": "sha256:…",
  "messages": [{"role": "user", "content": "…"}]
}'
```

See `docs/STORE.md` for the adapter store and lineage, and
`docs/LOGPROB_LIFECYCLE.md` for the sampling/training logprob contract.

## Not here yet

- **Bit-exact resume under dropout.** Weights, Adam moments and the step are
  restored together by `POST /v1/jobs/{id}/resume`, but the MLX RNG stream is
  not persisted, so a resumed run diverges if `lora_dropout > 0`. At the default
  of 0.0 it has no effect.
- **A trained-on-rollouts evaluation beyond mean reward.** The on-policy lane
  reports paired held-out mean reward before and after; it does not yet report a
  significance interval, and a run of this size could not support one.
- **More models.** A second base model arrives behind a measured admission probe
  that records peak memory and throughput on the caller's machine, never by
  relaxing the pin.
- **A Tinker bridge.** If one is built it must advertise only operations proven
  against this LoRA backend.

## License

Apache-2.0

"""Prove the product: a real LoRA SFT job end to end, then serve what it made.

This is not a smoke test and there is nothing fake in its path. It loads
Qwen3.5-0.8B on Apple Silicon, trains a real LoRA adapter through the local job
API, scores that adapter against a held-out split before and after on the same
rows, and then answers a chat request pinned to the adapter it just produced.

Every stage refuses rather than degrades:

  * admission runs before the model is spawned, and refuses on platform,
    missing MLX, or insufficient free memory;
  * the server is killed in a `finally`, and killed hard if it will not stop;
  * a job that does not reach `succeeded` fails the run, and so does a handoff
    that is not `mlx-lora.v1`, a digest that does not match the adapter tree on
    disk, or a paired evaluation that did not complete.

    python scripts/real_sft_run.py [--rows 8] [--steps 4] [--keep]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Measured on Qwen3.5-0.8B, one 768-token datum, gradient checkpointing on:
# 7.30 GB peak. Ask for headroom above it rather than exactly it.
REQUIRED_FREE_BYTES = 12 * 1024**3
BASE_MODEL = "Qwen/Qwen3.5-0.8B"


class Refused(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"refused: {message}")


def free_memory_bytes() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = 4096
    counts = {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split("bytes")[0].strip())
        if ":" in line:
            key, _, value = line.partition(":")
            value = value.strip().rstrip(".")
            if value.isdigit():
                counts[key.strip()] = int(value)
    reclaimable = sum(
        counts.get(key, 0)
        for key in ("Pages free", "Pages inactive", "Pages speculative")
    )
    return reclaimable * page


def admit() -> None:
    """Everything that can refuse before a model is spawned, refuses here."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise Refused("this run needs macOS on Apple Silicon")
    try:
        import mlx.core  # noqa: F401
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise Refused(f"MLX is not installed: {exc}") from exc
    free = free_memory_bytes()
    if free < REQUIRED_FREE_BYTES:
        raise Refused(
            f"need {REQUIRED_FREE_BYTES / 1024**3:.0f} GB free, have "
            f"{free / 1024**3:.1f} GB -- close something rather than swapping a "
            "training run"
        )
    print(f"admitted: arm64, MLX present, {free / 1024**3:.1f} GB free")


def write_dataset(path: Path, rows: list[tuple[str, str]]) -> Path:
    path.write_text(
        "".join(
            json.dumps({"prompt": prompt, "completion": completion}) + "\n"
            for prompt, completion in rows
        ),
        encoding="utf-8",
    )
    return path


def request(method: str, url: str, body: dict | None = None, timeout: float = 120.0):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def wait_for_health(base: str, process: subprocess.Popen, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise Refused(f"service exited early with code {process.returncode}")
        try:
            status, payload = request("GET", f"{base}/healthz", timeout=5)
            if status == 200:
                print(f"service up: {payload}")
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(0.5)
    raise Refused("service did not become healthy")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--root", type=Path, default=Path(".real-sft-run"))
    parser.add_argument("--keep", action="store_true", help="leave the service running")
    parser.add_argument(
        "--prove-resume",
        action="store_true",
        help="cancel mid-run and resume, asserting the run continues rather than restarting",
    )
    args = parser.parse_args()

    admit()

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"

    # A task with a rule the base model does not already follow, so a paired
    # before/after score is capable of moving at all.
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
             "india", "juliett", "kilo", "lima", "mike", "november", "oscar", "papa"]
    pairs = [(f"echo {words[i % len(words)]}-{i}", f"<{words[i % len(words)].upper()}-{i}>")
             for i in range(args.rows + 4)]
    train = write_dataset(root / "train.jsonl", pairs[: args.rows])
    heldout = write_dataset(root / "heldout.jsonl", pairs[args.rows : args.rows + 4])

    env = dict(os.environ, SYNTH_MLX_RL_MODEL=BASE_MODEL)
    process = subprocess.Popen(
        [sys.executable, "-m", "synth_mlx_rl", "serve",
         "--root", str(root / "service"), "--port", str(args.port), "--log-level", "warning"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=None,
        start_new_session=True,
    )
    try:
        wait_for_health(base, process, seconds=300)

        status, caps = request("GET", f"{base}/v1/capabilities")
        assert status == 200, caps
        qwen = caps["capabilities"]["qwen_lora_training"]
        if not qwen["supported"]:
            raise Refused(f"qwen_lora_training unsupported: {qwen['reason']}")
        assert "fixture_training" not in caps["capabilities"]
        assert "mlx_scalar_smoke" not in caps["capabilities"]
        print(f"capabilities: one backend, {caps['qwen_lora_contract']}")

        contract = caps["qwen_lora_contract"]
        payload = {
            "job_id": "real-sft",
            "config": {
                "backend": "qwen_lora",
                "base_model": BASE_MODEL,
                "dataset": {"path": str(train)},
                "evaluation_dataset": {"path": str(heldout)},
                "output_dir": str(root / "output"),
                "max_steps": args.steps,
                "batch_size": args.batch_size,
                "micro_batch_size": args.micro_batch_size,
                "shuffle": True,
                "checkpoint_every": max(1, args.steps // 2),
                "learning_rate": 1e-4,
                "lora_rank": contract["lora_rank"],
                "lora_alpha": contract["lora_alpha"],
                "max_seq_length": contract["max_seq_length"],
                "enable_thinking": contract["enable_thinking"],
                "seed": 0,
            },
        }

        status, preflight = request("POST", f"{base}/v1/jobs/preflight", payload)
        assert status == 200, preflight
        if not preflight["accepted"]:
            raise Refused(f"preflight refused: {preflight['checks']}")
        print(f"preflight accepted; dataset sha256={preflight['dataset_sha256'][:16]}…")

        status, job = request("POST", f"{base}/v1/jobs", payload)
        assert status == 201, job
        status, job = request("POST", f"{base}/v1/jobs/real-sft/launch")
        assert status == 202, job

        if args.prove_resume:
            # Cancel once a checkpoint exists, then resume. A run that restarted
            # would repeat step 1 and its loss would jump back up.
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                _, job = request("GET", f"{base}/v1/jobs/real-sft")
                if job["checkpoints"] and job["current_step"] >= 2:
                    break
                if job["status"] in {"succeeded", "failed"}:
                    raise Refused("job finished before it could be interrupted")
                time.sleep(0.5)
            request("POST", f"{base}/v1/jobs/real-sft/cancel")
            while True:
                _, job = request("GET", f"{base}/v1/jobs/real-sft")
                if job["status"] in {"cancelled", "succeeded", "failed"}:
                    break
                time.sleep(0.5)
            if job["status"] != "cancelled":
                raise Refused(f"expected cancelled, got {job['status']}")
            interrupted_at = job["current_step"]
            if not job["resume_supported"]:
                raise Refused("a job with a checkpoint must report resume_supported")
            print(f"cancelled at step {interrupted_at}; resuming")
            status, job = request("POST", f"{base}/v1/jobs/real-sft/resume")
            if status != 202:
                raise Refused(f"resume refused: {status} {job}")

        seen = 0
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            status, job = request("GET", f"{base}/v1/jobs/real-sft")
            assert status == 200, job
            _, events = request("GET", f"{base}/v1/jobs/real-sft/events?after={seen}")
            for event in events["events"]:
                seen = event["sequence"]
                if event["type"] == "training.metric":
                    metric = event["payload"]
                    print(
                        f"  step {metric['step']} (epoch {metric['epoch']})  "
                        f"loss={metric['loss']:.4f}  "
                        f"{metric['tokens']:.0f} tok in {metric['step_seconds']:.1f}s "
                        f"= {metric['tokens_per_second']:.0f} tok/s  "
                        f"peak={(metric.get('memory_bytes') or 0) / 1024**3:.2f} GB"
                    )
                elif event["type"] not in {"training.metric"}:
                    print(f"  {event['type']}")
            if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            time.sleep(1.0)
        else:
            raise Refused("training did not terminate within 30 minutes")

        if job["status"] != "succeeded":
            raise Refused(f"job {job['status']}: {job.get('error_detail')}")

        if args.prove_resume:
            _, page = request("GET", f"{base}/v1/jobs/real-sft/events?after=0")
            metrics = [e["payload"] for e in page["events"] if e["type"] == "training.metric"]
            steps = [m["step"] for m in metrics]
            if steps != sorted(set(steps)) or len(steps) != args.steps:
                raise Refused(f"resumed run did not cover each step exactly once: {steps}")
            resumed = next(m for m in metrics if m["step"] == interrupted_at + 1)
            before = next(m for m in metrics if m["step"] == interrupted_at)
            print(
                f"resume seam: step {before['step']} loss={before['loss']:.4f} -> "
                f"step {resumed['step']} loss={resumed['loss']:.4f}"
            )
            if resumed["loss"] > before["loss"]:
                raise Refused(
                    "loss rose across the resume seam -- optimizer state was not carried"
                )

        status, handoff = request("GET", f"{base}/v1/jobs/real-sft/handoff")
        assert status == 200, handoff
        if handoff["inference"]["kind"] != "mlx-lora.v1":
            raise Refused(f"handoff is not a deployable adapter: {handoff['inference']}")

        adapter = Path(handoff["checkpoint"]["path"])
        for required in ("adapter_config.json", "adapters.safetensors"):
            if not (adapter / required).is_file():
                raise Refused(f"adapter is missing {required}")
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from synth_mlx_rl.storage import sha256_path

        digest, size = sha256_path(adapter)
        if digest != handoff["checkpoint"]["sha256"]:
            raise Refused("handoff digest does not match the adapter on disk")
        print(f"adapter: {adapter}  {size / 1024**2:.1f} MB  sha256={digest[:16]}…")

        evaluation = handoff["evaluation"]
        if evaluation.get("status") != "completed":
            raise Refused(f"paired evaluation did not complete: {evaluation}")
        print(
            f"paired held-out evaluation: {evaluation['item_count']} rows, "
            f"mcnemar={evaluation.get('mcnemar')}"
        )

        # The adapter that was just trained is the live one; freeze it and answer
        # a request pinned to that exact snapshot.
        status, snapshot = request("POST", f"{base}/v1/synth/snapshots", {"metadata": {"source": "real-sft"}})
        assert status == 200, snapshot
        snapshot_id = snapshot["policy_snapshot_id"]
        status, completion = request(
            "POST",
            f"{base}/v1/chat/completions",
            {
                "model": BASE_MODEL,
                "policy_snapshot_id": snapshot_id,
                "messages": [{"role": "user", "content": pairs[0][0]}],
                "max_tokens": 16,
            },
            timeout=300,
        )
        assert status == 200, completion
        answer = completion["choices"][0]["message"]["content"]
        print(f"served from snapshot {snapshot_id[:20]}…: {answer!r}")

        print("\nREAL RUN PASSED — trained a model and served it, no stubs in the path")
        return 0
    finally:
        if args.keep:
            print(f"service left running at {base} (pid {process.pid})")
            return 0
        if process.poll() is None:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=10)
        print("service stopped")


if __name__ == "__main__":
    raise SystemExit(main())

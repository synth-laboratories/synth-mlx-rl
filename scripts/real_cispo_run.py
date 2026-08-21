"""Prove the on-policy lane: real rollouts, real reward, real update, real uplift.

Nothing in this path is simulated. A real Banking77 task container serves the
real PolyAI/banking77 dataset over the hosted `training.rollout.request.v1`
contract; it calls back into this service to sample, so the policy generating
the rollouts is the model being trained; the reward is exact-match on the intent
label, computed by the container and never by the trainer. CISPO turns grouped
rewards into advantages and applies one update per step.

Uplift is measured on a frozen held-out slice, greedily, on the same instances
before and after. It is a paired comparison of one policy against itself, which
is the only comparison a run this size can support.

    python scripts/real_cispo_run.py --steps 8 --group-size 4

Requires a Banking77 container on --container, started with
SYNTH_CONTAINERS_ALLOW_LOOPBACK_SAMPLER=1 and SYNTH_BANKING77_SOURCE=hf.
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

REQUIRED_FREE_BYTES = 12 * 1024**3
BASE_MODEL = "Qwen/Qwen3.5-0.8B"


class Refused(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"refused: {message}")


def free_memory_bytes() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page, counts = 4096, {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split("bytes")[0].strip())
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[key.strip()] = int(value)
    return page * sum(
        counts.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative")
    )


def request(method: str, url: str, body: dict | None = None, timeout: float = 300.0):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=6)
    parser.add_argument("--signal-attempts", type=int, default=24)
    parser.add_argument("--groups-per-step", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--heldout", type=int, default=16)
    parser.add_argument("--train-instances", type=int, default=64)
    parser.add_argument("--container", default="http://127.0.0.1:8114")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--root", type=Path, default=Path(".cispo-run"))
    parser.add_argument("--warmstart-train", type=Path, default=None,
                        help="SFT this dataset first, in the same process")
    parser.add_argument("--warmstart-eval", type=Path, default=None)
    parser.add_argument("--warmstart-steps", type=int, default=40)
    parser.add_argument("--warmstart-batch", type=int, default=4)
    parser.add_argument("--warmstart-lr", type=float, default=1e-4)
    args = parser.parse_args()

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise Refused("needs macOS on Apple Silicon")
    free = free_memory_bytes()
    if free < REQUIRED_FREE_BYTES:
        raise Refused(f"need 12 GB free, have {free / 1024**3:.1f} GB")
    print(f"admitted: arm64, {free / 1024**3:.1f} GB free")

    status, capabilities = request("GET", f"{args.container}/training/capabilities", timeout=15)
    if status != 200:
        raise Refused(f"no task container at {args.container}")
    print(
        f"environment: {capabilities['container_id']} task={capabilities['task_id']} "
        f"digest={capabilities['container_digest'][:23]}…"
    )

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    base = f"http://127.0.0.1:{args.port}"
    process = subprocess.Popen(
        [sys.executable, "-m", "synth_mlx_rl", "serve",
         "--root", str(root / "service"), "--port", str(args.port), "--log-level", "warning"],
        env=dict(os.environ, SYNTH_MLX_RL_MODEL=BASE_MODEL),
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise Refused(f"service exited early ({process.returncode})")
            try:
                if request("GET", f"{base}/healthz", timeout=5)[0] == 200:
                    break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            time.sleep(0.5)
        else:
            raise Refused("service never became healthy")

        _, caps = request("GET", f"{base}/v1/capabilities")
        contract = caps["qwen_lora_contract"]
        if not caps["capabilities"]["cispo_training"]["supported"]:
            raise Refused(caps["capabilities"]["cispo_training"]["reason"])

        if args.warmstart_train is not None:
            # RL cannot find a gradient in a policy that never succeeds. The
            # warm start runs in this same process, so the adapter CISPO begins
            # from is the one SFT just produced -- and CISPO's own baseline
            # evaluation therefore measures the warm-started policy, which is
            # the only honest reference for what the on-policy lane added.
            warm = {
                "job_id": "warmstart",
                "config": {
                    "backend": "qwen_lora",
                    "base_model": BASE_MODEL,
                    "dataset": {"path": str(args.warmstart_train.resolve())},
                    **(
                        {"evaluation_dataset": {"path": str(args.warmstart_eval.resolve())}}
                        if args.warmstart_eval
                        else {}
                    ),
                    "output_dir": str(root / "warmstart"),
                    "max_steps": args.warmstart_steps,
                    "batch_size": args.warmstart_batch,
                    "micro_batch_size": 1,
                    "learning_rate": args.warmstart_lr,
                    "checkpoint_every": max(1, args.warmstart_steps),
                    "lora_rank": contract["lora_rank"],
                    "lora_alpha": contract["lora_alpha"],
                    "max_seq_length": contract["max_seq_length"],
                    "enable_thinking": contract["enable_thinking"],
                    "seed": 0,
                },
            }
            status, pf = request("POST", f"{base}/v1/jobs/preflight", warm)
            if status != 200 or not pf["accepted"]:
                raise Refused(f"warm start preflight refused: {pf}")
            request("POST", f"{base}/v1/jobs", warm)
            request("POST", f"{base}/v1/jobs/warmstart/launch")
            seen_w = 0
            while True:
                status, wjob = request("GET", f"{base}/v1/jobs/warmstart")
                _, page = request("GET", f"{base}/v1/jobs/warmstart/events?after={seen_w}")
                for event in page["events"]:
                    seen_w = event["sequence"]
                    if event["type"] == "training.metric":
                        d = event["payload"]
                        if d["step"] % 5 == 0 or d["step"] == 1:
                            print(
                                f"  warmstart step {d['step']}  loss={d['loss']:.4f}  "
                                f"{d['tokens']:.0f} tok in {d['step_seconds']:.1f}s  "
                                f"peak={(d.get('memory_bytes') or 0)/1024**3:.1f} GB"
                            )
                if wjob["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                    break
                time.sleep(2.0)
            if wjob["status"] != "succeeded":
                raise Refused(f"warm start {wjob['status']}: {wjob.get('error_detail')}")
            print(f"warm start done: {wjob['evaluation'].get('status')}")

        payload = {
            "job_id": "real-cispo",
            "config": {
                "backend": "cispo",
                "base_model": BASE_MODEL,
                "output_dir": str(root / "output"),
                "rollout": {
                    "url": args.container,
                    "task_id": capabilities["task_id"],
                    "max_tokens": 24,
                    "temperature": args.temperature,
                    "train_instances": args.train_instances,
                    "heldout_instances": args.heldout,
                    "heldout_world_ref": "world:banking77@heldout",
                    "train_world_ref": "world:banking77@train",
                },
                "max_steps": args.steps,
                "group_size": args.group_size,
                "signal_attempts": args.signal_attempts,
                "groups_per_step": args.groups_per_step,
                "micro_batch_size": args.micro_batch_size,
                "objective": "cispo_minimax",
                "eps_low": 1.0,
                "eps_high": 4.0,
                "learning_rate": args.learning_rate,
                "checkpoint_every": max(1, args.steps),
                "lora_rank": contract["lora_rank"],
                "lora_alpha": contract["lora_alpha"],
                "max_seq_length": contract["max_seq_length"],
                "enable_thinking": contract["enable_thinking"],
                "seed": 0,
            },
        }

        status, preflight = request("POST", f"{base}/v1/jobs/preflight", payload)
        if status != 200 or not preflight["accepted"]:
            raise Refused(f"preflight refused: {preflight}")
        print("preflight accepted; container reachable and lane supported")

        status, job = request("POST", f"{base}/v1/jobs", payload)
        if status != 201:
            raise Refused(f"configure failed: {job}")
        request("POST", f"{base}/v1/jobs/real-cispo/launch")

        seen = 0
        deadline = time.monotonic() + 7200
        while time.monotonic() < deadline:
            status, job = request("GET", f"{base}/v1/jobs/real-cispo")
            _, page = request("GET", f"{base}/v1/jobs/real-cispo/events?after={seen}")
            for event in page["events"]:
                seen = event["sequence"]
                kind, data = event["type"], event["payload"]
                if kind == "training.metric":
                    print(
                        f"  step {data['step']}  loss={data['loss']:+.4f}  "
                        f"reward={data['reward_mean']:.3f}±{data['reward_std']:.3f}  "
                        f"adv±{data['advantage_std']:.2f}  "
                        f"clip={data.get('clip_fraction') or 0:.2f}  "
                        f"{data['tokens']:.0f} tok  "
                        f"[rollout {data['rollout_seconds']:.0f}s "
                        f"({data['rollouts_used']}/{data['rollouts_collected']} kept) "
                        f"+ update {data['update_seconds']:.0f}s "
                        f"= {data['step_seconds']:.0f}s]"
                    )
                elif kind == "rollout.group_filtered":
                    print(
                        f"    group filtered (attempt {data['signal_attempt']}): "
                        f"rewards {data['rewards']} on instance {data.get('instance')}"
                    )
                elif kind == "heldout_eval.completed":
                    print(
                        f"  held-out {data['phase']}: mean reward "
                        f"{data['mean_reward']:.4f} over {data['instances']} instances"
                    )
                elif kind in {"evaluation.completed", "job.failed", "job.succeeded",
                              "rollout.container_admitted"}:
                    print(f"  {kind} {json.dumps(data)[:200]}")
            if job["status"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                break
            time.sleep(2.0)
        else:
            raise Refused("run did not terminate")

        if job["status"] != "succeeded":
            raise Refused(f"job {job['status']}: {job.get('error_detail')}")

        evaluation = job["evaluation"]
        uplift = evaluation["uplift"]
        print(
            f"\nUPLIFT {uplift:+.4f}   "
            f"before={evaluation['mean_reward_before']:.4f} "
            f"after={evaluation['mean_reward_after']:.4f}   "
            f"improved={evaluation['instances_improved']} "
            f"regressed={evaluation['instances_regressed']}"
        )
        return 0 if uplift > 0 else 2
    finally:
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

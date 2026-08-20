"""GSM8K training lab: baseline -> SFT or CISPO -> held-out eval, on real MLX.

Reward and data come from the container's own world module, so the training
loop and the container agree by construction rather than by convention.

The loop publishes a snapshot after every optimizer step. `resolve(None)`
returns the newest PUBLISHED snapshot and `optim_step` publishes nothing, so a
loop that skips the refresh samples stale weights forever and training looks
like a no-op.
"""
from __future__ import annotations

import argparse, json, os, random, sys, time
from pathlib import Path

sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import mlx_guard

CONTAINERS = Path.home() / "Documents/GitHub/containers-mlx-local-rl-20260818/src"
sys.path.insert(0, str(CONTAINERS))
os.environ.setdefault("SYNTH_GSM8K_SOURCE", "hf")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from synth_containers.platform.gsm8k_world import (  # noqa: E402
    HELDOUT_SPLIT, SOLVE_SYSTEM, TRAIN_SPLIT, load_row, parse_answer, user_prompt,
)
from synth_mlx_rl.config import Settings  # noqa: E402
from synth_mlx_rl.engine import MLXEngine  # noqa: E402
from synth_mlx_rl.mismatch import MismatchPolicy, measure_mismatch  # noqa: E402
from synth_mlx_rl.rewards import group_normalize_rewards  # noqa: E402
from synth_mlx_rl.schemas import (  # noqa: E402
    AdamParams, Datum, ForwardBackwardRequest, SampleRequest,
)
from synth_mlx_rl.store import AdapterStore, StoreConfig  # noqa: E402


def messages(question: str) -> list[dict]:
    return [{"role": "system", "content": SOLVE_SYSTEM},
            {"role": "user", "content": user_prompt(question)}]


def sample(engine, question, *, n, temperature, max_tokens, snapshot):
    return engine.sample(SampleRequest(
        messages=messages(question), num_samples=n, max_tokens=max_tokens,
        temperature=temperature, top_p=1.0, top_k=0, min_p=0.0,
        policy_snapshot_id=snapshot))


def reward(text: str, gold: str) -> float:
    parsed = parse_answer(text)
    # An unparseable completion is a failed attempt, not an absent signal.
    return 1.0 if parsed.parsed and parsed.value == gold else 0.0


def evaluate(engine, seeds, *, max_tokens, snapshot, label) -> float:
    hits = 0
    t0 = time.time()
    for seed in seeds:
        row = load_row(HELDOUT_SPLIT, seed)
        out = sample(engine, row.question, n=1, temperature=0.0,
                     max_tokens=max_tokens, snapshot=snapshot)
        hits += reward(out.samples[0].text, row.answer)
    accuracy = hits / len(seeds)
    print(f"  [{label}] held-out {hits}/{len(seeds)} = {accuracy:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    return accuracy


def completion_datum(prompt_ids, completion_ids, *, behavior=None, advantage=None) -> Datum:
    ids = list(prompt_ids) + list(completion_ids)
    n = len(completion_ids)
    pad = len(ids) - 1 - n
    kwargs = {}
    if behavior is not None:
        kwargs["behavior_logprobs"] = [0.0] * pad + list(behavior)
    if advantage is not None:
        kwargs["advantages"] = [float(advantage)] * (len(ids) - 1)
    return Datum(input_ids=ids[:-1], target_ids=ids[1:],
                 weights=[0.0] * pad + [1.0] * n, **kwargs)


def archive(engine, *, run_id, step, mode, baseline, final, store_root) -> str | None:
    """Checkpoint the adapter into the content-addressed store with provenance.

    The digest is the identity on all three surfaces (D10), so an adapter that
    only ever lives in the engine's memory cannot later be told apart from any
    other, nor pointed at by an eval candidate.
    """
    try:
        checkpoint = engine.save_checkpoint(run_id)
    except Exception as exc:  # a failed archive must not void a measured run
        print(f"  [store] checkpoint failed: {type(exc).__name__}: {exc}")
        return None
    store = AdapterStore(StoreConfig(root=Path(store_root).expanduser()))
    record = store.put_adapter(
        Path(checkpoint.path), run_id=run_id, step=step, algorithm=mode,
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        base_model=engine.settings.model)
    run = store.open_run(run_id, {
        "base_model": engine.settings.model, "algorithm": mode,
        "lora_rank": engine.settings.lora_rank, "dataset": "openai/gsm8k",
        "baseline_accuracy": baseline, "final_accuracy": final})
    run.checkpoint(step=step, adapter_digest=record.digest,
                   metrics={"heldout_accuracy": final, "baseline_accuracy": baseline})
    run.event("trained", {"mode": mode, "steps": step})
    print(f"  [store] {record.digest[:24]}..  run={run_id}")
    return record.digest


def refresh(engine) -> str:
    """Publish the trained weights. Without this the next sample is stale."""
    return engine.publish_snapshot().id


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sft", "cispo"])
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--train-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=1)
    # SFT and RL want rates an order of magnitude apart: SFT fits a fixed set of
    # accepted traces, while a policy-gradient step at 1e-4 on rank-8 LoRA moves
    # the sampler far enough that the next round collects from a different model
    # than the one the advantages were computed for.
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--accum", type=int, default=4,
                    help="datums accumulated per optimizer step (1 datum per "
                         "forward_backward call; the engine normalizes globally)")
    ap.add_argument("--max-traces", type=int, default=0,
                    help="cap accepted traces (0 = all)")
    ap.add_argument("--cache", default="runs/traces.jsonl",
                    help="collected SFT traces, reused instead of re-sampled")
    ap.add_argument("--rounds", type=int, default=8, help="cispo collection rounds")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.lr is None:
        args.lr = 1e-4 if args.mode == "sft" else 1e-5

    mlx_guard.install(3.0)
    engine = MLXEngine(Settings(lora_rank=8, max_seq_length=1600))
    base_snapshot = refresh(engine)
    eval_seeds = list(range(args.eval_n))

    print(f"GSM8K {args.mode}  model={engine.settings.model}  lora r={engine.settings.lora_rank}")
    print(f"eval seeds 0..{args.eval_n-1} (fixed, shared by every arm)\n")
    baseline = evaluate(engine, eval_seeds, max_tokens=args.max_tokens,
                        snapshot=base_snapshot, label="baseline")

    rng = random.Random(20260818)
    train_seeds = rng.sample(range(1000), args.train_prompts)
    history = []

    if args.mode == "sft":
        cache_path = Path(args.cache) if args.cache else None
        cached = []
        if cache_path and cache_path.exists():
            for line in cache_path.read_text().splitlines():
                if line.strip():
                    cached.append(json.loads(line))
        if cached:
            # Collection is the expensive half (24 minutes for 96 prompts). A
            # crash in the training half must not cost it twice.
            print(f"\nreusing {len(cached)} cached traces from {cache_path}")
            accepted = [completion_datum(row["prompt_ids"], row["completion_ids"])
                        for row in cached]
            seen = cached[0].get("seen_total", len(cached))
        else:
            accepted, seen = None, 0

    if args.mode == "sft" and accepted is None:
        print(f"\ncollecting: {args.train_prompts} train prompts x {args.group} samples, temp=0.8")
        accepted, seen, t0 = [], 0, time.time()
        rows = []
        for i, seed in enumerate(train_seeds):
            row = load_row(TRAIN_SPLIT, seed)
            out = sample(engine, row.question, n=args.group, temperature=0.8,
                         max_tokens=args.max_tokens, snapshot=base_snapshot)
            for s in out.samples:
                seen += 1
                if reward(s.text, row.answer) == 1.0:
                    accepted.append(completion_datum(s.prompt_token_ids, s.completion_token_ids))
                    rows.append({"prompt_ids": list(s.prompt_token_ids),
                                 "completion_ids": list(s.completion_token_ids)})
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{len(train_seeds)} prompts, {len(accepted)}/{seen} accepted "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  accepted {len(accepted)}/{seen} traces "
              f"({len(accepted)/max(seen,1):.1%})", flush=True)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            for entry in rows:
                entry["seen_total"] = seen
            cache_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            print(f"  cached -> {cache_path}", flush=True)
        if not accepted:
            print("no accepted traces; nothing to train on")
            return 1

        # Length-BUCKETED, then the buckets shuffled. Sorting alone keeps each
        # step's memory peak predictable, but it also replaced the shuffle it
        # was written over: the model then sees every short trace first and
        # every long one last, an unintended curriculum whose final updates are
        # dominated by long examples. Bucketing keeps the padding efficiency;
        # shuffling the buckets removes the ordering.
        accepted.sort(key=lambda d: len(d.input_ids))
        buckets = [accepted[i:i + args.accum] for i in range(0, len(accepted), args.accum)]
        rng.shuffle(buckets)
        accepted = [datum for bucket in buckets for datum in bucket]
        if args.max_traces:
            accepted = accepted[:args.max_traces]
            print(f"  capped to {len(accepted)} traces")
        step = 0
        train_tokens = 0
        train_seconds = 0.0
        for epoch in range(args.epochs):
            for start in range(0, len(accepted), args.accum):
                batch = accepted[start:start + args.accum]
                if not batch:
                    continue
                # ONE datum per forward_backward, accumulated. A single call
                # with N long sequences materializes [N, T, 151936] fp32 logits
                # plus their gradient, which took free memory to 7% and got the
                # process killed by the watchdog. The engine scales each call's
                # gradient by its token count and divides by the total at
                # optim_step, so accumulating N singles is the same gradient at
                # a fraction of the peak. Plan section 14.
                t_step = time.time()
                step_tokens = 0
                for datum in batch:
                    fb = engine.forward_backward(ForwardBackwardRequest(
                        data=[datum], loss_fn="cross_entropy"))
                    step_tokens += int(fb.metrics.get("token_count", 0))
                engine.optim_step(AdamParams(learning_rate=args.lr, max_grad_norm=1.0))
                step += 1
                train_tokens += step_tokens
                train_seconds += time.time() - t_step
                if step % 5 == 0:
                    print(f"  step {step:3d}  loss {fb.loss:.4f}  "
                          f"{step_tokens/max(time.time()-t_step,1e-9):.0f} train tok/s", flush=True)
        snapshot = refresh(engine)
        final = evaluate(engine, eval_seeds, max_tokens=args.max_tokens,
                         snapshot=snapshot, label="after sft")
        tps = train_tokens / max(train_seconds, 1e-9)
        print(f"  training: {step} steps, {train_tokens} weighted tokens, "
              f"{train_seconds:.0f}s -> {tps:.0f} train tok/s", flush=True)
        history.append({"stage": "sft", "steps": step, "accepted": len(accepted),
                        "train_tokens": train_tokens, "train_seconds": train_seconds,
                        "train_tokens_per_s": tps})

    else:  # cispo
        policy = MismatchPolicy()
        snapshot = base_snapshot
        step = 0
        for round_index in range(args.rounds):
            prompts = train_seeds[round_index * 8:(round_index + 1) * 8] or train_seeds[:8]
            datums, rewards_seen, refused = [], [], 0
            for seed in prompts:
                row = load_row(TRAIN_SPLIT, seed)
                out = sample(engine, row.question, n=args.group, temperature=1.0,
                             max_tokens=args.max_tokens, snapshot=snapshot)
                rewards = [reward(s.text, row.answer) for s in out.samples]
                advantages = group_normalize_rewards(rewards)
                rewards_seen.extend(rewards)
                for s, adv in zip(out.samples, advantages):
                    if adv == 0.0:
                        continue  # zero-variance group carries no signal
                    ids = list(s.prompt_token_ids) + list(s.completion_token_ids)
                    scored = engine.score_logprobs(ids, policy_snapshot_id=snapshot)
                    n = len(s.completion_token_ids)
                    behavior = [0.0 if v is None else float(v) for v in scored[len(ids) - n:]]
                    report = policy.evaluate(measure_mismatch(
                        behavior_logprobs=behavior, rollout_logprobs=s.rollout_logprobs))
                    if report.verdict == "refuse":
                        refused += 1
                        continue
                    datums.append(completion_datum(
                        s.prompt_token_ids, s.completion_token_ids,
                        behavior=behavior, advantage=float(adv)))
            mean_reward = sum(rewards_seen) / max(len(rewards_seen), 1)
            if datums:
                for start in range(0, len(datums), args.accum):
                    batch = datums[start:start + args.accum]
                    for datum in batch:  # accumulate singles; see the SFT note
                        engine.forward_backward(ForwardBackwardRequest(
                            data=[datum], loss_fn="cispo_minimax",
                            eps_low=1.0, eps_high=4.0))
                    engine.optim_step(AdamParams(learning_rate=args.lr, max_grad_norm=1.0))
                    step += 1
            snapshot = refresh(engine)
            print(f"  round {round_index}: train_reward={mean_reward:.3f} "
                  f"datums={len(datums)} refused={refused} steps={step}", flush=True)
            history.append({"round": round_index, "train_reward": mean_reward,
                            "datums": len(datums), "refused": refused})
        final = evaluate(engine, eval_seeds, max_tokens=args.max_tokens,
                         snapshot=snapshot, label="after cispo")

    digest = archive(engine, run_id=f"gsm8k-{args.mode}-{int(time.time())}",
                     step=history[-1].get("steps", len(history)) if history else 0,
                     mode=args.mode, baseline=baseline, final=final,
                     store_root=os.environ.get("SYNTH_MLX_RL_STORE", "~/.synth/mlx-rl"))

    uplift = final - baseline
    print(f"\n{'='*58}")
    print(f"  baseline   {baseline:.4f}")
    print(f"  final      {final:.4f}")
    print(f"  uplift     {uplift:+.4f}  ({uplift/max(baseline,1e-9):+.1%} relative)")
    print(f"{'='*58}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"mode": args.mode, "baseline": baseline, "final": final, "uplift": uplift,
             "eval_n": args.eval_n, "adapter_digest": digest, "history": history,
             "args": vars(args)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

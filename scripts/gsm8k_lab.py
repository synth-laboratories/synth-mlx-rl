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
from synth_mlx_rl.objectives import group_normalize_rewards  # noqa: E402
from synth_mlx_rl.schemas import (  # noqa: E402
    AdamParams, Datum, ForwardBackwardRequest, SampleRequest,
)


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
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--micro", type=int, default=4, help="datums per optimizer step")
    ap.add_argument("--rounds", type=int, default=8, help="cispo collection rounds")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

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
        print(f"\ncollecting: {args.train_prompts} train prompts x {args.group} samples, temp=0.8")
        accepted, seen, t0 = [], 0, time.time()
        for i, seed in enumerate(train_seeds):
            row = load_row(TRAIN_SPLIT, seed)
            out = sample(engine, row.question, n=args.group, temperature=0.8,
                         max_tokens=args.max_tokens, snapshot=base_snapshot)
            for s in out.samples:
                seen += 1
                if reward(s.text, row.answer) == 1.0:
                    accepted.append(completion_datum(s.prompt_token_ids, s.completion_token_ids))
            if (i + 1) % 16 == 0:
                print(f"  {i+1}/{len(train_seeds)} prompts, {len(accepted)}/{seen} accepted "
                      f"({time.time()-t0:.0f}s)", flush=True)
        print(f"  accepted {len(accepted)}/{seen} traces ({len(accepted)/max(seen,1):.1%})")
        if not accepted:
            print("no accepted traces; nothing to train on")
            return 1

        step = 0
        for epoch in range(args.epochs):
            rng.shuffle(accepted)
            for start in range(0, len(accepted), args.micro):
                batch = accepted[start:start + args.micro]
                if not batch:
                    continue
                fb = engine.forward_backward(ForwardBackwardRequest(
                    data=batch, loss_fn="cross_entropy"))
                engine.optim_step(AdamParams(learning_rate=args.lr, max_grad_norm=1.0))
                step += 1
                if step % 5 == 0:
                    print(f"  step {step:3d}  loss {fb.loss:.4f}", flush=True)
        snapshot = refresh(engine)
        final = evaluate(engine, eval_seeds, max_tokens=args.max_tokens,
                         snapshot=snapshot, label="after sft")
        history.append({"stage": "sft", "steps": step, "accepted": len(accepted)})

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
                for start in range(0, len(datums), args.micro):
                    batch = datums[start:start + args.micro]
                    engine.forward_backward(ForwardBackwardRequest(
                        data=batch, loss_fn="cispo_minimax", eps_low=1.0, eps_high=4.0))
                    engine.optim_step(AdamParams(learning_rate=args.lr, max_grad_norm=1.0))
                    step += 1
            snapshot = refresh(engine)
            print(f"  round {round_index}: train_reward={mean_reward:.3f} "
                  f"datums={len(datums)} refused={refused} steps={step}", flush=True)
            history.append({"round": round_index, "train_reward": mean_reward,
                            "datums": len(datums), "refused": refused})
        final = evaluate(engine, eval_seeds, max_tokens=args.max_tokens,
                         snapshot=snapshot, label="after cispo")

    uplift = final - baseline
    print(f"\n{'='*58}")
    print(f"  baseline   {baseline:.4f}")
    print(f"  final      {final:.4f}")
    print(f"  uplift     {uplift:+.4f}  ({uplift/max(baseline,1e-9):+.1%} relative)")
    print(f"{'='*58}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"mode": args.mode, "baseline": baseline, "final": final, "uplift": uplift,
             "eval_n": args.eval_n, "history": history, "args": vars(args)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

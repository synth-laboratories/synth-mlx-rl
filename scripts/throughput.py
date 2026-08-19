"""Sampling throughput, and whether the batched path is faithful.

Correctness before speed: at temperature 0 the batched decode must produce
exactly what the sequential path produces -- same tokens, same log-probabilities
-- or the speedup is bought with a different policy than the one being measured.
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import mlx_guard

from synth_mlx_rl.config import Settings
from synth_mlx_rl.engine import MLXEngine
from synth_mlx_rl.schemas import SampleRequest

PROMPT = [{"role": "user", "content":
           "A shop sells pens for $3 and books for $12. Ann buys 4 pens and 2 books. "
           "Show your reasoning, then give the total in dollars."}]


def request(n, *, temp, max_tokens=192):
    return SampleRequest(messages=PROMPT, num_samples=n, max_tokens=max_tokens,
                         temperature=temp, top_p=1.0, top_k=0, min_p=0.0)


def main() -> int:
    mlx_guard.install(3.0)
    engine = MLXEngine(Settings(lora_rank=8, max_seq_length=1024))
    engine.publish_snapshot()

    # warm up kernels so the first timing is not a compile measurement
    engine.sample(request(1, temp=0.0, max_tokens=8))

    print("=== faithfulness: greedy batched must equal greedy sequential ===")
    seq = engine._generate(
        engine._render_prompt(request(1, temp=0.0)).token_ids, request(1, temp=0.0), 0)
    bat = engine.generate_batch(
        engine._render_prompt(request(2, temp=0.0)).token_ids, request(2, temp=0.0))
    same_tokens = seq[0] == bat[0][0]
    max_lp_delta = max(
        (abs(a - b) for a, b in zip(seq[1], bat[0][1])), default=0.0)
    print(f"  tokens identical : {same_tokens}  ({len(seq[0])} vs {len(bat[0][0])})")
    print(f"  max |logprob diff|: {max_lp_delta:.3e}")
    # every row of a greedy batch must also agree with every other row
    rows_agree = all(row[0] == bat[0][0] for row in bat)
    print(f"  rows agree       : {rows_agree}")

    print("\n=== throughput ===")
    print(f"{'batch':>6s} {'wall s':>8s} {'tokens':>8s} {'tok/s':>9s} {'vs B=1':>8s}")
    base = None
    for n in (1, 2, 4, 8, 16, 32):
        t0 = time.time()
        out = engine.sample(request(n, temp=1.0))
        elapsed = time.time() - t0
        tokens = sum(len(s.completion_token_ids) for s in out.samples)
        tps = tokens / elapsed
        base = base or tps
        print(f"{n:6d} {elapsed:8.2f} {tokens:8d} {tps:9.1f} {tps/base:7.2f}x", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

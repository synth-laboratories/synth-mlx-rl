"""Why is training ~50x slower than the FLOPs say it should be?

Separates the two candidate causes:
  A. shape-driven recompilation -- MLX compiles per shape, and every trace has a
     different length, so each step may rebuild the graph.
  B. the full-vocabulary fp32 logits path -- [1, T, 151936] plus its gradient.

Fixed-shape repeats isolate A: if the first call is slow and the rest are fast,
it is compilation. If every call is slow, it is the logits path and chunked
cross-entropy is the fix.
"""
from __future__ import annotations

import sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import mlx_guard

from synth_mlx_rl.config import Settings
from synth_mlx_rl.engine import MLXEngine
from synth_mlx_rl.schemas import AdamParams, Datum, ForwardBackwardRequest


def datum(length: int) -> Datum:
    ids = [(i % 900) + 100 for i in range(length + 1)]
    completion = length // 2
    return Datum(input_ids=ids[:-1], target_ids=ids[1:],
                 weights=[0.0] * (length - completion) + [1.0] * completion)


def timed(engine, d: Datum) -> tuple[float, int]:
    t0 = time.time()
    fb = engine.forward_backward(ForwardBackwardRequest(data=[d], loss_fn="cross_entropy"))
    return time.time() - t0, int(fb.metrics.get("token_count", 0))


def main() -> int:
    mlx_guard.install(4.0)
    engine = MLXEngine(Settings(lora_rank=8, max_seq_length=2048))
    print(f"model {engine.settings.model}")
    print(f"  top-level modules: {[k for k, _ in engine.model.children().items()][:8]}\n")

    print("=== A. same shape, repeated (isolates compilation) ===")
    fixed = datum(640)
    for i in range(6):
        dt, tok = timed(engine, fixed)
        print(f"  call {i}: {dt*1000:8.1f} ms   {tok/dt:8.1f} tok/s")
    engine.zero_grad()

    print("\n=== B. every call a different shape (the real training loop) ===")
    for length in (600, 617, 634, 651, 668, 685):
        dt, tok = timed(engine, datum(length))
        print(f"  len {length}: {dt*1000:8.1f} ms   {tok/dt:8.1f} tok/s")
    engine.zero_grad()

    print("\n=== C. scaling with length, same-shape warm ===")
    for length in (128, 256, 512, 1024):
        d = datum(length)
        timed(engine, d)              # warm this shape
        dt, tok = timed(engine, d)
        print(f"  len {length:5d}: {dt*1000:8.1f} ms   {tok/dt:8.1f} tok/s")
    engine.zero_grad()

    print("\n=== D. cost split: forward-only vs forward+backward ===")
    ids = [(i % 900) + 100 for i in range(641)]
    import mlx.core as mx
    t0 = time.time(); engine.score_logprobs(ids); mx.synchronize()
    fwd = time.time() - t0
    dt, _ = timed(engine, datum(640))
    print(f"  forward only (score_logprobs): {fwd*1000:8.1f} ms")
    print(f"  forward + backward           : {dt*1000:8.1f} ms   ratio {dt/max(fwd,1e-9):.1f}x")
    engine.zero_grad()

    print(f"\nreference: 0.758B params over 640 tokens is ~2.9 TFLOP fwd+bwd;")
    print(f"a 0.2s step would be ~3200 tok/s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

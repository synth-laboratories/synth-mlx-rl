"""Every objective, one optimizer step each, on the real MLX engine.

Also answers the question the fake engine structurally cannot: on real Metal,
does the sampler's rollout distribution agree with the trainer's forward pass?
"""
import sys, time
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import mlx_guard; mlx_guard.install(3.0)

import mlx.core as mx
from synth_mlx_rl.config import Settings
from synth_mlx_rl.engine import MLXEngine
from synth_mlx_rl.mismatch import MismatchPolicy, measure_mismatch
from synth_mlx_rl.objective_spec import ObjectiveError
from synth_mlx_rl.schemas import AdamParams, Datum, ForwardBackwardRequest, SampleRequest

engine = MLXEngine(Settings(lora_rank=8, max_seq_length=512))
print(f"model {engine.settings.model}  lora r={engine.settings.lora_rank}\n")

# top_p=1, top_k=0, min_p=0: rollout logprobs are the true behavior distribution
# only when sampling is untruncated.
rollout = engine.sample(SampleRequest(
    messages=[{"role": "user", "content": "Give the integer answer only: 6 * 7"}],
    num_samples=1, max_tokens=12, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0))
s0 = rollout.samples[0]
n = len(s0.completion_token_ids)
print(f"sampled {n} tokens: {s0.text!r}")

# THE measurement the fake engine cannot make: recompute behavior_logprobs
# through the training forward path and compare against what the sampler emitted.
ids = list(s0.prompt_token_ids) + list(s0.completion_token_ids)
scored = engine.score_logprobs(ids)
behavior = [0.0 if v is None else float(v) for v in scored[len(ids) - n:]]
report = MismatchPolicy().evaluate(
    measure_mismatch(behavior_logprobs=behavior, rollout_logprobs=s0.rollout_logprobs))
print(f"\nREAL sampler-vs-trainer mismatch on Metal:")
print(f"  max |behavior - rollout| = {report.max_abs_diff:.3e}")
print(f"  mean                     = {report.mean_abs_diff:.3e}")
print(f"  ess_ratio                = {report.ess_ratio:.6f}")
print(f"  verdict                  = {report.verdict}")

def datum(advantage: float) -> Datum:
    pad = len(ids) - 1 - n
    return Datum(
        input_ids=ids[:-1], target_ids=ids[1:],
        weights=[0.0] * pad + [1.0] * n,
        behavior_logprobs=[0.0] * pad + behavior,   # the ratio DENOMINATOR
        advantages=[advantage] * (len(ids) - 1),
    )

ALGOS = ["cross_entropy", "importance_sampling", "grpo", "cispo_minimax", "cispo_two_sided"]
print(f"\n{'objective':22s} {'loss':>12s} {'grad_norm':>11s} {'ver':>4s} {'peak GB':>8s} {'s':>6s}")
print("-" * 70)
for name in ALGOS:
    kwargs = {"loss_fn": name, "data": [datum(1.0)]}
    if name.startswith("cispo"):
        kwargs["eps_low"] = 1.0 if name == "cispo_minimax" else 0.2
        kwargs["eps_high"] = 4.0
    t0 = time.time()
    fb = engine.forward_backward(ForwardBackwardRequest(**kwargs))
    step = engine.optim_step(AdamParams(learning_rate=1e-5, max_grad_norm=1.0))
    print(f"{name:22s} {fb.loss:12.6f} {step.grad_norm:11.5f} {step.training_version:4d} "
          f"{mx.get_peak_memory()/1024**3:8.2f} {time.time()-t0:6.2f}")

print()
for label, kwargs in (
    ("ppo (no value head)", {"loss_fn": "ppo"}),
    ("cispo_minimax eps_low=0.2", {"loss_fn": "cispo_minimax", "eps_low": 0.2, "eps_high": 4.0}),
):
    try:
        engine.forward_backward(ForwardBackwardRequest(data=[datum(1.0)], **kwargs))
        print(f"{label:28s} UNEXPECTEDLY ACCEPTED")
    except (ObjectiveError, ValueError) as exc:
        # pydantic wraps the message; the real reason is the line after the
        # field name, not the trailing docs URL.
        lines = [l.strip() for l in str(exc).splitlines() if l.strip()]
        reason = next((l for l in lines if "Value error," in l), lines[-1])
        print(f"{label:28s} refused: {reason.replace('Value error, ', '')[:76]}")

print(f"\nall objectives stepped on Metal; final training_version = {engine.state().training_version}")

# The logprob lifecycle

Four populations exist. Only two belong in a ratio together, and confusing them
is the failure mode this whole subsystem exists to prevent.

```text
  name                produced by                 when         grad?  role
  ---------------------------------------------------------------------------------
  rollout_logprobs    sampler, raw distribution   generation    no    mismatch input
                      BEFORE top-p / top-k / min-p
  behavior_logprobs   trainer fwd @ snapshot v    after collect no    ratio DENOMINATOR
  current_logprobs    trainer fwd @ v+k           each fb pass  YES   ratio NUMERATOR
  reference_logprobs  frozen reference (optional) once          no    k3 KL penalty
```

Two ratios, never multiplied into one:

```text
  policy update ratio     exp(current  - behavior)   -> clipping   (kernel.py)
  sampling mismatch ratio exp(behavior - rollout)    -> TIS        (mismatch.py)
```

They answer different questions. The first asks how far the policy has moved
since collection. The second asks whether the sampler and the trainer even agree
about what the *same* weights predict for the *same* tokens — a disagreement
caused by kernel differences, dtype, batching, or a sampler that quietly served
a different snapshot. Folding them together hides the second inside the first,
and the second is the one that says the run is invalid rather than merely
off-policy.

## The stages

```text
  1. sample            POST /v1/chat/completions | /v1/responses
                       -> proxy_request_id, and a server-side record holding
                          token ids + rollout_logprobs (the training authority)

  2. collect           GET  /v1/synth/rollouts/{proxy_request_id}
                       the container never carries logprobs; it carries the id

  3. recompute         POST /v1/synth/mismatch
     + measure         scores behavior_logprobs under the snapshot the record
     + verdict         NAMES, aligns them to the completion tokens, measures the
                       disagreement, and returns one of three verdicts

  4. correct or refuse ok               -> train as collected
                       correct_with_tis -> apply the returned tis_weights
                       refuse           -> do not train on this collection
```

Step 3 is one endpoint on purpose. The steps are only meaningful together:
behavior logprobs scored under a *different* snapshot than the record names
measure nothing, and a caller assembling the sequence by hand is one slice away
from a confident number about the wrong comparison.

## What is reported

Every call returns the full picture whatever the verdict, because a refused run
is still evidence:

```text
  train_rollout_logprob_abs_diff        max |behavior - rollout|
  train_rollout_logprob_abs_diff_mean
  tis_ratio_mean / _p50 / _p95 / _max   pre-clamp, always
  ess_ratio                             (sum w)^2 / (sum w^2 * n)
  clamped_token_count                   log-ratios that hit the +/-20 clamp
  nonfinite_token_count
  verdict, reason
```

## Refusal is the default past a bound

`MismatchPolicy` defaults are strict: `ok_abs_diff=1e-3`, `max_abs_diff=0.5`,
`min_ess_ratio=0.5`, and any nonfinite compared token disqualifies. On a
single-process service sampling and training the same resident weights,
agreement should be near-exact; a large disagreement is structural — a different
snapshot served, tokenizer drift, an alignment bug — not noise to reweight.
TIS-weighting garbage produces a confident update in an arbitrary direction.

Three refusals are deliberate and worth keeping:

- **Misaligned arrays are refused, not truncated.** Silently comparing the
  shorter of the two is how an off-by-one becomes a plausible-looking metric.
- **An all-masked comparison raises.** It is a failed collection, not a zero
  mismatch.
- **A refused batch hands back no TIS weights**, so a caller cannot apply them
  by accident.

## TIS is a separate stage

`policy_terms(..., tis_weights=...)` multiplies the weights into the per-token
surrogate *after* the objective has formed it, and never inside its clip. The
clip fraction stays a property of the policy ratio alone. A test asserts both:
that TIS scales the loss, and that it leaves `clip_fraction` untouched.

Metrics report the **pre-clamp** ratio. In the module's own test the post-clamp
mean is exactly 1.0 — which would read as perfect agreement — while the
pre-clamp mean is 0.928 and does not.

## What the fake engine cannot tell you

`scripts/lifecycle_check.py` runs all of this against the fake engine and
reports `max |behavior - rollout| = 0.0`. That is expected and it is not
evidence: the fake's sampler and trainer are the same deterministic code path,
so they cannot disagree. Whether the real MLX sampling path and the training
forward pass agree is exactly what this metric exists to discover, and it can
only be answered on Apple Silicon.

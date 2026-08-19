"""The objective math the learner actually executes.

Written once, against a small array-operation namespace, so that the expression
tree evaluated by MLX on Apple Silicon is byte-for-byte the same source that the
portable test suite exercises with a NumPy backend and with a forward-mode
autodiff backend. MLX is never imported here.

``objectives.py`` holds a second, independently written pure-NumPy
implementation. It is the parity oracle: if the two disagree, one of them is
wrong, and the test suite says so.
"""

from __future__ import annotations

from typing import Any, Protocol

from .objective_spec import LOG_RATIO_CLAMP, ObjectiveSpec


class ArrayOps(Protocol):
    """The array operations an objective needs.

    MLX, NumPy, and the forward-mode autodiff backend all satisfy this.
    """

    def scalar(self, value: float) -> Any: ...

    def exp(self, x: Any) -> Any: ...

    def clip(self, x: Any, low: float, high: float) -> Any: ...

    def minimum(self, a: Any, b: Any) -> Any: ...

    def maximum(self, a: Any, b: Any) -> Any: ...

    def abs(self, x: Any) -> Any: ...

    def square(self, x: Any) -> Any: ...

    def sum(self, x: Any) -> Any: ...

    def where(self, condition: Any, a: Any, b: Any) -> Any: ...

    def stop_gradient(self, x: Any) -> Any: ...

    def isfinite(self, x: Any) -> Any:
        """Return a 0/1 float mask, not a boolean array."""
        ...

    def greater(self, a: Any, b: Any) -> Any:
        """Return a 0/1 float mask, not a boolean array."""
        ...

    def float32(self, x: Any) -> Any: ...


def _sanitize(
    ops: ArrayOps, current: Any, behavior: Any, weights: Any
) -> tuple[Any, Any, Any, Any, Any]:
    """Return ``(log_ratio, current, weights, nonfinite_tokens, clamped_tokens)``.

    A position whose log-ratio or current log-probability is non-finite is
    zeroed *and* dropped from the loss weights, so a single poisoned token
    cannot decide a step. Zeroing the weight is not enough on its own: CISPO
    multiplies by ``current_logprobs`` directly, and ``0 * nan`` is ``nan``, so
    the value has to be replaced as well as masked.

    Clamping to +/-20 happens in float32 before ``exp``. Both counts are
    reported, so a run that is quietly leaning on the clamp is visible in its
    own metrics rather than only in its results.
    """

    current_f32 = ops.float32(current)
    raw = current_f32 - ops.float32(behavior)
    finite = ops.isfinite(raw) * ops.isfinite(current_f32)
    nonfinite_tokens = ops.sum((ops.scalar(1.0) - finite) * weights)
    safe = ops.where(finite, raw, ops.scalar(0.0))
    safe_current = ops.where(finite, current, ops.scalar(0.0))
    weights = weights * finite

    clamped = ops.clip(safe, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    was_clamped = ops.greater(ops.abs(safe), ops.scalar(LOG_RATIO_CLAMP))
    clamped_tokens = ops.sum(was_clamped * weights)
    return clamped, safe_current, weights, nonfinite_tokens, clamped_tokens


def sft_terms(
    ops: ArrayOps,
    current_logprobs: Any,
    weights: Any,
    reduction: str = "mean_tokens",
) -> tuple[Any, dict[str, Any]]:
    """Masked cross-entropy.

    ``weights`` is not clamped to 1: fractional weights keep their intended
    scale, and validation already guarantees a strictly positive sum.

    ``reduction`` decides step size and the two conventions differ by a factor
    of the token count. ``mean_tokens`` divides by the unmasked count, so a step
    is the same size whatever the sequence length. ``sum`` does not, which is
    the Tinker convention; it is offered so a ported script keeps its tuned
    learning rate instead of silently training hundreds of times harder.
    """

    token_count = ops.sum(weights)
    total_logprob = ops.sum(current_logprobs * weights)
    mean_logprob = total_logprob / token_count
    loss = -total_logprob if reduction == "sum" else -mean_logprob
    return loss, {
        "loss": loss,
        "token_count": token_count,
        "mean_target_logprob": mean_logprob,
    }


def policy_terms(
    ops: ArrayOps,
    spec: ObjectiveSpec,
    *,
    current_logprobs: Any,
    behavior_logprobs: Any,
    advantages: Any,
    weights: Any,
    reference_logprobs: Any | None = None,
    reference_mask: Any | None = None,
    entropy: Any | None = None,
    tis_weights: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Every policy objective in one expression tree.

    The three families differ only in how the ratio enters the surrogate:

    ``importance_sampling``  ``ratio * A``                     (unclipped, honest baseline)
    ``grpo``                 ``min(ratio * A, clip(ratio) * A)`` (pessimistic PPO-style clip)
    ``cispo_*``              ``sg(clip(ratio)) * A * current``  (MiniMax CISPO, Eq. 4-5)

    In the CISPO family the gradient reaches the policy only through
    ``current_logprobs``: the truncated ratio is detached, so a token whose ratio
    left the trust region still contributes gradient, with a constant weight.
    That is the whole point of the objective, and it is why ``advantages``,
    ``behavior_logprobs``, and ``reference_logprobs`` must never carry gradient.
    """

    # Only `current_logprobs` is a function of the parameters. Everything else
    # is data, and is detached here rather than by convention -- so that a later
    # change which computes one of them in-graph cannot quietly turn the ratio
    # denominator into a second thing being optimized.
    behavior_logprobs = ops.stop_gradient(behavior_logprobs)
    advantages = ops.stop_gradient(advantages)
    if reference_logprobs is not None:
        reference_logprobs = ops.stop_gradient(reference_logprobs)

    (
        log_ratio,
        current_logprobs,
        weights,
        nonfinite_tokens,
        clamped_tokens,
    ) = _sanitize(ops, current_logprobs, behavior_logprobs, weights)
    token_count = ops.sum(weights)
    ratio = ops.exp(log_ratio)

    if spec.is_cispo:
        truncated = ops.stop_gradient(ops.clip(ratio, spec.clip_low, spec.clip_high))
        per_token = truncated * advantages * current_logprobs
        clipped_tokens = ops.sum(
            ops.greater(ops.abs(ratio - truncated), ops.scalar(0.0)) * weights
        )
    elif spec.name == "grpo":
        clipped_ratio = ops.clip(ratio, spec.clip_low, spec.clip_high)
        per_token = ops.minimum(ratio * advantages, clipped_ratio * advantages)
        clipped_tokens = ops.sum(
            ops.greater(ops.abs(ratio - clipped_ratio), ops.scalar(0.0)) * weights
        )
    else:  # importance_sampling
        per_token = ratio * advantages
        clipped_tokens = ops.scalar(0.0)

    if tis_weights is not None:
        # Truncated importance sampling multiplies in AFTER the objective has
        # formed its own surrogate and never inside its clip. The two corrections
        # answer different questions -- how far the policy moved since collection
        # versus whether the sampler and the trainer agree about the same weights
        # -- and a codebase that folds them together can report neither honestly.
        # Detached: this is a measured property of the collection, not a thing
        # being optimized.
        per_token = per_token * ops.stop_gradient(tis_weights)

    policy_loss = -ops.sum(per_token * weights) / token_count
    approx_kl = ops.sum(ops.square(log_ratio) * weights) / (
        ops.scalar(2.0) * token_count
    )
    mean_ratio = ops.sum(ratio * weights) / token_count
    clip_fraction = clipped_tokens / token_count

    reference_kl = ops.scalar(0.0)
    if reference_logprobs is not None and reference_mask is not None:
        # k3 (Schulman): exp(r) - r - 1 >= 0 for all r, unlike the plain
        # difference estimator, which goes negative and hides a diverging policy.
        ref_weights = weights * reference_mask
        ref_count = ops.maximum(ops.sum(ref_weights), ops.scalar(1e-8))
        log_ref_ratio = ops.clip(
            ops.float32(reference_logprobs) - ops.float32(current_logprobs),
            -LOG_RATIO_CLAMP,
            LOG_RATIO_CLAMP,
        )
        k3 = ops.exp(log_ref_ratio) - log_ref_ratio - ops.scalar(1.0)
        reference_kl = ops.sum(k3 * ref_weights) / ref_count

    entropy_term = ops.scalar(0.0) if entropy is None else entropy
    loss = (
        policy_loss
        + ops.scalar(spec.kl_beta) * reference_kl
        - ops.scalar(spec.entropy_coef) * entropy_term
    )

    return loss, {
        "loss": loss,
        "policy_loss": policy_loss,
        "reference_kl": reference_kl,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
        "mean_ratio": mean_ratio,
        "entropy": entropy_term,
        "token_count": token_count,
        "clamped_token_count": clamped_tokens,
        "nonfinite_token_count": nonfinite_tokens,
    }

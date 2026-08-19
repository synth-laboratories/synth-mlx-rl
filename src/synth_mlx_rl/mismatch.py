"""The sampling/training mismatch stage: measure it, then correct or refuse.

Four logprob populations exist in this system and only two of them belong in a
ratio together:

    rollout_logprobs    the sampler emitted these while generating
    behavior_logprobs   the trainer recomputed them under the SAME frozen snapshot
    current_logprobs    differentiable, under the policy being updated
    reference_logprobs  optional frozen reference, for the KL term

Two ratios, which must never be multiplied into one:

    policy update ratio     exp(current  - behavior)   -> clipping, in kernel.py
    sampling mismatch ratio exp(behavior - rollout)    -> TIS, here

They answer different questions. The first asks how far the policy has moved
since collection. The second asks whether the sampler and the trainer even agree
about what the *same* weights predict for the *same* tokens -- a disagreement
caused by kernel differences, dtype, batching, or a sampler that quietly served
a different snapshot. Folding them together hides the second inside the first,
and the second is the one that indicates the run is invalid rather than merely
off-policy.

A large mismatch is not a number to correct away by default. `MismatchPolicy`
refuses past a bound, because past some point the collected data was not
produced by the policy the trainer thinks it was, and TIS-weighting garbage
yields a confident update in an arbitrary direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

__all__ = [
    "MismatchError",
    "MismatchPolicy",
    "MismatchReport",
    "TisWeights",
    "measure_mismatch",
    "tis_weights",
]

#: Log-ratios are clamped to this before `exp` everywhere in the codebase.
LOG_RATIO_CLAMP = 20.0

VERDICT_OK = "ok"
VERDICT_CORRECT = "correct_with_tis"
VERDICT_REFUSE = "refuse"


class MismatchError(RuntimeError):
    """The measured mismatch exceeds what the policy admits."""


@dataclass(frozen=True, slots=True)
class MismatchReport:
    """What the trainer and the sampler disagreed about, in full.

    `verdict` is advisory until a `MismatchPolicy` is applied; the numbers are
    reported whatever the verdict, because a refused run is still evidence.
    """

    token_count: int
    max_abs_diff: float
    mean_abs_diff: float
    ratio_mean: float
    ratio_p50: float
    ratio_p95: float
    ratio_max: float
    ess_ratio: float
    clamped_token_count: int
    nonfinite_token_count: int
    verdict: str = VERDICT_OK
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_count": self.token_count,
            "train_rollout_logprob_abs_diff": self.max_abs_diff,
            "train_rollout_logprob_abs_diff_mean": self.mean_abs_diff,
            "tis_ratio_mean": self.ratio_mean,
            "tis_ratio_p50": self.ratio_p50,
            "tis_ratio_p95": self.ratio_p95,
            "tis_ratio_max": self.ratio_max,
            "ess_ratio": self.ess_ratio,
            "clamped_token_count": self.clamped_token_count,
            "nonfinite_token_count": self.nonfinite_token_count,
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TisWeights:
    """Truncated importance sampling weights and what they cost."""

    weights: np.ndarray
    clipped_fraction: float
    metrics: dict[str, float] = field(default_factory=dict)


def _aligned(
    behavior_logprobs: Sequence[float],
    rollout_logprobs: Sequence[float],
    weights: Sequence[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    behavior = np.asarray(behavior_logprobs, dtype=np.float64)
    rollout = np.asarray(rollout_logprobs, dtype=np.float64)
    if behavior.shape != rollout.shape:
        # Silently truncating to the shorter of the two is how an off-by-one
        # alignment bug becomes a plausible-looking metric.
        raise MismatchError(
            f"behavior_logprobs has {behavior.size} entries and rollout_logprobs "
            f"has {rollout.size}; they must align one-to-one on the same tokens"
        )
    if behavior.size == 0:
        raise MismatchError("no tokens to compare")
    mask = (
        np.ones_like(behavior)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    if mask.shape != behavior.shape:
        raise MismatchError("weights must align with the logprob arrays")
    return behavior, rollout, mask


def measure_mismatch(
    *,
    behavior_logprobs: Sequence[float],
    rollout_logprobs: Sequence[float],
    weights: Sequence[float] | None = None,
) -> MismatchReport:
    """Compare what the trainer recomputed against what the sampler emitted.

    Everything is float64 here regardless of the training dtype: the quantity
    being measured is small differences between logprobs, and computing it in
    the same reduced precision that may have caused the difference would hide it.
    """

    behavior, rollout, mask = _aligned(behavior_logprobs, rollout_logprobs, weights)

    finite = np.isfinite(behavior) & np.isfinite(rollout)
    nonfinite_tokens = int(((~finite) * (mask > 0)).sum())
    active = (mask > 0) & finite
    token_count = int(active.sum())
    if token_count == 0:
        raise MismatchError(
            "every compared token was masked or nonfinite; there is nothing to "
            "measure, which is a failed collection rather than a zero mismatch"
        )

    diff = behavior[active] - rollout[active]
    clamped = np.abs(diff) > LOG_RATIO_CLAMP
    clamped_tokens = int(clamped.sum())
    log_ratio = np.clip(diff, -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
    ratio = np.exp(log_ratio)

    # ESS for self-normalized importance weights: (sum w)^2 / sum(w^2), as a
    # fraction of n. 1.0 means every token carries equal weight; near 0 means a
    # handful of tokens dominate the update and the rest are decoration.
    total = float(ratio.sum())
    squared = float((ratio**2).sum())
    ess_ratio = (total * total) / (squared * token_count) if squared > 0 else 0.0

    abs_diff = np.abs(diff)
    return MismatchReport(
        token_count=token_count,
        max_abs_diff=float(abs_diff.max()),
        mean_abs_diff=float(abs_diff.mean()),
        ratio_mean=float(ratio.mean()),
        ratio_p50=float(np.percentile(ratio, 50)),
        ratio_p95=float(np.percentile(ratio, 95)),
        ratio_max=float(ratio.max()),
        ess_ratio=float(min(ess_ratio, 1.0)),
        clamped_token_count=clamped_tokens,
        nonfinite_token_count=nonfinite_tokens,
    )


@dataclass(frozen=True, slots=True)
class MismatchPolicy:
    """When a measured mismatch is tolerable, correctable, or disqualifying.

    Defaults are deliberately strict. On a single-process service sampling and
    training the same resident weights, agreement should be near-exact; a large
    disagreement means something structural (a different snapshot served, a
    tokenizer drift, an alignment bug), not noise to be reweighted.
    """

    #: Below this, the two paths agree and no correction is applied.
    ok_abs_diff: float = 1e-3
    #: Above this, the collection is disqualified rather than corrected.
    max_abs_diff: float = 0.5
    #: Below this effective-sample-size fraction, a few tokens dominate.
    min_ess_ratio: float = 0.5
    #: Any nonfinite compared token disqualifies by default.
    allow_nonfinite: bool = False

    def evaluate(self, report: MismatchReport) -> MismatchReport:
        """Return the report with a verdict and a reason attached."""

        def refuse(reason: str) -> MismatchReport:
            return _with_verdict(report, VERDICT_REFUSE, reason)

        if report.nonfinite_token_count and not self.allow_nonfinite:
            return refuse(
                f"{report.nonfinite_token_count} compared tokens were nonfinite"
            )
        if report.max_abs_diff > self.max_abs_diff:
            return refuse(
                f"max |behavior - rollout| = {report.max_abs_diff:.4f} exceeds "
                f"{self.max_abs_diff}; the collected data was not produced by the "
                "policy the trainer is scoring"
            )
        if report.ess_ratio < self.min_ess_ratio:
            return refuse(
                f"effective sample size {report.ess_ratio:.3f} is below "
                f"{self.min_ess_ratio}; a few tokens would dominate the update"
            )
        if report.max_abs_diff <= self.ok_abs_diff:
            return _with_verdict(report, VERDICT_OK, None)
        return _with_verdict(
            report,
            VERDICT_CORRECT,
            f"max |behavior - rollout| = {report.max_abs_diff:.4f}; applying TIS",
        )

    def enforce(self, report: MismatchReport) -> MismatchReport:
        """`evaluate`, but raise on a refusal instead of returning it."""

        judged = self.evaluate(report)
        if judged.verdict == VERDICT_REFUSE:
            raise MismatchError(judged.reason or "mismatch policy refused the batch")
        return judged


def _with_verdict(report: MismatchReport, verdict: str, reason: str | None) -> MismatchReport:
    return MismatchReport(
        token_count=report.token_count,
        max_abs_diff=report.max_abs_diff,
        mean_abs_diff=report.mean_abs_diff,
        ratio_mean=report.ratio_mean,
        ratio_p50=report.ratio_p50,
        ratio_p95=report.ratio_p95,
        ratio_max=report.ratio_max,
        ess_ratio=report.ess_ratio,
        clamped_token_count=report.clamped_token_count,
        nonfinite_token_count=report.nonfinite_token_count,
        verdict=verdict,
        reason=reason,
    )


def tis_weights(
    *,
    behavior_logprobs: Sequence[float],
    rollout_logprobs: Sequence[float],
    weights: Sequence[float] | None = None,
    clip_low: float = 0.5,
    clip_high: float = 1.5,
) -> TisWeights:
    """Truncated importance sampling: clamp `exp(behavior - rollout)`.

    This multiplies into the per-token policy loss as a SEPARATE stage from the
    objective's own clipping. It is a correction for the sampler and the trainer
    disagreeing, not a trust region on the policy update, and a codebase that
    applies it inside the clip can no longer report either one honestly.

    Follows the MILES `vanilla_tis_function` shape: metrics report the pre-clamp
    ratio, so a clamp that is doing heavy lifting is visible rather than hidden
    behind its own output.
    """

    if not 0.0 < clip_low <= 1.0 <= clip_high:
        raise MismatchError(
            f"TIS bounds must satisfy 0 < clip_low <= 1 <= clip_high; got "
            f"[{clip_low}, {clip_high}]"
        )
    behavior, rollout, mask = _aligned(behavior_logprobs, rollout_logprobs, weights)
    finite = np.isfinite(behavior) & np.isfinite(rollout)
    log_ratio = np.clip(
        np.where(finite, behavior - rollout, 0.0), -LOG_RATIO_CLAMP, LOG_RATIO_CLAMP
    )
    raw = np.exp(log_ratio)
    clamped = np.clip(raw, clip_low, clip_high)
    # A nonfinite token contributes nothing rather than poisoning the sum.
    clamped = np.where(finite, clamped, 0.0)

    active = mask > 0
    denom = float(active.sum()) or 1.0
    clipped_fraction = float(((raw != clamped) & active & finite).sum() / denom)
    return TisWeights(
        weights=clamped,
        clipped_fraction=clipped_fraction,
        metrics={
            "tis": float(raw[active].mean()) if active.any() else 0.0,
            "tis_abs": float(np.abs(raw[active] - 1.0).mean()) if active.any() else 0.0,
            "tis_clipfrac": clipped_fraction,
            "tis_clip_low": clip_low,
            "tis_clip_high": clip_high,
        },
    )

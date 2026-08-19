"""Objective names and their validated hyper-parameters.

This module is deliberately free of both MLX and NumPy: it is the one place that
decides what an objective *is*, so the MLX path, the NumPy reference oracle, and
the HTTP schema cannot drift from one another.

Vocabulary (finalized plan, section 5.3). Four log-probability populations, never
conflated:

    rollout_logprobs    the sampler's raw distribution at generation time,
                        recorded BEFORE top-p / top-k / min-p truncation
    behavior_logprobs   trainer forward pass at the pinned snapshot; the ratio
                        DENOMINATOR; no gradient
    current_logprobs    trainer forward pass at the live adapter; the ratio
                        NUMERATOR; carries the gradient
    reference_logprobs  frozen reference policy; k3 KL penalty only; no gradient
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

#: Objectives this service will execute.
ObjectiveName = Literal[
    "cross_entropy",
    "importance_sampling",
    "grpo",
    "cispo_minimax",
    "cispo_two_sided",
    "ppo",
]

SUPPORTED_OBJECTIVES: Final[tuple[str, ...]] = (
    "cross_entropy",
    "importance_sampling",
    "grpo",
    "cispo_minimax",
    "cispo_two_sided",
)

#: Named but refused in v1. There is no value head and no critic, and aliasing
#: `ppo` onto the GRPO clipped surrogate would misdescribe every run that used
#: it (finalized plan, decision D8).
UNAVAILABLE_OBJECTIVES: Final[dict[str, str]] = {
    "ppo": (
        "ppo is not available in v1: this service has no value head and no "
        "critic, and aliasing ppo onto the grpo clipped surrogate would "
        "mislabel the run. Use grpo, cispo_minimax, cispo_two_sided, or "
        "importance_sampling."
    ),
}

#: Objectives that need behavior_logprobs and advantages on every datum.
POLICY_OBJECTIVES: Final[frozenset[str]] = frozenset(
    {"importance_sampling", "grpo", "cispo_minimax", "cispo_two_sided"}
)

CISPO_OBJECTIVES: Final[frozenset[str]] = frozenset(
    {"cispo_minimax", "cispo_two_sided"}
)

#: The log-ratio is clamped to this range before ``exp`` in every code path, in
#: float32, in both the NumPy reference and the MLX kernel. exp(20) ~ 4.85e8,
#: which is finite in float32 and far outside any range a sane run reaches.
LOG_RATIO_CLAMP: Final[float] = 20.0


class ObjectiveError(ValueError):
    """A refusal to run an objective as configured."""


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """A validated objective configuration.

    ``eps_low`` / ``eps_high`` follow the CISPO reference convention: they are
    deltas from 1, so the truncation interval is
    ``[1 - eps_low, 1 + eps_high]``. ``eps_low >= 1.0`` therefore pushes the
    lower bound to zero or below, which disables it -- that is what "canonical,
    single-sided CISPO" means.
    """

    name: str
    clip_epsilon: float = 0.2
    eps_low: float = 1.0
    eps_high: float = 4.0
    kl_beta: float = 0.0
    entropy_coef: float = 0.0

    def __post_init__(self) -> None:
        if self.name in UNAVAILABLE_OBJECTIVES:
            raise ObjectiveError(UNAVAILABLE_OBJECTIVES[self.name])
        if self.name not in SUPPORTED_OBJECTIVES:
            raise ObjectiveError(
                f"unknown objective {self.name!r}; supported: "
                + ", ".join(SUPPORTED_OBJECTIVES)
            )
        if self.kl_beta < 0.0:
            raise ObjectiveError("kl_beta must be non-negative")
        if self.entropy_coef < 0.0:
            raise ObjectiveError("entropy_coef must be non-negative")
        if self.name == "grpo" and not 0.0 < self.clip_epsilon < 1.0:
            raise ObjectiveError("clip_epsilon must be in (0, 1)")
        if self.name in CISPO_OBJECTIVES:
            if self.eps_high <= 0.0:
                raise ObjectiveError("eps_high must be positive")
            if self.eps_low < 0.0:
                raise ObjectiveError("eps_low must be non-negative")
            if self.name == "cispo_minimax" and self.eps_low < 1.0:
                # SLIME only warns here (slime/utils/arguments.py). A warning is
                # exactly how a two-sided run ends up labelled CISPO in a report.
                raise ObjectiveError(
                    "cispo_minimax is single-sided by definition: eps_low must "
                    f"be >= 1.0 so the lower truncation bound 1 - eps_low = "
                    f"{1.0 - self.eps_low:.4g} is inactive, but eps_low="
                    f"{self.eps_low!r} keeps it active. Either set eps_low=1.0 "
                    "(canonical; tune eps_high, e.g. 4.0), or ask for the "
                    "two-sided objective by its own name, cispo_two_sided."
                )

    @property
    def clip_low(self) -> float:
        """Lower truncation bound on the ratio."""

        if self.name in CISPO_OBJECTIVES:
            return 1.0 - self.eps_low
        return 1.0 - self.clip_epsilon

    @property
    def clip_high(self) -> float:
        """Upper truncation bound on the ratio."""

        if self.name in CISPO_OBJECTIVES:
            return 1.0 + self.eps_high
        return 1.0 + self.clip_epsilon

    @property
    def is_cispo(self) -> bool:
        return self.name in CISPO_OBJECTIVES

    @property
    def needs_behavior_logprobs(self) -> bool:
        return self.name in POLICY_OBJECTIVES

"""A tiny forward-mode autodiff backend for :mod:`synth_mlx_rl.kernel`.

MLX cannot be installed on the machine this package is developed on, so the
claim "the CISPO gradient reaches the policy only through ``current_logprobs``"
cannot be checked by running MLX's autograd. It can be checked exactly, though,
by evaluating the *same kernel source* over dual numbers.

A :class:`Dual` carries a value and a directional derivative. Seeding the
derivative of one input and reading the derivative of the loss gives the
directional derivative of the loss with respect to that input -- which is zero
if and only if no gradient flows. ``stop_gradient`` zeroes the derivative part,
exactly as MLX's does.

Forward mode is enough here because every question is of the form "does this
input influence the loss, and by how much in a known direction". It is not a
training path and is not used outside tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class Dual:
    """``value + eps * tangent``, elementwise over NumPy arrays."""

    __slots__ = ("value", "tangent")

    #: Stop NumPy from absorbing a Dual into an object array when a plain
    #: ndarray is on the left of an operator. With this set, NumPy returns
    #: NotImplemented and Python defers to Dual's reflected operator.
    __array_ufunc__ = None
    __array_priority__ = 1000.0

    def __init__(self, value: Any, tangent: Any = None):
        self.value = np.asarray(value, dtype=np.float64)
        self.tangent = (
            np.zeros_like(self.value)
            if tangent is None
            else np.broadcast_to(
                np.asarray(tangent, dtype=np.float64), self.value.shape
            ).copy()
        )

    # -- construction ----------------------------------------------------

    @staticmethod
    def lift(x: Any) -> "Dual":
        return x if isinstance(x, Dual) else Dual(x)

    @staticmethod
    def seed(value: Any, direction: Any) -> "Dual":
        """A variable whose derivative is followed along ``direction``."""

        return Dual(value, direction)

    # -- arithmetic ------------------------------------------------------

    def __add__(self, other: Any) -> "Dual":
        o = Dual.lift(other)
        return Dual(self.value + o.value, self.tangent + o.tangent)

    __radd__ = __add__

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.tangent)

    def __sub__(self, other: Any) -> "Dual":
        return self + (-Dual.lift(other))

    def __rsub__(self, other: Any) -> "Dual":
        return Dual.lift(other) + (-self)

    def __mul__(self, other: Any) -> "Dual":
        o = Dual.lift(other)
        return Dual(
            self.value * o.value,
            self.tangent * o.value + self.value * o.tangent,
        )

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Dual":
        o = Dual.lift(other)
        return Dual(
            self.value / o.value,
            (self.tangent * o.value - self.value * o.tangent) / (o.value * o.value),
        )

    def __rtruediv__(self, other: Any) -> "Dual":
        return Dual.lift(other) / self

    # -- comparisons return plain masks, never duals ---------------------

    def __gt__(self, other: Any) -> Any:
        return self.value > Dual.lift(other).value

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Dual(value={self.value!r}, tangent={self.tangent!r})"


def _value(x: Any) -> Any:
    return x.value if isinstance(x, Dual) else np.asarray(x, dtype=np.float64)


def _tangent(x: Any) -> Any:
    return x.tangent if isinstance(x, Dual) else 0.0


class DualOps:
    """Forward-mode backend satisfying :class:`synth_mlx_rl.kernel.ArrayOps`."""

    def scalar(self, value: float) -> Any:
        return Dual(np.float64(value))

    def exp(self, x: Any) -> Any:
        v = np.exp(_value(x))
        return Dual(v, v * _tangent(x))

    def clip(self, x: Any, low: float, high: float) -> Any:
        v = _value(x)
        inside = ((v > low) & (v < high)).astype(np.float64)
        return Dual(np.clip(v, low, high), inside * _tangent(x))

    def minimum(self, a: Any, b: Any) -> Any:
        av, bv = _value(a), _value(b)
        pick_a = (av <= bv).astype(np.float64)
        return Dual(
            np.minimum(av, bv),
            pick_a * _tangent(a) + (1.0 - pick_a) * _tangent(b),
        )

    def maximum(self, a: Any, b: Any) -> Any:
        av, bv = _value(a), _value(b)
        pick_a = (av >= bv).astype(np.float64)
        return Dual(
            np.maximum(av, bv),
            pick_a * _tangent(a) + (1.0 - pick_a) * _tangent(b),
        )

    def abs(self, x: Any) -> Any:
        v = _value(x)
        return Dual(np.abs(v), np.sign(v) * _tangent(x))

    def square(self, x: Any) -> Any:
        v = _value(x)
        return Dual(v * v, 2.0 * v * _tangent(x))

    def sum(self, x: Any) -> Any:
        return Dual(np.sum(_value(x)), np.sum(_tangent(x)))

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        mask = np.asarray(_value(condition)) > 0.0
        return Dual(
            np.where(mask, _value(a), _value(b)),
            np.where(mask, _tangent(a), _tangent(b)),
        )

    def stop_gradient(self, x: Any) -> Any:
        return Dual(_value(x))

    def isfinite(self, x: Any) -> Any:
        return Dual(np.isfinite(_value(x)).astype(np.float64))

    def greater(self, a: Any, b: Any) -> Any:
        return Dual((_value(a) > _value(b)).astype(np.float64))

    def float32(self, x: Any) -> Any:
        cast = np.asarray(_value(x), dtype=np.float32).astype(np.float64)
        return Dual(cast, _tangent(x))

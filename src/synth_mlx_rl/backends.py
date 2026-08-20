"""The array-operation backend for :mod:`synth_mlx_rl.kernel`.

:class:`MlxOps` receives the ``mlx.core`` module as a constructor argument
rather than importing it, so this file is importable before the engine thread
has been built.
"""

from __future__ import annotations

from typing import Any


class MlxOps:
    """MLX backend. ``mx`` is ``mlx.core``, injected rather than imported."""

    __slots__ = ("mx",)

    def __init__(self, mx: Any):
        self.mx = mx

    def scalar(self, value: float) -> Any:
        return self.mx.array(value, dtype=self.mx.float32)

    def exp(self, x: Any) -> Any:
        return self.mx.exp(x)

    def clip(self, x: Any, low: float, high: float) -> Any:
        return self.mx.clip(x, low, high)

    def minimum(self, a: Any, b: Any) -> Any:
        return self.mx.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return self.mx.maximum(a, b)

    def abs(self, x: Any) -> Any:
        return self.mx.abs(x)

    def square(self, x: Any) -> Any:
        return self.mx.square(x)

    def sum(self, x: Any) -> Any:
        return self.mx.sum(x)

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        return self.mx.where(condition > 0.0, a, b)

    def stop_gradient(self, x: Any) -> Any:
        return self.mx.stop_gradient(x)

    def isfinite(self, x: Any) -> Any:
        return self.mx.isfinite(x).astype(self.mx.float32)

    def greater(self, a: Any, b: Any) -> Any:
        return (a > b).astype(self.mx.float32)

    def float32(self, x: Any) -> Any:
        return x.astype(self.mx.float32)

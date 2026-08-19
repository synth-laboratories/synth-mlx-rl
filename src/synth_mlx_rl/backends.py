"""Array-operation backends for :mod:`synth_mlx_rl.kernel`.

Two backends live here. Neither imports MLX at module scope, and
:class:`MlxOps` receives the ``mlx.core`` module as a constructor argument, so
importing this file on a machine with no MLX is always safe.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class NumpyOps:
    """NumPy backend. Used by the portable test suite and by CPU tooling.

    ``stop_gradient`` is the identity here because NumPy carries no tape; the
    gradient claims are checked by :mod:`synth_mlx_rl.testing.autodiff`, whose
    backend honours it.
    """

    def scalar(self, value: float) -> Any:
        return np.float64(value)

    def exp(self, x: Any) -> Any:
        return np.exp(x)

    def clip(self, x: Any, low: float, high: float) -> Any:
        return np.clip(x, low, high)

    def minimum(self, a: Any, b: Any) -> Any:
        return np.minimum(a, b)

    def maximum(self, a: Any, b: Any) -> Any:
        return np.maximum(a, b)

    def abs(self, x: Any) -> Any:
        return np.abs(x)

    def square(self, x: Any) -> Any:
        return np.square(x)

    def sum(self, x: Any) -> Any:
        return np.sum(x)

    def where(self, condition: Any, a: Any, b: Any) -> Any:
        return np.where(np.asarray(condition) > 0.0, a, b)

    def stop_gradient(self, x: Any) -> Any:
        return x

    def isfinite(self, x: Any) -> Any:
        return np.isfinite(x).astype(np.float64)

    def greater(self, a: Any, b: Any) -> Any:
        return (np.asarray(a) > np.asarray(b)).astype(np.float64)

    def float32(self, x: Any) -> Any:
        # Compute the log-ratio at float32 precision, then keep accumulating in
        # the backend's working dtype. MLX runs float32 throughout; this keeps
        # the one step where precision is load-bearing identical.
        return np.asarray(x, dtype=np.float32).astype(np.float64)


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

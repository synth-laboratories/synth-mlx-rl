"""Test-support code that ships with the package.

Everything in here runs without MLX. The fake engine makes the HTTP protocol,
the snapshot pool, and the rollout record store testable on any machine, which
is the only way this package is developed at all: MLX cannot be installed on the
development host (finalized plan, correction C11).
"""

from __future__ import annotations

from .autodiff import Dual, DualOps
from .fake_engine import FakeEngine, FakeTokenizer

__all__ = ["Dual", "DualOps", "FakeEngine", "FakeTokenizer"]

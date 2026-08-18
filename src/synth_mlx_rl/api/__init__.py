"""HTTP surfaces.

Two OpenAI-compatible families (decision D9), plus the ``/v1/synth`` namespace
that holds what OpenAI does not define: policy snapshots, rollout records, and
the capability report.

``create_app`` is imported lazily so that importing this package never
constructs an application (and never reads the environment) as a side effect.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_app"]


def __getattr__(name: str) -> Any:
    if name == "create_app":
        from .app import create_app

        return create_app
    raise AttributeError(name)

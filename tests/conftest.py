"""One resident MLX engine for the whole session.

There is no fake engine. The HTTP suite runs against the real model, because a
protocol that only ever answered a stub proves nothing about the service people
actually run. The engine is session-scoped so the model loads once, and it is
shut down on the way out.
"""

from __future__ import annotations

import pytest

from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.config import Settings

mlx_required = pytest.importorskip


@pytest.fixture(scope="session")
def settings(tmp_path_factory) -> Settings:
    return Settings(checkpoint_dir=tmp_path_factory.mktemp("adapters"))


@pytest.fixture(scope="session")
def app(settings):
    mlx_required("mlx.core")
    mlx_required("mlx_lm")
    return create_app(settings=settings)


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def engine(client):
    """The live engine behind the running app.

    Bound during the app lifespan, so it is reached through a started client
    rather than constructed separately -- there is exactly one resident model
    and tests share it.
    """

    return client.app.state.engine

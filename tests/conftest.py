from __future__ import annotations

import pytest

from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing import FakeEngine


@pytest.fixture
def engine(tmp_path) -> FakeEngine:
    return FakeEngine(tmp_path)


@pytest.fixture
def client(engine):
    from fastapi.testclient import TestClient

    with TestClient(create_app(engine=engine)) as test_client:
        yield test_client

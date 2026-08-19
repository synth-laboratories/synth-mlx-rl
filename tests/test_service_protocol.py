"""The native learner protocol, end to end against the fake engine.

Ported from the MIT-licensed prototype's app test and extended for snapshots,
rollout records, and the objective vocabulary.
"""

from __future__ import annotations

import pytest

from synth_mlx_rl.api.app import create_app, mounted_paths


def test_both_api_families_are_mounted(client) -> None:
    paths = mounted_paths(client.app)
    assert "/v1/chat/completions" in paths
    assert "/v1/responses" in paths


def test_end_to_end_protocol(client) -> None:
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["latest_policy_snapshot_id"]

    state = client.get("/v1/state").json()
    assert state["enable_thinking"] is False
    assert state["tokenizer_digest"] and state["template_digest"]
    assert state["training_version"] == 0

    sample = client.post(
        "/v1/sample",
        json={"prompt": "hello", "max_tokens": 8, "temperature": 0.0, "num_samples": 2},
    )
    assert sample.status_code == 200
    samples = sample.json()["samples"]
    assert len(samples) == 2
    assert all(s["proxy_request_id"] for s in samples)
    assert all(
        len(s["rollout_logprobs"]) == len(s["completion_token_ids"]) for s in samples
    )

    logprobs = client.post("/v1/synth/logprobs", json={"token_ids": [1, 2, 3]})
    scored = logprobs.json()["logprobs"]
    # The first token has no predecessor to be scored against, and the rest are
    # real log-probabilities. Their values belong to the model, not to this test.
    assert scored[0] is None
    assert len(scored) == 3
    assert all(isinstance(value, float) and value < 0 for value in scored[1:])

    datum = {"input_ids": [1, 2, 3], "target_ids": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
    forward = client.post(
        "/v1/forward_backward", json={"data": [datum], "loss_fn": "cross_entropy"}
    )
    assert forward.status_code == 200
    assert forward.json()["accumulation_count"] == 1

    step = client.post("/v1/optim_step", json={"params": {"learning_rate": 5e-5}})
    assert step.status_code == 200
    assert step.json()["training_version"] == 1

    saved = client.post("/v1/checkpoints/save", json={"name": "unit-test"})
    assert saved.status_code == 200


def test_optim_step_without_gradients_is_400(client) -> None:
    response = client.post("/v1/optim_step", json={"params": {}})
    assert response.status_code == 400
    assert "no accumulated gradients" in response.json()["detail"]


def test_missing_checkpoint_is_404(client) -> None:
    response = client.post("/v1/checkpoints/load", json={"name": "does-not-exist"})
    assert response.status_code == 404


def test_cispo_forward_backward_over_http(client) -> None:
    datum = {
        "input_ids": [1, 2, 3],
        "target_ids": [2, 3, 4],
        "weights": [0.0, 1.0, 1.0],
        "behavior_logprobs": [0.0, -0.5, -0.4],
        "advantages": 1.5,
    }
    response = client.post(
        "/v1/forward_backward",
        json={
            "data": [datum],
            "loss_fn": "cispo_minimax",
            "eps_low": 1.0,
            "eps_high": 4.0,
        },
    )
    assert response.status_code == 200, response.text
    metrics = response.json()["metrics"]
    for key in (
        "policy_loss",
        "clip_fraction",
        "mean_ratio",
        "approx_kl",
        "clamped_token_count",
        "nonfinite_token_count",
        "token_count",
    ):
        assert key in metrics
    assert metrics["token_count"] == 2.0


def test_cispo_minimax_refusal_is_a_422_over_http(client) -> None:
    datum = {
        "input_ids": [1],
        "target_ids": [2],
        "weights": [1.0],
        "behavior_logprobs": [-1.0],
        "advantages": 1.0,
    }
    response = client.post(
        "/v1/forward_backward",
        json={"data": [datum], "loss_fn": "cispo_minimax", "eps_low": 0.2},
    )
    assert response.status_code == 422
    assert "cispo_two_sided" in response.text


def test_accumulation_weight_is_a_global_token_total(client) -> None:
    """Two microbatches of different length must not be averaged as equals."""

    short = {"input_ids": [1, 2], "target_ids": [2, 3], "weights": [1.0, 1.0]}
    long = {
        "input_ids": [1, 2, 3, 4],
        "target_ids": [2, 3, 4, 5],
        "weights": [1.0, 1.0, 1.0, 1.0],
    }
    client.post("/v1/forward_backward", json={"data": [short]})
    second = client.post("/v1/forward_backward", json={"data": [long]})
    assert second.json()["metrics"]["accumulation_token_weight"] == 6.0

    step = client.post("/v1/optim_step", json={"params": {}}).json()
    assert step["applied_accumulations"] == 2


def test_service_starts_without_importing_mlx() -> None:
    """Constructing the app must not import mlx.

    MLX streams are thread-affine, so the engine has to be BUILT on the worker
    thread that will call it, not on whatever thread happens to construct the
    app. An import at construction time is the first step towards building it
    in the wrong place, and the failure it produces -- `There is no Stream(gpu,
    N) in current thread` inside mx.eval -- points nowhere near the cause.

    Checked in a subprocess, deliberately: asserting on this process's
    `sys.modules` is order-dependent, because by the time this runs the session
    engine has legitimately loaded mlx.
    """

    import subprocess
    import sys

    probe = (
        "import sys;"
        "from synth_mlx_rl.api.app import create_app;"
        "app = create_app();"
        "assert app is not None;"
        "leaked = sorted(m for m in sys.modules if m.split('.')[0] in {'mlx', 'mlx_lm'});"
        "print('LEAKED:' + ','.join(leaked)) if leaked else print('CLEAN')"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "CLEAN", result.stdout


def test_mixing_reductions_in_one_accumulation_window_is_refused(client) -> None:
    """`mean_tokens` weights each call by its tokens and divides by the total at
    optim_step; `sum` accumulates as-is and divides by nothing. Mixing them
    composes the two normalizations into something that is neither convention,
    so it is refused rather than silently averaged."""
    datum = {"input_ids": [1, 2, 3], "target_ids": [2, 3, 4], "weights": [0.0, 1.0, 1.0]}
    first = client.post(
        "/v1/forward_backward",
        json={"data": [datum], "loss_fn": "cross_entropy", "reduction": "sum"},
    )
    assert first.status_code == 200, first.text
    second = client.post(
        "/v1/forward_backward",
        json={"data": [datum], "loss_fn": "cross_entropy", "reduction": "mean_tokens"},
    )
    assert second.status_code >= 400
    assert "mix reductions" in second.text

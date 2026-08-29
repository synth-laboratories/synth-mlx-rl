"""The Chat Completions family."""

from __future__ import annotations

import json


def test_non_streaming_completion(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "temperature": 0.0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["total_tokens"] > 0
    assert body["synth"]["proxy_request_ids"]
    assert response.headers["X-Proxy-Request-Id"]
    assert response.headers["X-Policy-Pin"] == body["synth"]["policy_snapshot_id"]


def test_n_greater_than_one_returns_one_record_per_choice(client) -> None:
    body = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "n": 3},
    ).json()
    assert len(body["choices"]) == 3
    assert len(body["synth"]["proxy_request_ids"]) == 3


def test_wrong_model_is_refused(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "not-the-resident-model",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"]["error_code"] == "model_not_resident"


def test_workshop_local_model_alias_resolves_to_the_resident_model(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-local-base",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["model"] != "mlx-local-base"


def test_workshop_json_response_format_is_accepted(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mlx-local-base",
            "messages": [{"role": "user", "content": "return JSON"}],
            "response_format": {"type": "json_object"},
        },
    )
    assert response.status_code == 200, response.text


def test_logprobs_on_the_wire_are_refused_with_a_pointer(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "logprobs": True},
    )
    assert response.status_code == 422
    assert "/v1/synth/rollouts/" in response.text


def test_unknown_field_is_refused_rather_than_ignored(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": 1.5,
        },
    )
    assert response.status_code == 422


def test_streaming_frames_deltas_and_terminates(client) -> None:
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        proxy_request_id = response.headers["X-Proxy-Request-Id"]
        text = "".join(response.iter_text())

    chunks = [
        line[len("data: ") :]
        for line in text.split("\n\n")
        if line.startswith("data: ")
    ]
    assert chunks[-1] == "[DONE]"
    payloads = [json.loads(chunk) for chunk in chunks[:-1]]
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    streamed = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload["choices"]
    )
    assert streamed

    # The record was written before the first byte left the server: the stream
    # is advisory, the record is the authority.
    record = client.get(f"/v1/synth/rollouts/{proxy_request_id}").json()["record"]
    assert record["api_family"] == "chat_completions"
    usage_chunk = payloads[-1]
    assert usage_chunk["usage"]["completion_tokens"] == len(
        record["completion_token_ids"]
    )


def test_streamed_text_matches_the_non_streamed_text(client) -> None:
    body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    plain = client.post("/v1/chat/completions", json=body).json()
    with client.stream("POST", "/v1/chat/completions", json={**body, "stream": True}) as response:
        text = "".join(response.iter_text())
    payloads = [
        json.loads(line[len("data: ") :])
        for line in text.split("\n\n")
        if line.startswith("data: ") and not line.endswith("[DONE]")
    ]
    streamed = "".join(
        payload["choices"][0]["delta"].get("content", "")
        for payload in payloads
        if payload["choices"]
    )
    assert streamed == plain["choices"][0]["message"]["content"]


def test_policy_pin_header_is_honored(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Policy-Pin": snapshot["policy_snapshot_id"]},
    )
    assert response.json()["synth"]["policy_snapshot_id"] == snapshot["policy_snapshot_id"]


def test_conflicting_pins_are_refused_rather_than_resolved(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "policy_snapshot_id": "snap_other",
        },
        headers={"X-Policy-Pin": snapshot["policy_snapshot_id"]},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["error_code"] == "policy_pin_conflict"


def test_sampling_against_an_evicted_snapshot_fails_loudly(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    client.delete(f"/v1/synth/snapshots/{snapshot['policy_snapshot_id']}")
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "policy_snapshot_id": snapshot["policy_snapshot_id"],
        },
    )
    assert response.status_code == 410
    assert response.json()["detail"]["error_code"] == "policy_snapshot_evicted"


def test_unknown_snapshot_is_a_404_not_a_fallback(client) -> None:
    response = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "policy_snapshot_id": "snap_never_issued",
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "policy_snapshot_not_found"

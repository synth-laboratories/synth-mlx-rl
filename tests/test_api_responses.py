"""The Responses family.

Protocol properties carried over from the org's native Responses gateway: the
full history arrives every turn, tools are forwarded, and
``previous_response_id`` alone is refused before anything is sampled.
"""

from __future__ import annotations

import json


def test_non_streaming_response(client) -> None:
    response = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": "hi"}], "temperature": 0.0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "response"
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"][0]["type"] == "output_text"
    assert body["usage"]["total_tokens"] > 0
    assert response.headers["X-Policy-Pin"] == body["synth"]["policy_snapshot_id"]


def test_string_input_is_accepted(client) -> None:
    body = client.post("/v1/responses", json={"input": "hi"}).json()
    assert body["output"][0]["content"][0]["text"]


def test_previous_response_id_alone_is_refused_before_sampling(client, engine) -> None:
    before = len(engine.rollouts)
    response = client.post(
        "/v1/responses",
        json={
            "input": [{"role": "user", "content": "continue"}],
            "previous_response_id": "resp_prior",
        },
    )
    assert response.status_code == 422
    assert (
        response.json()["detail"]["error_code"]
        == "previous_response_id_requires_full_history"
    )
    # Nothing was sampled: no record was written.
    assert len(engine.rollouts) == before


def test_previous_response_id_with_inline_history_is_allowed(client) -> None:
    response = client.post(
        "/v1/responses",
        json={
            "input": [
                {"role": "user", "content": "weather?"},
                {
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "get_weather",
                    "arguments": "{}",
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_a",
                    "output": "75F",
                },
            ],
            "previous_response_id": "resp_prior",
        },
    )
    assert response.status_code == 200, response.text


def test_growing_full_history_is_accepted_turn_after_turn(client) -> None:
    history = [{"role": "user", "content": "weather in nyc, then celsius?"}]
    digests = []
    for round_index in range(3):
        response = client.post(
            "/v1/responses",
            json={
                "input": history,
                "tools": [
                    {"type": "function", "name": "get_weather", "parameters": {}}
                ],
            },
        )
        assert response.status_code == 200, response.text
        digests.append(response.json()["synth"]["render_digest"])
        history = history + [
            {
                "type": "function_call",
                "call_id": f"call_{round_index}",
                "name": "get_weather",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": f"call_{round_index}",
                "output": "75F",
            },
        ]
    # Each turn renders a longer transcript, so each digest is distinct: the
    # service is not silently reusing a cached prompt.
    assert len(set(digests)) == 3


def test_unsupported_input_item_is_named_not_guessed(client) -> None:
    response = client.post(
        "/v1/responses",
        json={"input": [{"type": "image_generation_call", "id": "x"}]},
    )
    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "unsupported_input_item"


def test_streaming_emits_the_responses_event_vocabulary(client) -> None:
    with client.stream(
        "POST",
        "/v1/responses",
        json={"input": [{"role": "user", "content": "hi"}], "stream": True},
    ) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = "".join(response.iter_text())

    events = [
        block.split("\n")[0][len("event: ") :]
        for block in text.strip().split("\n\n")
        if block.startswith("event: ")
    ]
    assert events[0] == "response.created"
    assert "response.output_text.delta" in events
    assert events[-1] == "response.completed"

    terminal = json.loads(text.strip().split("\n\n")[-1].split("\ndata: ")[1])
    assert terminal["response"]["status"] == "completed"
    assert terminal["response"]["synth"]["proxy_request_ids"]


def test_idempotency_key_replays_instead_of_resampling(client, engine) -> None:
    body = {"input": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    headers = {"Idempotency-Key": "retry-key-1"}

    first = client.post("/v1/responses", json=body, headers=headers)
    count_after_first = len(engine.rollouts)
    second = client.post("/v1/responses", json=body, headers=headers)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.headers.get("Idempotency-Replayed") == "true"
    assert first.json() == second.json()
    # The decisive assertion: the retry produced no second rollout record, so a
    # trainer joining on proxy_request_id cannot double-count the episode.
    assert len(engine.rollouts) == count_after_first


def test_reusing_a_key_for_a_different_body_is_refused(client) -> None:
    headers = {"Idempotency-Key": "retry-key-2"}
    client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": "a"}]},
        headers=headers,
    )
    conflict = client.post(
        "/v1/responses",
        json={"input": [{"role": "user", "content": "b"}]},
        headers=headers,
    )
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["error_code"] == "idempotency_key_conflict"


def test_policy_pin_header_carries_the_snapshot(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    response = client.post(
        "/v1/responses",
        json={"input": "hi"},
        headers={"X-Policy-Pin": snapshot["policy_snapshot_id"]},
    )
    assert response.json()["synth"]["policy_snapshot_id"] == snapshot["policy_snapshot_id"]
    assert response.headers["X-Policy-Pin"] == snapshot["policy_snapshot_id"]

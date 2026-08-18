"""The /v1/synth namespace."""

from __future__ import annotations


def test_capability_reports_both_families_and_the_join_key(client) -> None:
    body = client.get("/v1/synth/capability").json()
    assert body["api_families"] == ["chat_completions", "responses"]
    assert body["token_emission"]["token_ids"] is True
    assert body["token_emission"]["logprobs"] is True
    assert body["rollout_records"]["join_key"] == "proxy_request_id"
    assert body["rollout_records"]["rollout_logprobs_pre_truncation"] is True
    assert body["policy_snapshots"]["pinned_per_request"] is True
    assert "cispo_minimax" in body["objectives"]["supported"]
    assert "cispo_two_sided" in body["objectives"]["supported"]
    assert "ppo" in body["objectives"]["unavailable"]


def test_rollout_retrieval_round_trip(client) -> None:
    chat = client.post(
        "/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    ).json()
    proxy_request_id = chat["synth"]["proxy_request_ids"][0]

    single = client.get(f"/v1/synth/rollouts/{proxy_request_id}")
    assert single.status_code == 200
    record = single.json()["record"]
    assert record["proxy_request_id"] == proxy_request_id
    assert len(record["rollout_logprobs"]) == len(record["completion_token_ids"])

    batch = client.post(
        "/v1/synth/rollouts/query",
        json={"proxy_request_ids": [proxy_request_id, "prid_missing"]},
    ).json()
    assert len(batch["records"]) == 1
    assert batch["missing"] == ["prid_missing"]


def test_missing_rollout_is_404_not_an_empty_record(client) -> None:
    response = client.get("/v1/synth/rollouts/prid_nope")
    assert response.status_code == 404
    assert response.json()["detail"]["error_code"] == "rollout_record_not_found"


def test_snapshot_lifecycle_over_http(client) -> None:
    listing = client.get("/v1/synth/snapshots").json()
    assert listing["latest_policy_snapshot_id"] is not None

    published = client.post(
        "/v1/synth/snapshots", json={"metadata": {"reason": "test"}}
    ).json()
    assert published["metadata"] == {"reason": "test"}
    assert (
        client.get("/v1/synth/snapshots").json()["latest_policy_snapshot_id"]
        == published["policy_snapshot_id"]
    )

    fetched = client.get(
        f"/v1/synth/snapshots/{published['policy_snapshot_id']}"
    ).json()
    assert fetched["policy_snapshot_id"] == published["policy_snapshot_id"]
    # No weights on the wire, ever.
    assert "payload" not in fetched

    client.delete(f"/v1/synth/snapshots/{published['policy_snapshot_id']}")
    gone = client.get(f"/v1/synth/snapshots/{published['policy_snapshot_id']}")
    assert gone.status_code == 410


def test_a_reused_snapshot_id_is_refused(client) -> None:
    client.post("/v1/synth/snapshots", json={"snapshot_id": "candidate-a"})
    again = client.post("/v1/synth/snapshots", json={"snapshot_id": "candidate-a"})
    assert again.status_code == 409


def test_logprobs_endpoint_pins_a_snapshot(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    response = client.post(
        "/v1/synth/logprobs",
        json={
            "token_ids": [1, 2, 3],
            "policy_snapshot_id": snapshot["policy_snapshot_id"],
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["logprobs"][0] is None
    assert body["policy_snapshot_id"] == snapshot["policy_snapshot_id"]


def test_logprobs_against_an_evicted_snapshot_is_refused(client) -> None:
    snapshot = client.post("/v1/synth/snapshots", json={}).json()
    client.delete(f"/v1/synth/snapshots/{snapshot['policy_snapshot_id']}")
    response = client.post(
        "/v1/synth/logprobs",
        json={
            "token_ids": [1, 2, 3],
            "policy_snapshot_id": snapshot["policy_snapshot_id"],
        },
    )
    assert response.status_code == 410

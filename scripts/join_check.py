"""Prove the join resolves: container trace -> proxy token record.

The proxy owns the authoritative token ids and rollout logprobs; the container
owns the reward. The `token_capture` event the container seals is the only
thing connecting them, so this asserts the id actually resolves and that the
snapshot the container recorded is the snapshot the proxy sampled under.

Run `uv run python scripts/serve_fake.py` here first, then run this from a
containers checkout.
"""
import json, os, urllib.request
from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app

PORT, TELEMETRY = 8791, {"enabled": True, "transport": "sse", "retention": "run"}
os.environ.setdefault("SYNTH_MLX_RL_API_KEY", "local-dev-token")
client = TestClient(create_compat_app("gsm8k_solve"))

for family in ("chat_completions", "responses"):
    cid = f"j_{family}"
    client.post("/policy-configs", json={"config_id": cid, "harness": "solve", "config": {
        "provider": "synth_mlx_rl", "model": "fake/Qwen3.5-0.8B", "api_family": family,
        "base_url": f"http://127.0.0.1:{PORT}/v1", "api_key_env": "SYNTH_MLX_RL_API_KEY",
        "max_tokens": 24}})
    rid = f"join_{family}"
    client.post("/rollouts/prepare", json={"rollout_id": rid, "telemetry": TELEMETRY})
    client.post("/rollouts", json={"rollout_id": rid, "telemetry": TELEMETRY, "slot": "stream",
        "world_ref": "world:gsm8k@heldout", "task_instance_id": "seed:0",
        "policy_ref": {"harness": "solve", "config": cid}})
    reward = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    events = client.get(f"/rollouts/{rid}/events", params={"after": 0}).json()["events"]
    cap = next((e["payload"] for e in events if e["kind"] == "token_capture"), None)
    assert cap, f"{family}: no token_capture sealed"

    prid = cap["proxy_request_ids"][0]
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/synth/rollouts/{prid}", timeout=10) as r:
        rec = json.loads(r.read())["record"]

    assert rec["policy_snapshot_id"] == cap["policy_snapshot_id"], "snapshot mismatch"
    assert len(rec["completion_token_ids"]) == len(rec["rollout_logprobs"])
    print(f"{family:18s} reward={reward['reward']} prid={prid[:18]}.. "
          f"tokens={len(rec['completion_token_ids'])} logprobs={len(rec['rollout_logprobs'])} "
          f"snapshot={rec['policy_snapshot_id'][:14]}..")

print("\nJOIN RESOLVES: container reward <-> proxy token record, both families")

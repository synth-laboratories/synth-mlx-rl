"""Cross-repo smoke: containers GSM8K -> synth-mlx-rl proxy, both api families.

No MLX, no Docker. Proves the wire the plan describes: the container calls the
local proxy, the proxy answers and records the rollout server-side, and the
container seals a reward it computed itself.
"""
import json, os, urllib.request
from fastapi.testclient import TestClient
from synth_containers.platform import create_compat_app

PORT = 8791
BASE = f"http://127.0.0.1:{PORT}/v1"
TELEMETRY = {"enabled": True, "transport": "sse", "retention": "run"}

def prepare_start(client, rollout_id, body):
    assert client.post("/rollouts/prepare", json={"rollout_id": rollout_id, "telemetry": TELEMETRY}).status_code == 200
    r = client.post("/rollouts", json={"rollout_id": rollout_id, "telemetry": TELEMETRY, "slot": "stream", **body})
    assert r.status_code == 200, r.text
    return r.json()

def events(client, rollout_id):
    return client.get(f"/rollouts/{rollout_id}/events", params={"after": 0}).json()["events"]

os.environ.setdefault("SYNTH_MLX_RL_API_KEY", "local-dev-token")
client = TestClient(create_compat_app("gsm8k_solve"))

results = {}
for family in ("chat_completions", "responses"):
    cfg_id = f"mlx_{family}"
    resp = client.post("/policy-configs", json={
        "config_id": cfg_id, "harness": "solve",
        "config": {
            "provider": "synth_mlx_rl",
            "model": "fake/Qwen3.5-0.8B",
            "api_family": family,
            "base_url": BASE,
            "api_key_env": "SYNTH_MLX_RL_API_KEY",
            "max_tokens": 24,
        },
    })
    assert resp.status_code == 200, resp.text

    rid = f"gsm8k_{family}"
    prepare_start(client, rid, {
        "world_ref": "world:gsm8k@heldout",
        "task_instance_id": "seed:0",
        "policy_ref": {"harness": "solve", "config": cfg_id},
    })
    reward = client.post("/reward", json={"rollout_id": rid, "mode": "terminal"}).json()
    evs = events(client, rid)
    action = next((e for e in evs if e["kind"] == "action"), None)
    status = next((e for e in evs if e["kind"] == "status"), None)
    results[family] = {
        "reward": reward.get("reward"),
        "reward_status": reward.get("status"),
        "action": action["payload"] if action else None,
        "rollout_status": status["payload"] if status else None,
    }

print(json.dumps(results, indent=2))

for family, row in results.items():
    assert row["rollout_status"]["status"] == "completed", family
    # A wrong answer scores 0.0. It must not be null: the eval layer drops null
    # metrics from the denominator, which would delete a bad model's worst trials.
    assert row["reward"] == 0.0 and row["reward_status"] == "scored", family
    assert row["action"]["parse_status"] == "parsed", family
print("\nboth api families: container -> proxy -> reward, OK")

# Run with:
#   1. from this repo:      uv run python scripts/serve_fake.py     (port 8791)
#   2. from a containers checkout: uv run python <this file>
#
# The join is now closed: the container seals a `token_capture` event carrying
# the proxy_request_ids, and that id resolves against the proxy's server-side
# token record. See scripts/join_check.py.

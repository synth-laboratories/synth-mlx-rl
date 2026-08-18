"""The logprob lifecycle, end to end, against a running service.

    sample -> record -> recompute behavior -> align -> measure -> verdict -> TIS

Run `uv run python scripts/serve_fake.py` first, then this.
"""
import json, sys, urllib.request

BASE = "http://127.0.0.1:8791"


def post(path, payload):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=20) as response:
        return json.loads(response.read())


print("1. sample")
sampled = post("/v1/chat/completions", {
    "model": "fake/Qwen3.5-0.8B",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 12})
synth = sampled["synth"]
prid, snapshot = synth["proxy_request_ids"][0], synth["policy_snapshot_id"]
print(f"   prid={prid[:20]}.. snapshot={snapshot[:18]}..")

print("2. server-side record (the training authority)")
record = get(f"/v1/synth/rollouts/{prid}")["record"]
n = len(record["completion_token_ids"])
assert n == len(record["rollout_logprobs"]), "record tokens/logprobs misaligned"
print(f"   {n} completion tokens, {n} rollout_logprobs, aligned")

print("3. recompute behavior + measure mismatch")
result = post("/v1/synth/mismatch", {"proxy_request_id": prid})
report = result["report"]
assert len(result["behavior_logprobs"]) == n, "behavior logprobs misaligned to completion"
assert result["policy_snapshot_id"] == snapshot, "compared under the wrong snapshot"
print(f"   max|behavior-rollout| = {report['train_rollout_logprob_abs_diff']:.3e}")
print(f"   ess_ratio = {report['ess_ratio']:.4f}   nonfinite = {report['nonfinite_token_count']}")
print(f"   verdict = {report['verdict']}")

print("4. verdict routing")
if report["verdict"] == "ok":
    assert result["tis_weights"] is None
    print("   agree; train as collected, no weights handed back")
elif report["verdict"] == "correct_with_tis":
    assert result["tis_weights"] is not None
    print(f"   TIS weights returned ({len(result['tis_weights'])})")
else:
    print("   REFUSED:", report["reason"])

print("5. a forced-strict policy must route to TIS, proving the bound is live")
strict = post("/v1/synth/mismatch",
              {"proxy_request_id": prid, "ok_abs_diff": -1.0, "max_abs_diff": 1e9})
assert strict["report"]["verdict"] == "correct_with_tis"
assert strict["tis_weights"] is not None and len(strict["tis_weights"]) == n
print(f"   verdict={strict['report']['verdict']} weights={len(strict['tis_weights'])}")

print("6. a refusing policy must refuse")
refused = post("/v1/synth/mismatch",
               {"proxy_request_id": prid, "max_abs_diff": -1.0})
assert refused["report"]["verdict"] == "refuse"
assert refused["tis_weights"] is None, "a refused batch must not hand back weights"
print(f"   verdict={refused['report']['verdict']}, no weights handed back")

print("\nLIFECYCLE OK: sample -> record -> behavior -> mismatch -> verdict -> TIS")

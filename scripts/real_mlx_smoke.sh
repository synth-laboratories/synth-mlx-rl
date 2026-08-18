#!/usr/bin/env bash
# Milestone 0, not a regression check.
#
# Nothing in the MLX path of this package has ever executed: it was seeded from
# a prototype that was only ever run in a Linux packaging environment where MLX
# and Metal cannot execute (finalized plan, correction C11). The first time this
# script passes is the first time any of that code has run at all.
#
# Adapted from the MIT-licensed `mlx-local-rl` prototype's smoke script.
set -euo pipefail

cd "$(dirname "$0")/.."

PORT="${PORT:-8787}"
# Qwen/Qwen3.5-0.8B is verified present in the local Hugging Face cache. The
# prototype's `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` default is an unverified
# repo id; confirm it resolves before pinning it here.
MODEL="${MODEL:-Qwen/Qwen3.5-0.8B}"
LOG_FILE="${TMPDIR:-/tmp}/synth-mlx-rl-smoke.log"
BASE="http://127.0.0.1:${PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

synth-mlx-rl serve \
  --model "$MODEL" \
  --lora-rank 8 \
  --lora-alpha 16 \
  --num-layers 4 \
  --max-seq-length 1024 \
  --no-thinking \
  --clear-cache-every 1 \
  --port "$PORT" >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

healthy=0
for _ in $(seq 1 180); do
  if curl -fsS "${BASE}/healthz" >/dev/null 2>&1; then
    healthy=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    break
  fi
  sleep 1
done

if [[ "$healthy" != "1" ]]; then
  cat "$LOG_FILE"
  exit 1
fi

echo "--- MLX-only unit tests ---"
python -m pytest -q -m mlx

echo "--- capability ---"
curl -fsS "${BASE}/v1/synth/capability"

echo
echo "--- both API families, same conversation, one render digest ---"
python - "$BASE" <<'PY'
import json
import sys
import urllib.request

base = sys.argv[1]


def post(path, payload):
    request = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


chat = post(
    "/v1/chat/completions",
    {
        "messages": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "say hello"},
        ],
        "temperature": 0.0,
        "max_tokens": 16,
    },
)
responses = post(
    "/v1/responses",
    {
        "instructions": "be terse",
        "input": [{"role": "user", "content": "say hello"}],
        "temperature": 0.0,
        "max_output_tokens": 16,
    },
)

assert chat["synth"]["render_digest"] == responses["synth"]["render_digest"], (
    "the two API families rendered different transcripts; every mixed-family "
    "comparison would be invalid"
)
print("render digest matches across families:", chat["synth"]["render_digest"][:16])

record = urllib.request.urlopen(
    base + "/v1/synth/rollouts/" + chat["synth"]["proxy_request_ids"][0]
)
record = json.load(record)["record"]
assert len(record["rollout_logprobs"]) == len(record["completion_token_ids"])
print("rollout record aligned:", len(record["completion_token_ids"]), "tokens")
PY

echo "--- one SFT step and one CISPO step ---"
python examples/sft_identity.py --base-url "$BASE" --steps 1 --checkpoint smoke-sft
python examples/cispo_smoke.py --base-url "$BASE"

curl -fsS "${BASE}/v1/state"
printf '\nReal MLX smoke test passed.\n'

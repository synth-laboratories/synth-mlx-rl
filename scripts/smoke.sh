#!/usr/bin/env bash
# The portable smoke test. Runs anywhere: no MLX, no Apple Silicon, no GPU.
#
# MLX cannot be installed on the machine this package is developed on, so this
# is the suite that actually gates changes. Everything MLX-specific is marked
# `mlx` and skips cleanly.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m compileall -q src tests
python -m pytest -q

printf '\nPortable suite passed with no MLX import.\n'
printf 'For the first real MLX run on Apple Silicon:\n'
printf '  bash scripts/real_mlx_smoke.sh\n'

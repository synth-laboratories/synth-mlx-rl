"""Build a GSM8K SFT set from the container's own pinned world, not a lookalike.

Mirrors `banking77_sft_dataset.py`: the prompt comes from
`synth_containers.platform.gsm8k_world`, the module the container uses to build
the observation it sends at rollout time, so the policy trains on the format it
is scored on. What GSM8K adds is the pin: the world module names one
`openai/gsm8k` revision and a digest per split, and it refuses rows that do not
reproduce them, so the JSONL written here is traceable to exact bytes.

The profile is declared in code (`declare_profile("hf")`), never through an
environment variable, and the output files are named by their own content
digest; `manifest.json` records the dataset pin, the seeds each file covers,
and every file's sha256, so a training run can cite what it was fed.

    python scripts/gsm8k_sft_dataset.py --out .gsm8k --train 256 --eval 64 \
        --containers-src ../containers/src

Set HF_HUB_OFFLINE=1 to refuse any download: the pinned revision must then
already be in the local Hugging Face cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

SCHEMA = "gsm8k.sft-dataset.v1"
CANONICAL_PIN_KEYS = ("dataset", "config", "revision", "splits", "shuffle_seed", "profile", "profile_source")


def load_world(containers_src: Path):
    sys.path.insert(0, str(containers_src.resolve()))
    from synth_containers.platform import gsm8k_world  # noqa: E402

    return gsm8k_world


def example(world, row) -> dict[str, Any]:
    """One chat example in exactly the container's prompt shape."""
    return {
        "messages": [
            {"role": "system", "content": world.SOLVE_SYSTEM},
            {"role": "user", "content": world.user_prompt(row.question)},
            {"role": "assistant", "content": row.answer_text},
        ]
    }


def write_split(world, *, split: str, count: int, out: Path, label: str) -> dict[str, Any]:
    """Write ``count`` rows of ``split`` (seeds 0..count-1) to a digest-named file."""
    rows: list[str] = []
    seeds: list[int] = []
    for seed in range(count):
        row = world.load_row(split, seed)
        if row is None:
            break
        if not row.answer:
            raise SystemExit(f"{split} seed {seed}: the reference answer does not parse; refusing")
        rows.append(json.dumps(example(world, row), ensure_ascii=False))
        seeds.append(seed)
    if len(seeds) < count:
        raise SystemExit(f"{split} has only {len(seeds)} rows; {count} were requested")
    payload = ("\n".join(rows) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = out / f"gsm8k-{label}-{len(seeds)}-{digest[:12]}.jsonl"
    path.write_bytes(payload)
    return {
        "path": path.name,
        "split": split,
        "hf_split": world.SPLIT_PINS[split].hf_split,
        "rows": len(seeds),
        "seeds": {"first": seeds[0], "last": seeds[-1]},
        "sha256": f"sha256:{digest}",
        "bytes": len(payload),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(".gsm8k"))
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--eval", type=int, default=64)
    parser.add_argument(
        "--containers-src",
        type=Path,
        default=Path("../containers/src"),
        help="the containers checkout whose prompt and dataset pin this must match",
    )
    args = parser.parse_args(argv)

    world = load_world(args.containers_src)
    world.declare_profile("hf")  # in code; the pin itself lives in the world module
    pin = world.dataset_manifest()
    if not pin.get("pinned"):
        raise SystemExit(f"the world reports an unpinned profile ({pin.get('profile')!r}); refusing")

    args.out.mkdir(parents=True, exist_ok=True)
    train = write_split(world, split=world.TRAIN_SPLIT, count=args.train, out=args.out, label="train")
    held = write_split(world, split=world.HELDOUT_SPLIT, count=args.eval, out=args.out, label="eval")
    manifest = {
        "schema_version": SCHEMA,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": {key: pin[key] for key in CANONICAL_PIN_KEYS},
        "prompt": {
            "source": "synth_containers.platform.gsm8k_world",
            "system_sha256": "sha256:" + hashlib.sha256(world.SOLVE_SYSTEM.encode("utf-8")).hexdigest(),
        },
        "files": {"train": train, "eval": held},
        "containers_src": str(args.containers_src.resolve()),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{train['rows']} train rows -> {args.out / train['path']}")
    print(f"{held['rows']} heldout rows -> {args.out / held['path']}")
    print(
        f"openai/gsm8k @ {pin['revision'][:12]}  train {pin['splits'][world.TRAIN_SPLIT]['digest'][:19]}..  "
        f"heldout {pin['splits'][world.HELDOUT_SPLIT]['digest'][:19]}..  shuffle_seed={pin['shuffle_seed']}"
    )
    print(f"manifest -> {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

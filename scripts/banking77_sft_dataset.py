"""Build a Banking77 SFT set from the container's own prompt, not a lookalike.

The point of warm-starting before the on-policy lane is to make the policy
capable of *emitting a valid label at all*. Qwen3.5-0.8B understands the task --
asked to classify a card-fraud query it answers "fraud" -- but that is not one
of the 77 labels, so exact-match scores it zero, every rollout in a group scores
zero, and CISPO correctly refuses to train on a group with no reward variance.
There is no gradient to find until the policy sometimes succeeds.

The prompts here come from `synth_containers.platform.banking77_world`, the same
module the container uses to build the observation it will send at rollout time.
Writing a lookalike prompt would train the policy on a format it never sees in
the environment, and the whole exercise would measure train/serve skew instead
of learning.

    python scripts/banking77_sft_dataset.py --out .banking77 --train 256
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(".banking77"))
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--eval", type=int, default=32)
    parser.add_argument(
        "--containers-src",
        type=Path,
        default=Path("../containers/src"),
        help="the containers checkout whose prompt this must match",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.containers_src.resolve()))
    from synth_containers.platform.banking77_world import (  # noqa: E402
        CLASSIFY_SYSTEM,
        TRAIN_SPLIT,
        HELDOUT_SPLIT,
        load_row,
        label_vocabulary,
        user_prompt,
    )

    labels = label_vocabulary()
    args.out.mkdir(parents=True, exist_ok=True)

    def write(split: str, count: int, path: Path) -> int:
        written = 0
        with path.open("w", encoding="utf-8") as handle:
            for seed in range(count * 4):
                if written >= count:
                    break
                row = load_row(split, seed)
                if row is None:
                    break
                handle.write(
                    json.dumps(
                        {
                            "messages": [
                                {"role": "system", "content": CLASSIFY_SYSTEM},
                                {
                                    "role": "user",
                                    "content": user_prompt(row.text, labels=labels),
                                },
                                {"role": "assistant", "content": row.label},
                            ]
                        }
                    )
                    + "\n"
                )
                written += 1
        return written

    train_path = args.out / "train.jsonl"
    eval_path = args.out / "eval.jsonl"
    trained = write(TRAIN_SPLIT, args.train, train_path)
    evaluated = write(HELDOUT_SPLIT, args.eval, eval_path)
    print(f"{trained} train rows -> {train_path}")
    print(f"{evaluated} heldout rows -> {eval_path}")
    print(f"label space: {len(labels)} labels; prompt from the container's own world module")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""A tiny end-to-end LoRA SFT run.

Adapted from the MIT-licensed `mlx-local-rl` prototype (see NOTICE). Changed:
the sampler is a snapshot-pinned client, so the "before" and "after" probes are
demonstrably taken against different frozen policies rather than against
whatever the trainer happened to hold.
"""

from __future__ import annotations

import argparse
import json

from synth_mlx_rl import AdamParams, SamplingParams, ServiceClient

TRAINING_CONVERSATIONS = [
    [
        {"role": "system", "content": "Answer exactly and briefly."},
        {"role": "user", "content": "What backend are you running on?"},
        {"role": "assistant", "content": "LOCAL_MLX"},
    ],
    [
        {"role": "system", "content": "Answer exactly and briefly."},
        {"role": "user", "content": "Say the local learner readiness token."},
        {"role": "assistant", "content": "LOCAL_MLX_READY"},
    ],
    [
        {"role": "system", "content": "Answer exactly and briefly."},
        {"role": "user", "content": "Which adapter type is active?"},
        {"role": "assistant", "content": "LoRA"},
    ],
]

PROBE = [
    {"role": "system", "content": "Answer exactly and briefly."},
    {"role": "user", "content": "Say the local learner readiness token."},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny end-to-end LoRA SFT run")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--checkpoint", default="sft-identity")
    args = parser.parse_args()

    with ServiceClient(args.base_url) as service:
        trainer = service.create_lora_training_client()
        datums = [
            trainer.datum_from_messages(messages, assistant_only=True)
            for messages in TRAINING_CONVERSATIONS
        ]

        baseline = trainer.save_weights_and_get_sampling_client()
        before = baseline.sample(
            PROBE, SamplingParams(max_tokens=32, temperature=0.0)
        ).result()
        print(
            json.dumps(
                {
                    "before": before.samples[0].text,
                    "snapshot": baseline.policy_snapshot_id,
                },
                ensure_ascii=False,
            )
        )

        optimizer = AdamParams(
            learning_rate=args.learning_rate, weight_decay=0.0, max_grad_norm=1.0
        )
        for step in range(1, args.steps + 1):
            forward_future = trainer.forward_backward(datums, "cross_entropy")
            optim_future = trainer.optim_step(optimizer)
            forward = forward_future.result()
            update = optim_future.result()
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": forward.loss,
                        "tokens": forward.metrics["token_count"],
                        "grad_norm": update.grad_norm,
                        "training_version": update.training_version,
                    }
                )
            )

        trained = trainer.save_weights_and_get_sampling_client(args.checkpoint)
        after = trained.sample(
            PROBE, SamplingParams(max_tokens=32, temperature=0.0)
        ).result()
        print(
            json.dumps(
                {
                    "after": after.samples[0].text,
                    "snapshot": trained.policy_snapshot_id,
                },
                ensure_ascii=False,
            )
        )
        # The baseline snapshot is still resident and still frozen: sampling it
        # again after training reproduces the original completion.
        recheck = baseline.sample(
            PROBE, SamplingParams(max_tokens=32, temperature=0.0)
        ).result()
        print(
            json.dumps(
                {
                    "baseline_recheck": recheck.samples[0].text,
                    "pin_held": recheck.samples[0].text == before.samples[0].text,
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

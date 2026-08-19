#!/usr/bin/env python3
"""One on-policy update through each shipped policy objective.

A runtime-path smoke test, not a reward-learning experiment. It exists to prove
the MLX backward pass runs for every objective name the service advertises --
including that `cispo_minimax` refuses an active lower clip bound at the
service boundary and not only in a unit test.

The behavior log-probabilities come from a trainer forward pass at the pinned
snapshot (`/v1/synth/logprobs`), not from the sampler's rollout values. Those
are different populations, and using the second in place of the first is the
quiet way to get a ratio denominator that does not mean what the report says.
"""

from __future__ import annotations

import argparse
import json

import httpx

from synth_mlx_rl import AdamParams, SamplingParams, ServiceClient

OBJECTIVES = (
    ("importance_sampling", {}),
    ("grpo", {"clip_epsilon": 0.2}),
    ("cispo_minimax", {"eps_low": 1.0, "eps_high": 4.0}),
    ("cispo_two_sided", {"eps_low": 0.2, "eps_high": 0.28}),
)

MESSAGES = [
    {
        "role": "user",
        "content": "Return one short token that could label a successful smoke test.",
    }
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Exercise every policy objective")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=7000)
    args = parser.parse_args()

    optimizer = AdamParams(
        learning_rate=args.learning_rate, weight_decay=0.0, max_grad_norm=1.0
    )

    with ServiceClient(args.base_url) as service:
        trainer = service.create_lora_training_client()

        for index, (objective, options) in enumerate(OBJECTIVES):
            sampler = trainer.save_weights_and_get_sampling_client()
            rollout = sampler.sample(
                MESSAGES,
                SamplingParams(
                    max_tokens=8, temperature=1.0, seed=args.seed + index
                ),
            ).result()
            sample = rollout.samples[0]

            # Behavior log-probabilities: a trainer forward pass at the pinned
            # snapshot, over prompt + completion.
            scored = sampler.compute_logprobs(
                sample.prompt_token_ids + sample.completion_token_ids
            ).result()
            behavior = scored[len(sample.prompt_token_ids) :]

            datum = trainer.datum_from_sample(
                sample,
                advantage=1.0,
                behavior_logprobs=behavior,
                metadata={"smoke_objective": objective},
            )
            forward = trainer.forward_backward([datum], objective, **options).result()
            update = trainer.optim_step(optimizer).result()
            print(
                json.dumps(
                    {
                        "objective": objective,
                        "loss": forward.loss,
                        "token_count": forward.metrics["token_count"],
                        "mean_ratio": forward.metrics["mean_ratio"],
                        "clip_fraction": forward.metrics["clip_fraction"],
                        "clamped": forward.metrics["clamped_token_count"],
                        "grad_norm": update.grad_norm,
                        "training_version": update.training_version,
                    }
                )
            )

        # And the refusal, at the HTTP boundary.
        response = httpx.post(
            args.base_url.rstrip("/") + "/v1/forward_backward",
            json={
                "data": [
                    {
                        "input_ids": [1],
                        "target_ids": [2],
                        "weights": [1.0],
                        "behavior_logprobs": [-1.0],
                        "advantages": 1.0,
                    }
                ],
                "loss_fn": "cispo_minimax",
                "eps_low": 0.2,
            },
        )
        assert response.status_code == 422, response.text
        assert "cispo_two_sided" in response.text
        print(json.dumps({"cispo_minimax_refuses_eps_low_below_one": True}))


if __name__ == "__main__":
    main()

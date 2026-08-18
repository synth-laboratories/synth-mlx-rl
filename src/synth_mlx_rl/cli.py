"""Command line entry point.

Derived from the MIT-licensed `mlx-local-rl` prototype's CLI (see NOTICE).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .config import Settings, parse_lora_keys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="synth-mlx-rl",
        description=(
            "Run the local MLX LoRA learner and its two OpenAI-compatible "
            "sampling surfaces."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve", help="start the FastAPI service")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    serve.add_argument("--log-level", default="info")
    serve.add_argument("--model")
    serve.add_argument("--checkpoint-dir", type=Path)
    serve.add_argument("--adapter-path", type=Path)
    serve.add_argument("--lora-rank", type=int)
    serve.add_argument("--lora-alpha", type=float)
    serve.add_argument("--lora-dropout", type=float)
    serve.add_argument("--num-layers", type=int)
    serve.add_argument(
        "--lora-key",
        action="append",
        dest="lora_keys",
        help=(
            "relative module name to adapt; repeat the flag. By default all "
            "convertible linear/embedding modules in selected transformer "
            "layers are adapted"
        ),
    )
    serve.add_argument("--max-seq-length", type=int)
    serve.add_argument("--max-snapshots", type=int)
    serve.add_argument("--max-rollout-records", type=int)
    serve.add_argument(
        "--thinking",
        dest="enable_thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "default chat-template thinking mode; disabled by default for "
            "compact local SFT/RL rollouts"
        ),
    )
    serve.add_argument(
        "--grad-checkpoint", action=argparse.BooleanOptionalAction, default=None
    )
    serve.add_argument("--clear-cache-every", type=int)
    serve.add_argument("--seed", type=int)
    return parser


def settings_from_args(args: argparse.Namespace) -> Settings:
    base = Settings.from_env()
    overrides: dict[str, Any] = {}
    for field_name in (
        "model",
        "checkpoint_dir",
        "adapter_path",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "num_layers",
        "max_seq_length",
        "max_snapshots",
        "max_rollout_records",
        "enable_thinking",
        "grad_checkpoint",
        "clear_cache_every",
        "seed",
    ):
        value = getattr(args, field_name, None)
        if value is not None:
            overrides[field_name] = value
    if getattr(args, "lora_keys", None) is not None:
        overrides["lora_keys"] = parse_lora_keys(args.lora_keys)
    return base.with_overrides(**overrides)


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "serve":
        import uvicorn

        from .api.app import create_app

        settings = settings_from_args(args)
        uvicorn.run(
            create_app(settings=settings),
            host=args.host,
            port=args.port,
            log_level=args.log_level,
            # One worker. Generation, backward passes, and optimizer updates
            # share one resident model behind one lock; a second worker would
            # be a second model, and snapshot ids would stop meaning anything.
            workers=1,
        )

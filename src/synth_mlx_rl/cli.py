"""Reference local CLI for the v0.6 job API."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from synth_mlx_rl.service import create_app


def main() -> None:
    parser = argparse.ArgumentParser(prog="synth-mlx-rl")
    subcommands = parser.add_subparsers(dest="command", required=True)
    serve = subcommands.add_parser("serve", help="start the local training service")
    serve.add_argument("--root", default=".synth-mlx-rl", help="durable service state directory")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    if args.command == "serve":
        uvicorn.run(create_app(Path(args.root)), host=args.host, port=args.port, log_level="info")

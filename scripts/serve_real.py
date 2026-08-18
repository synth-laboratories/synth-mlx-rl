"""Serve the real MLX engine, guarded.

The engine is NOT built here: `create_app` builds it on its own worker thread,
because MLX streams are thread-local and a model built on the main thread
cannot be evaluated on a request worker.
"""
import sys
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import mlx_guard; mlx_guard.install(3.0)
import uvicorn
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.config import Settings

app = create_app(settings=Settings(lora_rank=8, max_seq_length=1024))
uvicorn.run(app, host="127.0.0.1", port=8791, log_level="warning")

import sys, uvicorn
sys.path.insert(0, "src")  # run from the repo root
from synth_mlx_rl.api.app import create_app
from synth_mlx_rl.testing.fake_engine import FakeEngine
from pathlib import Path
app = create_app(engine=FakeEngine(checkpoint_dir=Path("/tmp/fake-ckpt")))
uvicorn.run(app, host="127.0.0.1", port=8791, log_level="warning")

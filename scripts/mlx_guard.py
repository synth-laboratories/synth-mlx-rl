"""Memory admission, watchdog, and guaranteed reaping for a resident local model.

A resident model is a machine-wide hazard, not a child process. This enforces
the four rules on every real-MLX run in this repo:

  1. admission   refuse to load unless the expected footprint plus headroom fits
  2. watchdog    poll free memory; kill and abort if it collapses
  3. reaping     pidfile, killed on every exit path including SIGINT/SIGTERM
  4. one at a time  refuse if another resident model process is already alive
"""

from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

PIDFILE = Path.home() / ".synth" / "mlx-rl" / "resident.pid"
HEADROOM_GB = 8.0
WATCHDOG_FLOOR_PCT = 10.0


def _free_pct() -> float:
    out = subprocess.run(
        ["memory_pressure", "-Q"], capture_output=True, text=True, timeout=10
    ).stdout
    for line in out.splitlines():
        if "free percentage" in line:
            return float(line.rsplit(":", 1)[1].strip().rstrip("%"))
    raise RuntimeError("could not read memory pressure")


def _total_gb() -> float:
    out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return int(out.stdout.strip()) / 1024**3


def _swap_used_mb() -> float:
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    for part in out.split():
        if part.endswith("M") and "used" in out.split(part)[0][-8:]:
            return float(part.rstrip("M"))
    return 0.0


def admit(expected_gb: float) -> None:
    """Refuse the load rather than discovering the problem at 90% swap."""
    free_gb = _total_gb() * _free_pct() / 100.0
    need = expected_gb + HEADROOM_GB
    if free_gb < need:
        raise MemoryError(
            f"refusing to load: {free_gb:.1f} GB free, need {expected_gb:.1f} GB "
            f"+ {HEADROOM_GB:.0f} GB headroom"
        )
    stale = PIDFILE.read_text().strip() if PIDFILE.exists() else ""
    if stale:
        try:
            os.kill(int(stale), 0)
        except (ProcessLookupError, ValueError):
            PIDFILE.unlink(missing_ok=True)  # reap a leftover from a prior run
        else:
            raise RuntimeError(f"another resident model is alive (pid {stale})")
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    print(f"[guard] admitted: {free_gb:.1f} GB free, expect ~{expected_gb:.1f} GB, pid {os.getpid()}")


def _release(*_: object) -> None:
    try:
        if PIDFILE.exists() and PIDFILE.read_text().strip() == str(os.getpid()):
            PIDFILE.unlink()
    except OSError:
        pass


def start_watchdog() -> None:
    def loop() -> None:
        while True:
            time.sleep(2.0)
            try:
                free = _free_pct()
            except Exception:
                continue
            if free < WATCHDOG_FLOOR_PCT:
                print(f"[guard] FREE MEMORY {free:.0f}% -- killing self", file=sys.stderr)
                _release()
                os.kill(os.getpid(), signal.SIGKILL)

    threading.Thread(target=loop, daemon=True).start()


def install(expected_gb: float) -> None:
    admit(expected_gb)
    atexit.register(_release)
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous = signal.getsignal(sig)

        def handler(signum, frame, _previous=previous):  # noqa: ANN001
            _release()
            if callable(_previous):
                _previous(signum, frame)
            raise SystemExit(128 + signum)

        signal.signal(sig, handler)
    start_watchdog()

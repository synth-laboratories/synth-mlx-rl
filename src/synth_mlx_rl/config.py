"""Service configuration.

Derived from the MIT-licensed `mlx-local-rl` prototype (see NOTICE), with the
environment-variable prefix renamed and the model pin corrected.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

#: Verified present in the local Hugging Face cache. The prototype's default
#: (`mlx-community/Qwen3.5-0.8B-OptiQ-4bit`) is an unverified repo id and is
#: deliberately not pinned here; see the finalized plan, correction C12.
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"

ENV_PREFIX = "SYNTH_MLX_RL_"


def _valid_model_directory(path: Path) -> bool:
    """Whether ``path`` contains the files MLX-LM needs for this model."""

    path = path.expanduser()
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and (
            any(path.glob("*.safetensors"))
            or any(path.glob("*.bin"))
        )
    )


def _cached_model_directory(model: str) -> Path | None:
    """Resolve an already-downloaded HF snapshot without contacting the Hub."""

    cache_root = Path(
        os.getenv("HF_HUB_CACHE")
        or Path(os.getenv("HF_HOME", "~/.cache/huggingface")).expanduser() / "hub"
    ).expanduser()
    repository = cache_root / ("models--" + model.replace("/", "--"))
    snapshots = repository / "snapshots"
    reference = repository / "refs" / "main"
    candidates: list[Path] = []
    if reference.is_file():
        revision = reference.read_text(encoding="utf-8").strip()
        if revision:
            candidates.append(snapshots / revision)
    if snapshots.is_dir():
        candidates.extend(
            sorted(
                (path for path in snapshots.iterdir() if path.is_dir()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    return next((path.resolve() for path in candidates if _valid_model_directory(path)), None)


def _env(name: str) -> str | None:
    return os.getenv(ENV_PREFIX + name)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


def parse_lora_keys(values: Iterable[str] | None) -> tuple[str, ...] | None:
    if values is None:
        return None
    keys = tuple(v.strip() for v in values if v.strip())
    return keys or None


@dataclass(frozen=True, slots=True)
class Settings:
    model: str = DEFAULT_MODEL
    #: Workshop-managed HF snapshot. This is a load location, never a second
    #: model identity: the admitted policy remains ``DEFAULT_MODEL``.
    model_path: Path | None = None
    checkpoint_dir: Path = Path("./checkpoints")
    adapter_path: Path | None = None
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    num_layers: int = -1
    lora_keys: tuple[str, ...] | None = None
    max_seq_length: int = 4096
    enable_thinking: bool = False
    # On by default. Measured on Qwen3.5-0.8B, one 768-token datum:
    #   off:  33.40 GB peak,  9.8 train tok/s
    #   on :   7.30 GB peak, 35.0 train tok/s
    # It is faster AND smaller. Recomputation is supposed to trade time for
    # memory, but retained activations across 28 layers cost ~42 MB/token, and
    # at 1024 tokens peak reached 44 GB against Metal's 51.8 GB recommended
    # working set -- so the allocator thrashed and time went superlinear. Below
    # the working set the recompute is cheaper than the thrash it avoids.
    grad_checkpoint: bool = True
    clear_cache_every: int = 1
    seed: int = 0
    #: How many frozen sampling snapshots stay resident. One resident base model
    #: plus a pool of adapter copies; see decision D2.
    max_snapshots: int = 4
    #: How many server-side rollout records stay retrievable.
    max_rollout_records: int = 4096

    @property
    def lora_scale(self) -> float:
        return self.lora_alpha / self.lora_rank

    def validated(self) -> "Settings":
        if self.model != DEFAULT_MODEL:
            raise ValueError(f"the bounded recipe admits exactly {DEFAULT_MODEL}")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1)")
        if self.num_layers == 0 or self.num_layers < -1:
            raise ValueError("num_layers must be -1 (all layers) or a positive integer")
        if self.max_seq_length < 8:
            raise ValueError("max_seq_length must be at least 8")
        if self.clear_cache_every < 0:
            raise ValueError("clear_cache_every must be non-negative")
        if self.max_snapshots < 1:
            raise ValueError("max_snapshots must be at least 1")
        if self.max_rollout_records < 1:
            raise ValueError("max_rollout_records must be at least 1")
        return self

    def local_model_path(self) -> Path | None:
        """Prefer Workshop's managed snapshot, then an existing HF cache."""

        if self.model_path is not None:
            managed = self.model_path.expanduser()
            if _valid_model_directory(managed):
                return managed.resolve()
        return _cached_model_directory(self.model)

    def require_local_model_path(self) -> Path:
        path = self.local_model_path()
        if path is None:
            raise FileNotFoundError(
                f"{DEFAULT_MODEL} is not available locally. Download it in "
                "Workshop Settings → Models → On-device training, then pass "
                f"{ENV_PREFIX}MODEL_PATH as the downloaded snapshot directory."
            )
        return path

    @classmethod
    def from_env(cls) -> "Settings":
        adapter = _env("ADAPTER_PATH")
        model_path = _env("MODEL_PATH")
        keys = _env("LORA_KEYS")
        return cls(
            model=_env("MODEL") or DEFAULT_MODEL,
            model_path=Path(model_path) if model_path else None,
            checkpoint_dir=Path(_env("CHECKPOINT_DIR") or "./checkpoints"),
            adapter_path=Path(adapter) if adapter else None,
            lora_rank=_env_int("LORA_RANK", 8),
            lora_alpha=_env_float("LORA_ALPHA", 16.0),
            lora_dropout=_env_float("LORA_DROPOUT", 0.0),
            num_layers=_env_int("NUM_LAYERS", -1),
            lora_keys=parse_lora_keys(keys.split(",") if keys else None),
            max_seq_length=_env_int("MAX_SEQ_LENGTH", 4096),
            enable_thinking=_env_bool("ENABLE_THINKING", False),
            grad_checkpoint=_env_bool("GRAD_CHECKPOINT", True),
            clear_cache_every=_env_int("CLEAR_CACHE_EVERY", 1),
            seed=_env_int("SEED", 0),
            max_snapshots=_env_int("MAX_SNAPSHOTS", 4),
            max_rollout_records=_env_int("MAX_ROLLOUT_RECORDS", 4096),
        ).validated()

    def with_overrides(self, **kwargs: object) -> "Settings":
        return replace(self, **kwargs).validated()

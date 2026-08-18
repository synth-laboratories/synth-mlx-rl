"""Content-addressed adapter store and local run records.

The identity of a LoRA is the sha256 of its directory contents, and that one
identity is used on all three surfaces: it is the ``policy_snapshot_id`` the
proxy pins per request, the ``artifact_digest`` an Optimizers eval candidate
carries into its sealed manifest, and the key of this store. Asking "which run
produced this adapter" is therefore a lookup, never a reconstruction.

Two properties are load-bearing rather than stylistic:

- **Keyed by digest, not by a caller-supplied name.** The upstream prototype
  wrote ``checkpoints/<name>/`` from ``save_state("experiment-0042")``. A name
  is exactly how provenance is lost: two runs can reuse one, and a name cannot
  verify the bytes it points at.
- **A run references adapters; it does not own them.** ``checkpoints.jsonl`` is
  a list of ``(step, adapter_digest)`` pointers, so a warm start does not
  duplicate bytes and two runs starting from one adapter share its entry.
  ``parent_digest`` then makes base -> SFT -> RL a reconstructible chain;
  recording only ``run_id`` would answer one hop and lose the ancestry, which
  is the question actually asked when a number looks wrong.

``digest_of_tree`` deliberately reproduces ``synth_optimizers.eval.models``
byte for byte. If the two ever diverge the join key silently stops joining, so
``tests/test_store.py`` pins the algorithm against a hand-computed vector.
"""

from __future__ import annotations

import hashlib
import json
import stat
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "AdapterRecord",
    "AdapterStore",
    "RunRecord",
    "StoreConfig",
    "StoreError",
    "canonical_json",
    "digest_of",
    "digest_of_tree",
]

ADAPTER_REQUIRED_FILES: tuple[str, ...] = ("adapter_config.json", "adapters.safetensors")
DEFAULT_STORE_ROOT = Path("~/.synth/mlx-rl")


class StoreError(RuntimeError):
    """An invalid store operation. Never raised for a merely absent entry."""


def canonical_json(value: Any) -> str:
    """Stable JSON used wherever a digest must be reproducible."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest_of(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def digest_of_tree(root: Path) -> str:
    """Content address a directory: sorted relative path + bytes.

    Byte-identical to ``synth_optimizers.eval.models.digest_of_tree``. Do not
    "improve" this independently of that one.
    """

    if root.is_file():
        entries = [(root.name, root)]
    else:
        entries = sorted(
            ((str(p.relative_to(root)), p) for p in root.rglob("*") if p.is_file()),
            key=lambda item: item[0],
        )
    hasher = hashlib.sha256()
    for relative, path in entries:
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(path.read_bytes()).digest())
    return "sha256:" + hasher.hexdigest()


def _digest_dirname(digest: str) -> str:
    if not digest.startswith("sha256:") or len(digest) != 71:
        raise StoreError(f"not a tree digest: {digest!r}")
    return digest.replace(":", "-", 1)


@dataclass(frozen=True, slots=True)
class StoreConfig:
    """Store settings. TOML, never environment variables.

    A path that lives in the environment cannot be read back off a finished
    run, which is the whole reason a run record exists.
    """

    root: Path = DEFAULT_STORE_ROOT
    max_adapters: int | None = None

    @classmethod
    def load(cls, path: Path | str) -> "StoreConfig":
        source = Path(path).expanduser()
        if not source.exists():
            return cls()
        data = tomllib.loads(source.read_text(encoding="utf-8"))
        section = data.get("store") if isinstance(data.get("store"), dict) else data
        root = section.get("root")
        maximum = section.get("max_adapters")
        if maximum is not None and (not isinstance(maximum, int) or maximum < 1):
            raise StoreError("store.max_adapters must be a positive integer")
        return cls(
            root=Path(str(root)).expanduser() if root else DEFAULT_STORE_ROOT,
            max_adapters=maximum,
        )


@dataclass(frozen=True, slots=True)
class AdapterRecord:
    digest: str
    path: Path
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def is_adapter(self) -> bool:
        """False for the adapter-free base, which is a first-class candidate."""
        return (self.path / "adapters.safetensors").exists()


class AdapterStore:
    """Immutable, content-addressed adapters plus append-only run records."""

    def __init__(self, config: StoreConfig | None = None):
        self.config = config or StoreConfig()
        self.root = Path(self.config.root).expanduser()

    # -- layout ---------------------------------------------------------
    @property
    def adapters_dir(self) -> Path:
        return self.root / "adapters"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def adapter_path(self, digest: str) -> Path:
        return self.adapters_dir / _digest_dirname(digest)

    def run_dir(self, run_id: str) -> Path:
        if not run_id or "/" in run_id or run_id in {".", ".."}:
            raise StoreError(f"unusable run_id: {run_id!r}")
        return self.runs_dir / run_id

    # -- adapters -------------------------------------------------------
    def put_adapter(
        self,
        source: Path | str,
        *,
        run_id: str | None = None,
        step: int | None = None,
        parent_digest: str | None = None,
        algorithm: str | None = None,
        created_at: str,
        base_model: str,
        extra: dict[str, Any] | None = None,
    ) -> AdapterRecord:
        """Copy an adapter directory in under its own digest.

        Re-putting identical bytes is a no-op that returns the existing record:
        content addressing makes deduplication free, and a warm start that
        re-registers its starting adapter must not fail.
        """

        origin = Path(source).expanduser()
        if not origin.is_dir():
            raise StoreError(f"adapter source is not a directory: {origin}")
        missing = [name for name in ADAPTER_REQUIRED_FILES if not (origin / name).exists()]
        if missing:
            raise StoreError(
                f"adapter source is missing {', '.join(missing)}; an adapter that "
                "failed to copy must not be storable as though it were complete"
            )

        digest = digest_of_tree(origin)
        destination = self.adapter_path(digest)
        if destination.exists():
            return self.get_adapter(digest)

        staging = destination.with_name(destination.name + ".incoming")
        if staging.exists():
            _rmtree(staging)
        staging.mkdir(parents=True)
        for item in sorted(origin.rglob("*")):
            if not item.is_file():
                continue
            target = staging / item.relative_to(origin)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.read_bytes())

        provenance = {
            "digest": digest,
            "run_id": run_id,
            "step": step,
            "parent_digest": parent_digest,
            "algorithm": algorithm,
            "base_model": base_model,
            "created_at": created_at,
            **(extra or {}),
        }
        (staging / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
        _freeze(destination)
        return AdapterRecord(digest=digest, path=destination, provenance=provenance)

    def get_adapter(self, digest: str) -> AdapterRecord:
        path = self.adapter_path(digest)
        if not path.is_dir():
            raise StoreError(f"unknown adapter: {digest}")
        provenance_path = path / "provenance.json"
        provenance = (
            json.loads(provenance_path.read_text(encoding="utf-8"))
            if provenance_path.exists()
            else {}
        )
        return AdapterRecord(digest=digest, path=path, provenance=provenance)

    def has_adapter(self, digest: str) -> bool:
        return self.adapter_path(digest).is_dir()

    def list_adapters(self) -> list[str]:
        if not self.adapters_dir.is_dir():
            return []
        return sorted(
            entry.name.replace("-", ":", 1)
            for entry in self.adapters_dir.iterdir()
            if entry.is_dir() and not entry.name.endswith(".incoming")
        )

    def lineage(self, digest: str) -> list[str]:
        """Adapter ancestry, newest first, stopping at the first unknown parent.

        A cycle would otherwise hang here; a repeated digest terminates the walk
        rather than raising, because a malformed provenance file is not a reason
        to make an existing adapter unreadable.
        """

        chain: list[str] = []
        seen: set[str] = set()
        current: str | None = digest
        while current and current not in seen and self.has_adapter(current):
            chain.append(current)
            seen.add(current)
            parent = self.get_adapter(current).provenance.get("parent_digest")
            current = str(parent) if parent else None
        return chain

    # -- run records ----------------------------------------------------
    def open_run(self, run_id: str, manifest: dict[str, Any]) -> "RunRecord":
        directory = self.run_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        record = RunRecord(store=self, run_id=run_id, path=directory)
        run_json = directory / "run.json"
        if not run_json.exists():
            payload = dict(manifest)
            payload.setdefault("run_id", run_id)
            payload.setdefault("schema_version", "synth-mlx-rl.run.v1")
            if "resumable" not in payload:
                # The upstream engine persists neither RNG state nor outstanding
                # accumulated gradients. A layout that looks resumable but is
                # not is worse than one that says so.
                payload["resumable"] = False
            run_json.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return record

    def get_run(self, run_id: str) -> "RunRecord":
        directory = self.run_dir(run_id)
        if not directory.is_dir():
            raise StoreError(f"unknown run: {run_id}")
        return RunRecord(store=self, run_id=run_id, path=directory)

    def list_runs(self) -> list[str]:
        if not self.runs_dir.is_dir():
            return []
        return sorted(entry.name for entry in self.runs_dir.iterdir() if entry.is_dir())

    def runs_for_adapter(self, digest: str) -> list[str]:
        """Every run that checkpointed this adapter. The reverse of the lookup
        an eval scorecard performs, and the reason `evals.jsonl` exists."""

        matches: list[str] = []
        for run_id in self.list_runs():
            record = self.get_run(run_id)
            if any(row.get("adapter_digest") == digest for row in record.checkpoints()):
                matches.append(run_id)
        return matches


@dataclass(frozen=True, slots=True)
class RunRecord:
    store: AdapterStore
    run_id: str
    path: Path

    def manifest(self) -> dict[str, Any]:
        return json.loads((self.path / "run.json").read_text(encoding="utf-8"))

    def _append(self, name: str, row: dict[str, Any]) -> None:
        with (self.path / name).open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(row) + "\n")

    def _read(self, name: str) -> list[dict[str, Any]]:
        target = self.path / name
        if not target.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in target.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def event(self, kind: str, payload: dict[str, Any] | None = None) -> None:
        self._append("events.jsonl", {"kind": kind, "payload": payload or {}})

    def events(self) -> list[dict[str, Any]]:
        return self._read("events.jsonl")

    def checkpoint(
        self, *, step: int, adapter_digest: str, metrics: dict[str, Any] | None = None
    ) -> None:
        if not self.store.has_adapter(adapter_digest):
            raise StoreError(
                f"cannot record checkpoint for unknown adapter {adapter_digest}; "
                "put the adapter before referencing it"
            )
        self._append(
            "checkpoints.jsonl",
            {"step": step, "adapter_digest": adapter_digest, "metrics": metrics or {}},
        )

    def checkpoints(self) -> list[dict[str, Any]]:
        return self._read("checkpoints.jsonl")

    def record_eval(
        self, *, eval_run_id: str, adapter_digest: str, scorecard_ref: str | None = None
    ) -> None:
        self._append(
            "evals.jsonl",
            {
                "eval_run_id": eval_run_id,
                "adapter_digest": adapter_digest,
                "scorecard_ref": scorecard_ref,
            },
        )

    def evals(self) -> list[dict[str, Any]]:
        return self._read("evals.jsonl")


def _freeze(root: Path) -> None:
    """Drop write bits so a stored adapter cannot be edited in place."""

    for item in sorted(root.rglob("*"), reverse=True):
        if item.is_file():
            item.chmod(item.stat().st_mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)


def _rmtree(root: Path) -> None:
    for item in sorted(root.rglob("*"), reverse=True):
        item.chmod(item.stat().st_mode | stat.S_IWUSR)
        item.unlink() if item.is_file() else item.rmdir()
    root.rmdir()


def default_store(config_path: Path | str | None = None) -> AdapterStore:
    path = Path(config_path).expanduser() if config_path else (DEFAULT_STORE_ROOT.expanduser() / "store.toml")
    return AdapterStore(StoreConfig.load(path))

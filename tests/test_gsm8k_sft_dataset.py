"""scripts/gsm8k_sft_dataset.py: pinned rows in, digest-named JSONL out.

The containers world module is stubbed (its own pin and parser are tested in
containers); what is under test here is the script's contract: the profile is
declared in code rather than read from the environment, an unpinned world is
refused, files are named by their content digest, and the manifest cites the
dataset pin and the seeds each file covers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "gsm8k_sft_dataset.py"
REVISION = "740312add88f781978c0658806c59bc2815b9866"


@pytest.fixture(autouse=True)
def _fresh_world_import(monkeypatch):
    """Each test gets its own stub: drop any cached synth_containers import."""
    for name in list(sys.modules):
        if name == "synth_containers" or name.startswith("synth_containers."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    yield
    for name in list(sys.modules):
        if name == "synth_containers" or name.startswith("synth_containers."):
            sys.modules.pop(name, None)


def _load_script():
    spec = importlib.util.spec_from_file_location("gsm8k_sft_dataset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stub_containers(tmp_path: Path, *, pinned: bool = True, rows: int = 40) -> tuple[Path, Path]:
    src = tmp_path / "containers-src"
    package = src / "synth_containers" / "platform"
    package.mkdir(parents=True)
    (src / "synth_containers" / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    calls = tmp_path / "calls.json"
    (package / "gsm8k_world.py").write_text(
        f'''
import json
from dataclasses import dataclass
CALLS = {str(calls)!r}
SOLVE_SYSTEM = "Solve it. End with '#### <answer>'."
TRAIN_SPLIT, HELDOUT_SPLIT = "train", "heldout"
HF_REVISION = {REVISION!r}

@dataclass(frozen=True)
class SplitPin:
    hf_split: str

SPLIT_PINS = {{"train": SplitPin("train"), "heldout": SplitPin("test")}}

@dataclass(frozen=True)
class Gsm8kRow:
    question: str
    answer_text: str
    @property
    def answer(self):
        return self.answer_text.rsplit("####", 1)[-1].strip()

def declare_profile(name, *, snapshot_dir=None):
    json.dump({{"declared": name}}, open(CALLS, "w"))

def dataset_manifest():
    return {{"dataset": "openai/gsm8k", "config": "main", "revision": HF_REVISION,
            "profile": "hf", "profile_source": "declared", "pinned": {pinned!r},
            "splits": {{"train": {{"hf_split": "train", "rows": 7473, "digest": "sha256:" + "a" * 64}},
                       "heldout": {{"hf_split": "test", "rows": 1319, "digest": "sha256:" + "b" * 64}}}},
            "shuffle_seed": 20260820}}

def user_prompt(question):
    return "Problem:\\n" + question

def load_row(split, seed):
    if seed >= {rows}:
        return None
    return Gsm8kRow(f"{{split}} question {{seed}}", f"work\\n#### {{seed * 3}}")
'''
    )
    return src, calls


def test_writes_digest_named_jsonl_in_the_containers_prompt_shape(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTH_GSM8K_SOURCE", raising=False)
    src, calls = _stub_containers(tmp_path)
    out = tmp_path / "out"
    script = _load_script()
    assert script.main(["--out", str(out), "--train", "5", "--eval", "3", "--containers-src", str(src)]) == 0

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["schema_version"] == "gsm8k.sft-dataset.v1"
    assert manifest["dataset"]["revision"] == REVISION
    assert manifest["dataset"]["splits"]["heldout"]["rows"] == 1319
    assert manifest["dataset"]["shuffle_seed"] == 20260820
    assert manifest["files"]["train"]["seeds"] == {"first": 0, "last": 4}
    assert manifest["files"]["eval"]["seeds"] == {"first": 0, "last": 2}
    assert manifest["files"]["eval"]["hf_split"] == "test"

    for label, count in (("train", 5), ("eval", 3)):
        entry = manifest["files"][label]
        path = out / entry["path"]
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        assert entry["sha256"] == f"sha256:{digest}"
        assert path.name == f"gsm8k-{label}-{count}-{digest[:12]}.jsonl"
        lines = [json.loads(line) for line in payload.decode().splitlines()]
        assert len(lines) == count
        roles = [m["role"] for m in lines[0]["messages"]]
        assert roles == ["system", "user", "assistant"]
        assert lines[0]["messages"][0]["content"].startswith("Solve it.")
        assert lines[0]["messages"][1]["content"].startswith("Problem:\n")
        assert lines[0]["messages"][2]["content"].endswith("#### 0")
    # Train and eval come from different splits: same seed, different question.
    train0 = json.loads((out / manifest["files"]["train"]["path"]).read_text().splitlines()[0])
    eval0 = json.loads((out / manifest["files"]["eval"]["path"]).read_text().splitlines()[0])
    assert "train question 0" in train0["messages"][1]["content"]
    assert "heldout question 0" in eval0["messages"][1]["content"]


def test_the_profile_is_declared_in_code_not_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("SYNTH_GSM8K_SOURCE", raising=False)
    src, calls = _stub_containers(tmp_path)
    _load_script().main(["--out", str(tmp_path / "o"), "--train", "2", "--eval", "1", "--containers-src", str(src)])
    assert json.loads(calls.read_text()) == {"declared": "hf"}
    assert "SYNTH_GSM8K_SOURCE" not in os.environ


def test_an_unpinned_world_is_refused(tmp_path):
    src, _ = _stub_containers(tmp_path, pinned=False)
    with pytest.raises(SystemExit, match="unpinned"):
        _load_script().main(["--out", str(tmp_path / "o"), "--train", "2", "--eval", "1", "--containers-src", str(src)])
    assert not (tmp_path / "o" / "manifest.json").exists()


def test_asking_for_more_rows_than_the_split_has_is_refused(tmp_path):
    src, _ = _stub_containers(tmp_path, rows=3)
    with pytest.raises(SystemExit, match="only 3 rows"):
        _load_script().main(["--out", str(tmp_path / "o"), "--train", "10", "--eval", "1", "--containers-src", str(src)])

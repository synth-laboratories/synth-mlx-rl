"""The adapter store: one digest, three surfaces.

The store's whole reason to exist is that the sha256 of an adapter directory is
the same identity the proxy pins and the Optimizers eval candidate carries. The
first test here pins the hash algorithm against a hand-computed vector, because
a silent divergence from `synth_optimizers.eval.models.digest_of_tree` would not
fail anything — it would just stop joining.
"""

from __future__ import annotations

import hashlib
import json
import stat

import pytest

from synth_mlx_rl.store import (
    AdapterStore,
    StoreConfig,
    StoreError,
    digest_of_tree,
)

CREATED = "2026-08-18T00:00:00Z"
BASE = "Qwen/Qwen3.5-0.8B"


def _adapter(root, *, weights: bytes = b"weights", rank: int = 8):
    root.mkdir(parents=True, exist_ok=True)
    (root / "adapter_config.json").write_text(json.dumps({"rank": rank}), encoding="utf-8")
    (root / "adapters.safetensors").write_bytes(weights)
    return root


def _store(tmp_path) -> AdapterStore:
    return AdapterStore(StoreConfig(root=tmp_path / "store"))


def test_digest_matches_the_optimizers_algorithm_exactly(tmp_path) -> None:
    """sorted relative path + NUL + sha256(bytes), hashed in order.

    Hand-computed rather than golden-file'd: a golden file would be regenerated
    by whoever broke it.
    """
    root = tmp_path / "a"
    root.mkdir()
    (root / "b.txt").write_bytes(b"second")
    (root / "a.txt").write_bytes(b"first")

    expected = hashlib.sha256()
    for name, payload in (("a.txt", b"first"), ("b.txt", b"second")):
        expected.update(name.encode("utf-8"))
        expected.update(b"\0")
        expected.update(hashlib.sha256(payload).digest())

    assert digest_of_tree(root) == "sha256:" + expected.hexdigest()


def test_digest_is_path_sensitive_not_just_content(tmp_path) -> None:
    one = _adapter(tmp_path / "one")
    two = _adapter(tmp_path / "two")
    assert digest_of_tree(one) == digest_of_tree(two)
    (two / "extra.json").write_text("{}", encoding="utf-8")
    assert digest_of_tree(one) != digest_of_tree(two)


def test_put_is_content_addressed_and_deduplicates(tmp_path) -> None:
    store = _store(tmp_path)
    first = store.put_adapter(
        _adapter(tmp_path / "src1"), run_id="run-a", step=10, created_at=CREATED, base_model=BASE
    )
    # A warm start re-registering identical bytes must not fail, and must not
    # store a second copy.
    second = store.put_adapter(
        _adapter(tmp_path / "src2"), run_id="run-b", step=99, created_at=CREATED, base_model=BASE
    )
    assert first.digest == second.digest
    assert store.list_adapters() == [first.digest]
    # Provenance belongs to the bytes, so the first writer's record stands.
    assert second.provenance["run_id"] == "run-a"


def test_a_stored_adapter_is_read_only(tmp_path) -> None:
    store = _store(tmp_path)
    record = store.put_adapter(
        _adapter(tmp_path / "src"), run_id="r", step=1, created_at=CREATED, base_model=BASE
    )
    mode = (record.path / "adapters.safetensors").stat().st_mode
    assert not mode & stat.S_IWUSR


def test_an_incomplete_adapter_is_refused(tmp_path) -> None:
    """An adapter whose weights failed to copy must not be storable as though
    it were complete — that is how a checkpoint gets scored as its own baseline."""
    store = _store(tmp_path)
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(StoreError, match="adapters.safetensors"):
        store.put_adapter(partial, run_id="r", step=1, created_at=CREATED, base_model=BASE)


def test_lineage_walks_parents_and_survives_a_cycle(tmp_path) -> None:
    store = _store(tmp_path)
    base = store.put_adapter(
        _adapter(tmp_path / "b", weights=b"base"), created_at=CREATED, base_model=BASE
    )
    sft = store.put_adapter(
        _adapter(tmp_path / "s", weights=b"sft"),
        run_id="sft-run",
        step=20,
        parent_digest=base.digest,
        algorithm="sft",
        created_at=CREATED,
        base_model=BASE,
    )
    rl = store.put_adapter(
        _adapter(tmp_path / "r", weights=b"rl"),
        run_id="rl-run",
        step=5,
        parent_digest=sft.digest,
        algorithm="cispo_minimax",
        created_at=CREATED,
        base_model=BASE,
    )
    # base -> SFT -> RL is the question actually asked when a number looks wrong;
    # recording only run_id would answer one hop.
    assert store.lineage(rl.digest) == [rl.digest, sft.digest, base.digest]
    assert store.lineage(base.digest) == [base.digest]


def test_a_run_references_adapters_and_refuses_unknown_ones(tmp_path) -> None:
    store = _store(tmp_path)
    record = store.put_adapter(
        _adapter(tmp_path / "src"), run_id="run-1", step=10, created_at=CREATED, base_model=BASE
    )
    run = store.open_run("run-1", {"base_model": BASE, "algorithm": "sft", "seed": 0})
    run.checkpoint(step=10, adapter_digest=record.digest, metrics={"loss": 1.5})
    assert run.checkpoints()[0]["adapter_digest"] == record.digest

    with pytest.raises(StoreError, match="unknown adapter"):
        run.checkpoint(step=20, adapter_digest="sha256:" + "0" * 64)


def test_run_manifest_states_resumability_rather_than_implying_it(tmp_path) -> None:
    """The engine persists neither RNG state nor outstanding accumulated
    gradients. A layout that looks resumable but is not is worse than one that
    says so."""
    store = _store(tmp_path)
    run = store.open_run("run-2", {"base_model": BASE, "algorithm": "grpo"})
    assert run.manifest()["resumable"] is False


def test_the_eval_loop_closes_in_both_directions(tmp_path) -> None:
    store = _store(tmp_path)
    record = store.put_adapter(
        _adapter(tmp_path / "src"), run_id="run-3", step=40, created_at=CREATED, base_model=BASE
    )
    run = store.open_run("run-3", {"base_model": BASE, "algorithm": "cispo_minimax"})
    run.checkpoint(step=40, adapter_digest=record.digest)
    run.record_eval(eval_run_id="eval-9", adapter_digest=record.digest, scorecard_ref="s.json")

    # scorecard -> run
    assert store.runs_for_adapter(record.digest) == ["run-3"]
    # run -> every eval that scored it
    assert [row["eval_run_id"] for row in run.evals()] == ["eval-9"]
    assert run.evals()[0]["adapter_digest"] == record.digest


def test_events_are_append_only(tmp_path) -> None:
    store = _store(tmp_path)
    run = store.open_run("run-4", {"base_model": BASE, "algorithm": "sft"})
    run.event("started", {"seed": 0})
    run.event("finished", {"steps": 3})
    assert [row["kind"] for row in run.events()] == ["started", "finished"]


def test_reopening_a_run_does_not_rewrite_its_manifest(tmp_path) -> None:
    store = _store(tmp_path)
    store.open_run("run-5", {"base_model": BASE, "algorithm": "sft", "seed": 7})
    again = store.open_run("run-5", {"base_model": "something-else", "algorithm": "grpo"})
    assert again.manifest()["base_model"] == BASE
    assert again.manifest()["seed"] == 7


def test_store_config_comes_from_toml_not_the_environment(tmp_path, monkeypatch) -> None:
    """A path that lives in the environment cannot be read back off a finished
    run, which is the entire reason a run record exists."""
    config = tmp_path / "store.toml"
    config.write_text('[store]\nroot = "%s"\nmax_adapters = 4\n' % (tmp_path / "custom"), encoding="utf-8")
    monkeypatch.setenv("SYNTH_MLX_RL_STORE_ROOT", str(tmp_path / "ignored"))
    loaded = StoreConfig.load(config)
    assert loaded.root == tmp_path / "custom"
    assert loaded.max_adapters == 4

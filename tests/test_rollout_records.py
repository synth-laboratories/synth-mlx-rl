"""Server-side rollout records: the training authority."""

from __future__ import annotations

import pytest

from synth_mlx_rl.rollouts import (
    RolloutRecordNotFoundError,
    RolloutStore,
    new_proxy_request_id,
)
from synth_mlx_rl.schemas import SampleRequest, SamplingParamsModel


def _record(**overrides):
    from synth_mlx_rl.rollouts import RolloutRecord

    payload = dict(
        proxy_request_id=new_proxy_request_id(),
        policy_snapshot_id="snap_1",
        training_version=3,
        api_family="chat_completions",
        model="fake/Qwen3.5-0.8B",
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        rollout_logprobs=[-0.1, -0.2],
        finish_reason="stop",
        sampling_params=SamplingParamsModel(),
        tokenizer_digest="tok",
        template_digest="tpl",
        render_digest="rnd",
        enable_thinking=False,
        created_at=0.0,
        duration_ms=1.0,
    )
    payload.update(overrides)
    return RolloutRecord(**payload)


def test_misaligned_logprobs_are_refused() -> None:
    store = RolloutStore(capacity=4)
    with pytest.raises(ValueError) as excinfo:
        store.put(_record(rollout_logprobs=[-0.1]))
    assert "one-to-one" in str(excinfo.value)


def test_missing_record_is_an_error_not_an_empty_record() -> None:
    store = RolloutStore(capacity=4)
    with pytest.raises(RolloutRecordNotFoundError) as excinfo:
        store.get("prid_nope")
    assert "substituting zeros" in str(excinfo.value)


def test_eviction_reports_missing_rather_than_lying() -> None:
    store = RolloutStore(capacity=2)
    first = store.put(_record())
    store.put(_record())
    store.put(_record())
    found, missing = store.get_many([first.proxy_request_id, "prid_nope"])
    assert found == []
    assert missing == [first.proxy_request_id, "prid_nope"]


def test_token_ids_concatenate_prompt_and_completion() -> None:
    record = _record()
    assert record.token_ids == [1, 2, 3, 4, 5]


def test_sampling_writes_a_complete_record(engine) -> None:
    response = engine.sample(
        SampleRequest(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            temperature=0.0,
            top_p=0.9,
            min_p=0.05,
            top_k=7,
            seed=11,
        )
    )
    sample = response.samples[0]
    record = engine.rollouts.get(sample.proxy_request_id)

    assert record.policy_snapshot_id == sample.policy_snapshot_id
    assert record.completion_token_ids == sample.completion_token_ids
    assert record.rollout_logprobs == sample.rollout_logprobs
    assert record.alignment_ok()
    # The sampling knobs are recorded verbatim, so a rerun is reproducible and
    # the truncation actually applied is visible next to the raw log-probs.
    assert record.sampling_params.top_p == 0.9
    assert record.sampling_params.min_p == 0.05
    assert record.sampling_params.top_k == 7
    assert record.sampling_params.seed == 11
    assert record.tokenizer_digest and record.template_digest and record.render_digest
    assert record.duration_ms >= 0.0


def test_each_sample_in_a_group_gets_its_own_record(engine) -> None:
    response = engine.sample(SampleRequest(prompt="hello", num_samples=4))
    ids = [sample.proxy_request_id for sample in response.samples]
    assert len(set(ids)) == 4
    found, missing = engine.rollouts.get_many(ids)
    assert missing == []
    assert len(found) == 4
    assert len({record.policy_snapshot_id for record in found}) == 1

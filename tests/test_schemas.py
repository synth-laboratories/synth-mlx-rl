"""Schema-level refusals.

Ported from the MIT-licensed prototype's schema tests and extended for the
objective vocabulary.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from synth_mlx_rl.schemas import Datum, ForwardBackwardRequest, SampleRequest


def test_datum_requires_aligned_arrays() -> None:
    with pytest.raises(ValidationError):
        Datum(input_ids=[1, 2], target_ids=[2], weights=[1.0])


def test_datum_rejects_negative_or_non_finite_weights() -> None:
    with pytest.raises(ValidationError):
        Datum(input_ids=[1], target_ids=[2], weights=[-1.0])
    with pytest.raises(ValidationError):
        Datum(input_ids=[1], target_ids=[2], weights=[float("nan")])


def test_datum_requires_at_least_one_selected_token() -> None:
    with pytest.raises(ValidationError):
        Datum(input_ids=[1, 2], target_ids=[2, 3], weights=[0.0, 0.0])


def test_policy_objectives_require_behavior_logprobs_and_advantages() -> None:
    datum = Datum(input_ids=[1], target_ids=[2], weights=[1.0])
    for loss_fn in ("importance_sampling", "grpo", "cispo_minimax", "cispo_two_sided"):
        with pytest.raises(ValidationError):
            ForwardBackwardRequest(data=[datum], loss_fn=loss_fn)


def test_ppo_is_refused_with_a_reason(client=None) -> None:
    datum = Datum(input_ids=[1], target_ids=[2], weights=[1.0])
    with pytest.raises(ValidationError) as excinfo:
        ForwardBackwardRequest(data=[datum], loss_fn="ppo")
    assert "no value head" in str(excinfo.value)


def test_unknown_objective_lists_the_supported_ones() -> None:
    datum = Datum(input_ids=[1], target_ids=[2], weights=[1.0])
    with pytest.raises(ValidationError) as excinfo:
        ForwardBackwardRequest(data=[datum], loss_fn="dpo")
    assert "cispo_minimax" in str(excinfo.value)


def test_cispo_minimax_refusal_reaches_the_request_layer() -> None:
    datum = Datum(
        input_ids=[1],
        target_ids=[2],
        weights=[1.0],
        behavior_logprobs=[-1.0],
        advantages=1.0,
    )
    with pytest.raises(ValidationError) as excinfo:
        ForwardBackwardRequest(data=[datum], loss_fn="cispo_minimax", eps_low=0.2)
    assert "cispo_two_sided" in str(excinfo.value)
    # ...and the two-sided name accepts exactly what the single-sided one refused.
    ForwardBackwardRequest(data=[datum], loss_fn="cispo_two_sided", eps_low=0.2)


def test_sample_requires_exactly_one_prompt_source() -> None:
    with pytest.raises(ValidationError):
        SampleRequest(prompt="x", prompt_token_ids=[1])
    with pytest.raises(ValidationError):
        SampleRequest()


def test_sample_rejects_empty_or_negative_token_prompt() -> None:
    with pytest.raises(ValidationError):
        SampleRequest(prompt_token_ids=[])
    with pytest.raises(ValidationError):
        SampleRequest(prompt_token_ids=[-1])


def test_tool_message_requires_a_call_id() -> None:
    from synth_mlx_rl.schemas import ChatMessage

    with pytest.raises(ValidationError):
        ChatMessage(role="tool", content="result")

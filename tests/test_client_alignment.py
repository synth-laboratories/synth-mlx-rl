"""The prompt/completion alignment helper.

Ported from the MIT-licensed prototype. This is the easiest thing in the whole
system to get wrong: the first completion token is predicted from the *last
prompt* position, so the weighted span starts one before the completion.
"""

from __future__ import annotations

import pytest

from synth_mlx_rl.client import TrainingClient
from synth_mlx_rl.schemas import Datum, ForwardBackwardResponse, Sample


def _sample() -> Sample:
    return Sample(
        text="ok",
        prompt_token_ids=[10, 11, 12],
        completion_token_ids=[20, 21, 0],
        rollout_logprobs=[-1.0, -2.0, -3.0],
        finish_reason="stop",
        policy_snapshot_id="snap_abc",
        training_version=7,
        proxy_request_id="prid_abc",
    )


def test_rollout_alignment_predicts_every_completion_token() -> None:
    datum = TrainingClient.datum_from_sample(_sample(), advantage=2.5)
    assert datum.input_ids == [10, 11, 12, 20, 21]
    assert datum.target_ids == [11, 12, 20, 21, 0]
    assert datum.weights == [0.0, 0.0, 1.0, 1.0, 1.0]
    assert datum.behavior_logprobs == [0.0, 0.0, -1.0, -2.0, -3.0]
    assert datum.advantages == 2.5
    assert datum.metadata["policy_snapshot_id"] == "snap_abc"
    assert datum.metadata["proxy_request_id"] == "prid_abc"


def test_rollout_logprobs_standing_in_for_behavior_is_recorded() -> None:
    """The two populations are different. A datum that conflated them says so."""

    fallback = TrainingClient.datum_from_sample(_sample(), advantage=1.0)
    assert fallback.metadata["behavior_logprob_source"] == "rollout"

    explicit = TrainingClient.datum_from_sample(
        _sample(), advantage=1.0, behavior_logprobs=[-1.1, -2.1, -3.1]
    )
    assert explicit.metadata["behavior_logprob_source"] == "behavior"
    assert explicit.behavior_logprobs == [0.0, 0.0, -1.1, -2.1, -3.1]


def test_token_level_advantages_land_on_completion_positions() -> None:
    datum = TrainingClient.datum_from_sample(
        _sample(), advantage=[1.0, 2.0, 3.0]
    )
    assert datum.advantages == [0.0, 0.0, 1.0, 2.0, 3.0]


def test_misaligned_behavior_logprobs_are_refused() -> None:
    with pytest.raises(ValueError):
        TrainingClient.datum_from_sample(
            _sample(), advantage=1.0, behavior_logprobs=[-1.0]
        )


class _RecordingTransport:
    def __init__(self) -> None:
        self.submissions: list[tuple[str, dict, type]] = []
        self.posts: list[tuple[str, dict, type]] = []

    def submit_post(self, path: str, payload: dict, response_type: type):
        self.submissions.append((path, payload, response_type))
        return object()

    def post(self, path: str, payload: dict, response_type: type):
        from synth_mlx_rl.schemas import RenderChatResponse

        self.posts.append((path, payload, response_type))
        return RenderChatResponse(
            token_ids=[1, 2, 3],
            tokenizer_digest="tok",
            template_digest="tpl",
            render_digest="rnd",
            enable_thinking=False,
        )


def test_forward_backward_uses_one_response_type_argument() -> None:
    transport = _RecordingTransport()
    trainer = TrainingClient(transport)  # type: ignore[arg-type]
    trainer.forward_backward(
        [Datum(input_ids=[1], target_ids=[2], weights=[1.0])], "cross_entropy"
    )
    path, _, response_type = transport.submissions[-1]
    assert path == "/v1/forward_backward"
    assert response_type is ForwardBackwardResponse


def test_thinking_override_is_forwarded_by_the_client() -> None:
    from synth_mlx_rl.client import RemoteTokenizer, SamplingClient, SamplingParams

    transport = _RecordingTransport()
    RemoteTokenizer(transport).apply_chat_template(  # type: ignore[arg-type]
        [{"role": "user", "content": "hello"}], enable_thinking=True
    )
    assert transport.posts[-1][1]["enable_thinking"] is True

    sampler = SamplingClient(transport, "snap_pinned")  # type: ignore[arg-type]
    sampler.sample(
        [{"role": "user", "content": "hello"}], SamplingParams(enable_thinking=False)
    )
    payload = transport.submissions[-1][1]
    assert payload["enable_thinking"] is False
    # The pin travels on every request, not resolved per call.
    assert payload["policy_snapshot_id"] == "snap_pinned"

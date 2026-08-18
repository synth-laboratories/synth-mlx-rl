"""The D9 invariant: two API families, one renderer, one rollout record.

If a responses-family rollout renders a different transcript than a chat-family
one, every mixed-family comparison is silently invalid. These tests assert the
two surfaces produce byte-identical prompt tokens and identical digests for the
same conversation -- and that the record says which surface it came from, so a
divergence would be visible rather than inferred.
"""

from __future__ import annotations

from synth_mlx_rl.api.normalize import (
    normalize_responses_input,
    normalize_responses_tools,
)
from synth_mlx_rl.schemas import ChatMessage, ToolDefinition, ToolFunction

CHAT_BODY = {
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "weather in nyc?"},
    ],
    "temperature": 0.0,
    "max_tokens": 24,
}

RESPONSES_BODY = {
    "instructions": "be terse",
    "input": [{"role": "user", "content": [{"type": "input_text", "text": "weather in nyc?"}]}],
    "temperature": 0.0,
    "max_output_tokens": 24,
}


def test_both_families_normalize_to_the_same_messages() -> None:
    from_chat = [ChatMessage.model_validate(m) for m in CHAT_BODY["messages"]]
    from_responses = normalize_responses_input(
        RESPONSES_BODY["input"], instructions=RESPONSES_BODY["instructions"]
    )
    assert from_chat == from_responses


def test_flat_and_nested_tool_shapes_normalize_to_one_shape() -> None:
    nested = [
        ToolDefinition(
            function=ToolFunction(
                name="get_weather", parameters={"type": "object", "properties": {}}
            )
        )
    ]
    flat = normalize_responses_tools(
        [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
    )
    assert flat == nested


def test_identical_conversations_render_identically_across_families(client) -> None:
    chat = client.post("/v1/chat/completions", json=CHAT_BODY)
    responses = client.post("/v1/responses", json=RESPONSES_BODY)
    assert chat.status_code == 200, chat.text
    assert responses.status_code == 200, responses.text

    chat_synth = chat.json()["synth"]
    responses_synth = responses.json()["synth"]

    assert chat_synth["render_digest"] == responses_synth["render_digest"]
    assert chat_synth["tokenizer_digest"] == responses_synth["tokenizer_digest"]
    assert chat_synth["template_digest"] == responses_synth["template_digest"]
    # The families are still distinguishable on the record; the rendering is
    # what must match, not the provenance.
    assert chat_synth["api_family"] == "chat_completions"
    assert responses_synth["api_family"] == "responses"


def test_prompt_tokens_are_identical_across_families(client) -> None:
    chat = client.post("/v1/chat/completions", json=CHAT_BODY).json()
    responses = client.post("/v1/responses", json=RESPONSES_BODY).json()

    chat_record = client.get(
        f"/v1/synth/rollouts/{chat['synth']['proxy_request_ids'][0]}"
    ).json()["record"]
    responses_record = client.get(
        f"/v1/synth/rollouts/{responses['synth']['proxy_request_ids'][0]}"
    ).json()["record"]

    assert chat_record["prompt_token_ids"] == responses_record["prompt_token_ids"]
    assert chat_record["enable_thinking"] == responses_record["enable_thinking"]


def test_a_different_thinking_mode_changes_the_render_digest(client) -> None:
    """The digest has to be sensitive to the flag, or it detects nothing."""

    off = client.post("/v1/chat/completions", json={**CHAT_BODY, "enable_thinking": False})
    on = client.post("/v1/chat/completions", json={**CHAT_BODY, "enable_thinking": True})
    assert off.json()["synth"]["render_digest"] != on.json()["synth"]["render_digest"]


def test_tools_change_the_render_digest_on_both_families(client) -> None:
    tool_chat = client.post(
        "/v1/chat/completions",
        json={
            **CHAT_BODY,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "parameters": {}},
                }
            ],
        },
    )
    tool_responses = client.post(
        "/v1/responses",
        json={
            **RESPONSES_BODY,
            "tools": [{"type": "function", "name": "get_weather", "parameters": {}}],
        },
    )
    plain = client.post("/v1/chat/completions", json=CHAT_BODY)

    assert (
        tool_chat.json()["synth"]["render_digest"]
        == tool_responses.json()["synth"]["render_digest"]
    )
    assert (
        tool_chat.json()["synth"]["render_digest"]
        != plain.json()["synth"]["render_digest"]
    )


def test_a_tool_round_trip_renders_the_same_on_both_families(client) -> None:
    """Full history, including a function call and its output, on both shapes."""

    history_chat = {
        "messages": [
            {"role": "user", "content": "weather in nyc?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city":"nyc"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_abc", "content": "sunny, 75F"},
        ],
        "temperature": 0.0,
    }
    history_responses = {
        "input": [
            {"role": "user", "content": "weather in nyc?"},
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city":"nyc"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "sunny, 75F",
            },
        ],
        "temperature": 0.0,
    }
    chat = client.post("/v1/chat/completions", json=history_chat)
    responses = client.post("/v1/responses", json=history_responses)
    assert chat.status_code == 200, chat.text
    assert responses.status_code == 200, responses.text
    assert (
        chat.json()["synth"]["render_digest"]
        == responses.json()["synth"]["render_digest"]
    )


def test_native_sample_route_shares_the_same_renderer(client) -> None:
    """A third surface must not be a third renderer either."""

    chat = client.post("/v1/chat/completions", json=CHAT_BODY).json()
    native = client.post(
        "/v1/sample",
        json={
            "messages": CHAT_BODY["messages"],
            "max_tokens": 24,
            "temperature": 0.0,
        },
    ).json()
    native_record = client.get(
        f"/v1/synth/rollouts/{native['samples'][0]['proxy_request_id']}"
    ).json()["record"]
    chat_record = client.get(
        f"/v1/synth/rollouts/{chat['synth']['proxy_request_ids'][0]}"
    ).json()["record"]
    assert native_record["render_digest"] == chat_record["render_digest"]
    assert native_record["api_family"] == "native"

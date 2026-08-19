"""Wire models for the two OpenAI-compatible families.

Decision D9: ``/v1/chat/completions`` and ``/v1/responses`` are peers. Neither
wraps the other, and both normalize into the same canonical messages before
anything is rendered.

These models are strict. A field this service does not implement is a 422 with a
reason, not a silently ignored key -- an ignored ``logprobs: true`` would return
a response that looks complete and is missing the thing the caller asked for.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schemas import ChatMessage, StrictModel, ToolDefinition


class OpenAIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class StreamOptions(OpenAIModel):
    include_usage: bool = False


class _SamplingFields(OpenAIModel):
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    #: Not in the OpenAI schema; accepted because the local sampler implements
    #: them and a rollout record must report the truncation actually applied.
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0)
    enable_thinking: bool | None = None
    #: The snapshot this request is pinned to, resolved once at request start.
    policy_snapshot_id: str | None = None


class ChatCompletionRequest(_SamplingFields):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "none"] | None = None
    max_tokens: int | None = Field(default=None, ge=1, le=32768)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=32768)
    n: int = Field(default=1, ge=1, le=16)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    user: str | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None

    @model_validator(mode="after")
    def validate_request(self) -> "ChatCompletionRequest":
        if self.logprobs or self.top_logprobs:
            raise ValueError(
                "logprobs are not returned on the wire. The sampler's raw "
                "per-token log-probabilities live on the server-side rollout "
                "record; fetch them with GET /v1/synth/rollouts/"
                "{proxy_request_id} using the id returned with this response."
            )
        return self

    @property
    def resolved_max_tokens(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 128


class ResponsesRequest(_SamplingFields):
    model: str | None = None
    #: Full history, every turn. The Responses family here is stateless by
    #: construction: there is no server-side conversation to continue from.
    input: str | list[dict[str, Any]] = Field(default_factory=list)
    instructions: str | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["auto", "none"] | None = None
    max_output_tokens: int | None = Field(default=None, ge=1, le=32768)
    stream: bool = False
    previous_response_id: str | None = None
    metadata: dict[str, Any] | None = None
    store: bool | None = None
    include: list[str] | None = None

    @property
    def resolved_max_tokens(self) -> int:
        return self.max_output_tokens or 128


class Usage(StrictModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SynthRolloutRef(StrictModel):
    """The join key, plus everything needed to detect a rendering divergence.

    Returned on both families, identically. A caller that sees different
    digests for the same conversation on the two surfaces has found a real bug,
    not a formatting difference.
    """

    proxy_request_ids: list[str]
    policy_snapshot_id: str
    training_version: int
    api_family: Literal["chat_completions", "responses", "native"]
    tokenizer_digest: str
    template_digest: str
    render_digest: str


class ChatCompletionChoice(StrictModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length"]


class ChatCompletionResponse(StrictModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    synth: SynthRolloutRef


class ResponsesOutputText(StrictModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: list[Any] = Field(default_factory=list)


class ResponsesOutputMessage(StrictModel):
    type: Literal["message"] = "message"
    id: str
    role: Literal["assistant"] = "assistant"
    status: Literal["completed", "incomplete"] = "completed"
    content: list[ResponsesOutputText]


class ResponsesUsage(StrictModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class ResponsesResponse(StrictModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    model: str
    status: Literal["completed", "incomplete"] = "completed"
    output: list[ResponsesOutputMessage]
    usage: ResponsesUsage
    synth: SynthRolloutRef


def new_chat_completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex}"


def new_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def new_message_id() -> str:
    return f"msg_{uuid.uuid4().hex}"


def unix_now() -> int:
    return int(time.time())

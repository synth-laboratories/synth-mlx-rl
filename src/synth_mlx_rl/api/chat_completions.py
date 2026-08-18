"""``POST /v1/chat/completions`` -- the Chat Completions family.

A peer of ``/v1/responses``, not a wrapper around it and not wrapped by it. It
differs from the responses route in exactly two places: how a request becomes
canonical messages, and how a completion becomes a payload. Rendering,
tokenization, snapshot pinning, and the rollout record are shared code.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..schemas import ChatMessage
from .common import (
    check_model,
    protocol_http_error,
    get_engine,
    idempotency_key,
    resolve_policy_pin,
    response_headers,
    sample_once,
    synth_ref,
    token_text_deltas,
)
from .normalize import ProtocolError, normalize_chat_tools
from .openai_schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    new_chat_completion_id,
    unix_now,
)

router = APIRouter()


@router.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest, request: Request) -> Any:
    engine = get_engine(request)
    model = check_model(engine, body.model)
    pin = resolve_policy_pin(request, body.policy_snapshot_id)

    try:
        tools = normalize_chat_tools(body.tools)
    except ProtocolError as exc:
        raise protocol_http_error(exc) from exc

    messages: list[ChatMessage] = list(body.messages)
    samples, records = sample_once(
        engine,
        messages=messages,
        tools=tools,
        api_family="chat_completions",
        max_tokens=body.resolved_max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        min_p=body.min_p,
        top_k=body.top_k,
        num_samples=body.n,
        seed=body.seed,
        stop=body.stop,
        enable_thinking=body.enable_thinking,
        policy_snapshot_id=pin,
        idempotency_key=idempotency_key(request),
    )

    completion_id = new_chat_completion_id()
    created = unix_now()
    headers = response_headers(records)

    if body.stream:
        return StreamingResponse(
            _stream(engine, completion_id, created, model, samples, records, body),
            media_type="text/event-stream",
            headers=headers,
        )

    prompt_tokens = len(records[0].prompt_token_ids)
    completion_tokens = sum(len(record.completion_token_ids) for record in records)
    payload = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model,
        choices=[
            ChatCompletionChoice(
                index=index,
                message=ChatMessage(role="assistant", content=sample.text),
                finish_reason=sample.finish_reason,
            )
            for index, sample in enumerate(samples)
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        synth=synth_ref(records),
    )
    return JSONResponse(content=payload.model_dump(), headers=headers)


def _chunk(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _stream(
    engine: Any,
    completion_id: str,
    created: int,
    model: str,
    samples: Any,
    records: Any,
    body: ChatCompletionRequest,
) -> Iterator[str]:
    """Frame an already-sampled completion as Chat Completions SSE.

    The deltas are advisory. The training record was written before the first
    byte left this function, and it -- not the stream -- is the authority on
    what was generated (finalized plan, section 5.4).
    """

    base = {"id": completion_id, "object": "chat.completion.chunk",
            "created": created, "model": model}

    for index, sample in enumerate(samples):
        yield _chunk(
            {
                **base,
                "choices": [
                    {"index": index, "delta": {"role": "assistant"},
                     "finish_reason": None}
                ],
            }
        )
        for delta in token_text_deltas(engine, sample.completion_token_ids):
            if not delta:
                continue
            yield _chunk(
                {
                    **base,
                    "choices": [
                        {"index": index, "delta": {"content": delta},
                         "finish_reason": None}
                    ],
                }
            )
        yield _chunk(
            {
                **base,
                "choices": [
                    {"index": index, "delta": {},
                     "finish_reason": sample.finish_reason}
                ],
            }
        )

    if body.stream_options is not None and body.stream_options.include_usage:
        prompt_tokens = len(records[0].prompt_token_ids)
        completion_tokens = sum(
            len(record.completion_token_ids) for record in records
        )
        yield _chunk(
            {
                **base,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
                "synth": synth_ref(records).model_dump(),
            }
        )
    yield "data: [DONE]\n\n"

"""``POST /v1/responses`` -- the Responses family.

Net-new: the seed prototype had only a chat route. The protocol reference is the
org's existing native implementation, `synth-responses-gateway`, and the
properties carried over from it are the ones that make a stateless service
honest:

* the full history arrives on every turn, in ``input``;
* tools are forwarded, in the Responses flat shape;
* ``previous_response_id`` alone is refused before any sampling happens;
* SSE carries Responses events, not translated chat chunks.

What is deliberately different: this service is an *origin*, not a proxy, so it
emits the event stream rather than passing one through, and it honours
``X-Policy-Pin`` and ``Idempotency-Key`` itself.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .common import (
    check_model,
    get_engine,
    http_error,
    idempotency_key,
    protocol_http_error,
    resolve_policy_pin,
    response_headers,
    sample_once,
    synth_ref,
    token_text_deltas,
)
from .idempotency import IdempotencyConflict, IdempotentEntry, request_digest
from .normalize import (
    ProtocolError,
    normalize_responses_input,
    normalize_responses_tools,
    validate_previous_response_id,
)
from .openai_schemas import (
    ResponsesOutputMessage,
    ResponsesOutputText,
    ResponsesRequest,
    ResponsesResponse,
    ResponsesUsage,
    new_message_id,
    new_response_id,
    unix_now,
)

router = APIRouter()


@router.post("/v1/responses")
def responses(body: ResponsesRequest, request: Request) -> Any:
    engine = get_engine(request)
    model = check_model(engine, body.model)
    pin = resolve_policy_pin(request, body.policy_snapshot_id)
    key = idempotency_key(request)
    digest = request_digest(body.model_dump())

    cache = request.app.state.idempotency
    if key is not None:
        try:
            cached = cache.lookup(key, digest)
        except IdempotencyConflict as exc:
            raise http_error("idempotency_key_conflict", str(exc), 409) from exc
        if cached is not None:
            # Replay. No second sample, no second rollout record, same
            # proxy_request_id as the first attempt.
            return JSONResponse(
                status_code=cached.status_code,
                content=cached.body,
                headers={**cached.headers, "Idempotency-Replayed": "true"},
            )

    try:
        # Refused before anything is rendered or sampled, exactly like the
        # native gateway refuses it before any upstream call.
        validate_previous_response_id(body.previous_response_id, body.input)
        messages = normalize_responses_input(
            body.input, instructions=body.instructions
        )
        tools = normalize_responses_tools(body.tools)
    except ProtocolError as exc:
        raise protocol_http_error(exc) from exc

    samples, records = sample_once(
        engine,
        messages=messages,
        tools=tools,
        api_family="responses",
        max_tokens=body.resolved_max_tokens,
        temperature=body.temperature,
        top_p=body.top_p,
        min_p=body.min_p,
        top_k=body.top_k,
        num_samples=1,
        seed=body.seed,
        stop=None,
        enable_thinking=body.enable_thinking,
        policy_snapshot_id=pin,
        idempotency_key=key,
    )

    response_id = new_response_id()
    created_at = unix_now()
    headers = response_headers(records)
    sample = samples[0]
    record = records[0]

    if body.stream:
        # A streamed response is not cached for replay: the cache stores a
        # body, and half a stream is not a body. A caller that wants replay
        # safety on a retried paid call should not be streaming it.
        return StreamingResponse(
            _stream(engine, response_id, created_at, model, sample, records),
            media_type="text/event-stream",
            headers=headers,
        )

    payload = ResponsesResponse(
        id=response_id,
        created_at=created_at,
        model=model,
        status="completed" if sample.finish_reason == "stop" else "incomplete",
        output=[
            ResponsesOutputMessage(
                id=new_message_id(),
                status="completed" if sample.finish_reason == "stop" else "incomplete",
                content=[ResponsesOutputText(text=sample.text)],
            )
        ],
        usage=ResponsesUsage(
            input_tokens=len(record.prompt_token_ids),
            output_tokens=len(record.completion_token_ids),
            total_tokens=len(record.token_ids),
        ),
        synth=synth_ref(records),
    )
    body_json = payload.model_dump()
    if key is not None:
        cache.store(
            key,
            IdempotentEntry(
                request_digest=digest,
                status_code=200,
                body=body_json,
                headers=headers,
            ),
        )
    return JSONResponse(content=body_json, headers=headers)


def _event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _stream(
    engine: Any,
    response_id: str,
    created_at: int,
    model: str,
    sample: Any,
    records: Any,
) -> Iterator[str]:
    """Emit the Responses event vocabulary.

    Named events with a JSON ``type`` on each payload, matching what the native
    gateway forwards: ``response.created``, ``response.output_text.delta``,
    ``response.completed``.
    """

    message_id = new_message_id()
    record = records[0]
    skeleton = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "model": model,
        "status": "in_progress",
    }
    yield _event("response.created", {"type": "response.created", "response": skeleton})
    yield _event(
        "response.output_item.added",
        {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "message",
                "id": message_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
    )
    for delta in token_text_deltas(engine, sample.completion_token_ids):
        if not delta:
            continue
        yield _event(
            "response.output_text.delta",
            {
                "type": "response.output_text.delta",
                "item_id": message_id,
                "output_index": 0,
                "content_index": 0,
                "delta": delta,
            },
        )
    yield _event(
        "response.output_text.done",
        {
            "type": "response.output_text.done",
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "text": sample.text,
        },
    )
    status = "completed" if sample.finish_reason == "stop" else "incomplete"
    yield _event(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                **skeleton,
                "status": status,
                "output": [
                    {
                        "type": "message",
                        "id": message_id,
                        "role": "assistant",
                        "status": status,
                        "content": [
                            {
                                "type": "output_text",
                                "text": sample.text,
                                "annotations": [],
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": len(record.prompt_token_ids),
                    "output_tokens": len(record.completion_token_ids),
                    "total_tokens": len(record.token_ids),
                },
                "synth": synth_ref(records).model_dump(),
            },
        },
    )

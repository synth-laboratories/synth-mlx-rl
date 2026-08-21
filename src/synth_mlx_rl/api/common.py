"""Shared surface plumbing.

Both API families call exactly these helpers, so "one renderer, one rollout
record" is a property of the code rather than a note in a design document. If a
new surface is added later, it goes through here too, or it does not go in.
"""

from __future__ import annotations

from typing import Iterator, Sequence

from fastapi import HTTPException, Request

from ..protocols import LearnerEngine
from ..rollouts import RolloutRecord
from ..schemas import ChatMessage, Sample, SampleRequest, ToolDefinition
from ..snapshots import SnapshotEvictedError, SnapshotNotFoundError
from .normalize import ProtocolError
from .openai_schemas import SynthRolloutRef

POLICY_PIN_HEADER = "X-Policy-Pin"
PROXY_REQUEST_ID_HEADER = "X-Proxy-Request-Id"
IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"


def get_engine(request: Request) -> LearnerEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status_code=503, detail="engine is not ready")
    return engine


def http_error(code: str, message: str, status_code: int) -> HTTPException:
    return HTTPException(
        status_code=status_code, detail={"error_code": code, "message": message}
    )


def snapshot_http_error(exc: Exception) -> HTTPException:
    """Map a snapshot refusal to a status code. There is no fallback branch."""

    if isinstance(exc, SnapshotEvictedError):
        return http_error("policy_snapshot_evicted", str(exc), 410)
    if isinstance(exc, SnapshotNotFoundError):
        return http_error("policy_snapshot_not_found", str(exc), 404)
    return http_error("policy_snapshot_error", str(exc), 409)


def protocol_http_error(exc: ProtocolError) -> HTTPException:
    return http_error(exc.code, str(exc), exc.status_code)


def resolve_policy_pin(
    request: Request, body_snapshot_id: str | None
) -> str | None:
    """Read the pin from the body or the ``X-Policy-Pin`` header.

    Banking77's responses path already demands this header, so the responses
    family is where a snapshot id naturally travels. Disagreement between the
    two sources is refused rather than resolved by precedence: silently
    preferring one would mean a caller can believe it pinned a policy it did
    not pin.
    """

    header_value = request.headers.get(POLICY_PIN_HEADER)
    header_value = header_value.strip() if header_value else None
    if header_value and body_snapshot_id and header_value != body_snapshot_id:
        raise http_error(
            "policy_pin_conflict",
            f"{POLICY_PIN_HEADER}={header_value!r} disagrees with "
            f"policy_snapshot_id={body_snapshot_id!r}",
            409,
        )
    return body_snapshot_id or header_value


def idempotency_key(request: Request) -> str | None:
    value = request.headers.get(IDEMPOTENCY_KEY_HEADER)
    value = value.strip() if value else None
    return value or None


def check_model(engine: LearnerEngine, requested: str | None) -> str:
    resident = engine.state().model
    if requested is not None and requested != resident:
        raise http_error(
            "model_not_resident",
            f"resident model is {resident!r}, not {requested!r}; this service "
            "holds exactly one base model",
            400,
        )
    return resident


def sample_once(
    engine: LearnerEngine,
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[ToolDefinition] | None,
    api_family: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    top_k: int,
    num_samples: int,
    seed: int | None,
    stop: str | list[str] | None,
    enable_thinking: bool | None,
    policy_snapshot_id: str | None,
    idempotency_key: str | None = None,
) -> tuple[list[Sample], list[RolloutRecord]]:
    """The single sampling call both surfaces make.

    Everything above this line differs between the families; nothing below it
    does. The rollout records come straight back out so a surface can report the
    digests it actually rendered with.
    """

    try:
        request = SampleRequest(
            messages=list(messages),
            tools=list(tools) if tools else None,
            add_generation_prompt=True,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            num_samples=num_samples,
            seed=seed,
            stop=stop,
            enable_thinking=enable_thinking,
            policy_snapshot_id=policy_snapshot_id,
            api_family=api_family,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise http_error("invalid_request", str(exc), 422) from exc

    try:
        response = engine.sample(request, idempotency_key=idempotency_key)
    except (SnapshotNotFoundError, SnapshotEvictedError) as exc:
        raise snapshot_http_error(exc) from exc
    except ValueError as exc:
        raise http_error("invalid_request", str(exc), 400) from exc

    records = [
        engine.rollouts.get(sample.proxy_request_id) for sample in response.samples
    ]
    return response.samples, records


def synth_ref(records: Sequence[RolloutRecord]) -> SynthRolloutRef:
    head = records[0]
    return SynthRolloutRef(
        proxy_request_ids=[record.proxy_request_id for record in records],
        policy_snapshot_id=head.policy_snapshot_id,
        training_version=head.training_version,
        api_family=head.api_family,
        tokenizer_digest=head.tokenizer_digest,
        template_digest=head.template_digest,
        render_digest=head.render_digest,
    )


def response_headers(records: Sequence[RolloutRecord]) -> dict[str, str]:
    head = records[0]
    return {
        PROXY_REQUEST_ID_HEADER: ",".join(
            record.proxy_request_id for record in records
        ),
        POLICY_PIN_HEADER: head.policy_snapshot_id,
    }


def token_text_deltas(
    engine: LearnerEngine, completion_token_ids: Sequence[int]
) -> Iterator[str]:
    """Yield the text each completion token contributed, in order.

    Cumulative decoding rather than per-token decoding, because a token is not
    guaranteed to be a whole character: decoding one at a time turns multi-byte
    text into replacement characters.

    Cumulative decoding alone is not enough. When a multi-byte character
    straddles two tokens, the decode after the first one ends in U+FFFD, and
    once that has been yielded it cannot be taken back -- the completed
    character that arrives next is the same length, so the naive "only emit
    what grew" rule emits nothing and the stream keeps a replacement character
    the non-streamed response does not have. A trailing U+FFFD is therefore
    held back until a later token completes it, and flushed at the end if the
    model genuinely emitted one.
    """

    emitted = ""
    prefix: list[int] = []
    text = ""
    for token_id in completion_token_ids:
        prefix.append(int(token_id))
        text = engine.decode(prefix, skip_special_tokens=True)
        stable = text.rstrip("\ufffd")
        if stable == emitted:
            continue
        if stable.startswith(emitted):
            yield stable[len(emitted) :]
        else:
            # A detokenizer that rewrote earlier text: re-emit from scratch
            # rather than pretending the earlier bytes were right.
            yield stable[len(emitted) :] if len(stable) > len(emitted) else ""
        emitted = stable
    if len(text) > len(emitted) and text.startswith(emitted):
        yield text[len(emitted) :]

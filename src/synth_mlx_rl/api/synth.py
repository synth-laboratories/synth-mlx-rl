"""The ``/v1/synth`` namespace: everything OpenAI does not define.

The rollout record is the training authority. A container carries exactly one
joinable fact out of a rollout -- the proxy request id -- and the trainer comes
here for the tokens and the log-probabilities. That split is what stops a
container from being trusted with a training record it does not own.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import Field

from ..rollouts import (
    RolloutRecordNotFoundError,
    RolloutRecordQuery,
    RolloutRecordQueryResponse,
    RolloutRecordResponse,
)
from ..mismatch import (
    MismatchError,
    MismatchPolicy,
    measure_mismatch,
)
from ..mismatch import tis_weights as compute_tis_weights
from ..schemas import (
    LogprobsRequest,
    LogprobsResponse,
    MismatchRequest,
    MismatchResponse,
    StrictModel,
)
from ..snapshots import SnapshotError
from .common import get_engine, http_error, snapshot_http_error

router = APIRouter(prefix="/v1/synth")


class PublishSnapshotRequest(StrictModel):
    snapshot_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class SnapshotListResponse(StrictModel):
    snapshots: list[dict[str, Any]]
    capacity: int
    latest_policy_snapshot_id: str | None = None


@router.get("/capability")
def capability(request: Request) -> dict[str, Any]:
    """What this service can supply to a token-level RL pipeline.

    Deliberately explicit about what is *not* here, so a caller planning a run
    does not discover it after spending compute.
    """

    engine = get_engine(request)
    state = engine.state()
    from ..objective_spec import SUPPORTED_OBJECTIVES, UNAVAILABLE_OBJECTIVES

    return {
        "service": "synth-mlx-rl",
        "model": state.model,
        "api_families": ["chat_completions", "responses"],
        "token_emission": {
            "token_ids": True,
            "logprobs": True,
            "logits": False,
            "top_logprobs": False,
            "old_logprobs": True,
        },
        "rollout_records": {
            "retrievable": True,
            "join_key": "proxy_request_id",
            "rollout_logprobs_pre_truncation": True,
            "capacity": engine.rollouts.capacity,
            "resident": len(engine.rollouts),
        },
        "policy_snapshots": {
            "immutable": True,
            "pinned_per_request": True,
            "capacity": engine.snapshots.capacity,
            "resident": len(engine.snapshots),
            "latest": state.latest_policy_snapshot_id,
        },
        "objectives": {
            "supported": list(SUPPORTED_OBJECTIVES),
            "unavailable": dict(UNAVAILABLE_OBJECTIVES),
        },
        "tokenizer_digest": state.tokenizer_digest,
        "template_digest": state.template_digest,
    }


@router.get("/rollouts/{proxy_request_id}", response_model=RolloutRecordResponse)
def get_rollout(proxy_request_id: str, request: Request) -> RolloutRecordResponse:
    engine = get_engine(request)
    try:
        return RolloutRecordResponse(record=engine.rollouts.get(proxy_request_id))
    except RolloutRecordNotFoundError as exc:
        raise http_error("rollout_record_not_found", str(exc), 404) from exc


@router.post("/rollouts/query", response_model=RolloutRecordQueryResponse)
def query_rollouts(
    body: RolloutRecordQuery, request: Request
) -> RolloutRecordQueryResponse:
    """Fetch a whole episode's records at once.

    ``missing`` is returned rather than raising: a multi-turn episode with one
    dropped record is a partially usable episode, and the caller is the one who
    decides whether to refuse it.
    """

    engine = get_engine(request)
    records, missing = engine.rollouts.get_many(body.proxy_request_ids)
    return RolloutRecordQueryResponse(records=records, missing=missing)


@router.get("/snapshots", response_model=SnapshotListResponse)
def list_snapshots(request: Request) -> SnapshotListResponse:
    engine = get_engine(request)
    snapshots = engine.snapshots.list()
    latest = engine.snapshots.latest()
    return SnapshotListResponse(
        snapshots=[snapshot.describe() for snapshot in snapshots],
        capacity=engine.snapshots.capacity,
        latest_policy_snapshot_id=None if latest is None else latest.id,
    )


@router.post("/snapshots")
def publish_snapshot(body: PublishSnapshotRequest, request: Request) -> dict[str, Any]:
    """Freeze the current training adapter under a new immutable id."""

    engine = get_engine(request)
    try:
        snapshot = engine.publish_snapshot(
            snapshot_id=body.snapshot_id, metadata=body.metadata
        )
    except SnapshotError as exc:
        raise snapshot_http_error(exc) from exc
    return snapshot.describe()


@router.get("/snapshots/{snapshot_id}")
def get_snapshot(snapshot_id: str, request: Request) -> dict[str, Any]:
    engine = get_engine(request)
    try:
        return engine.snapshots.get(snapshot_id).describe()
    except SnapshotError as exc:
        raise snapshot_http_error(exc) from exc


@router.delete("/snapshots/{snapshot_id}")
def evict_snapshot(snapshot_id: str, request: Request) -> dict[str, Any]:
    engine = get_engine(request)
    try:
        engine.snapshots.evict(snapshot_id)
    except SnapshotError as exc:
        raise snapshot_http_error(exc) from exc
    return {"evicted": snapshot_id}


@router.post("/logprobs", response_model=LogprobsResponse)
def logprobs(body: LogprobsRequest, request: Request) -> LogprobsResponse:
    """Score a sequence under a pinned snapshot: the behavior log-probabilities.

    A different population from the sampler's ``rollout_logprobs`` on the record.
    The ratio denominator comes from here; the mismatch check compares the two.
    """

    engine = get_engine(request)
    try:
        values = engine.score_logprobs(
            body.token_ids, policy_snapshot_id=body.policy_snapshot_id
        )
    except SnapshotError as exc:
        raise snapshot_http_error(exc) from exc
    return LogprobsResponse(
        logprobs=values, policy_snapshot_id=body.policy_snapshot_id
    )


@router.post("/mismatch", response_model=MismatchResponse)
def mismatch(body: MismatchRequest, request: Request) -> MismatchResponse:
    """Close the logprob lifecycle for one recorded call, in one round trip.

    Fetch the record, recompute `behavior_logprobs` under the snapshot it was
    actually sampled under, align them to the completion tokens, measure the
    disagreement, and return a verdict. This exists as one endpoint because the
    steps are only meaningful together: behavior logprobs scored under a
    *different* snapshot than the record names measure nothing, and a caller
    assembling the sequence by hand is one mistake away from a confident number
    about the wrong comparison.

    The three verdicts mean different things to the caller:
      ok                the two paths agree; train as collected
      correct_with_tis  they disagree tolerably; `tis_weights` is returned
      refuse            the collection is disqualified; do not train on it
    """

    engine = get_engine(request)
    try:
        record = engine.rollouts.get(body.proxy_request_id)
    except RolloutRecordNotFoundError as exc:
        raise http_error("rollout_record_not_found", str(exc), 404) from exc

    snapshot_id = body.policy_snapshot_id or record.policy_snapshot_id
    sequence = list(record.prompt_token_ids) + list(record.completion_token_ids)
    try:
        scored = engine.score_logprobs(sequence, policy_snapshot_id=snapshot_id)
    except SnapshotError as exc:
        raise snapshot_http_error(exc) from exc

    # `score_logprobs` returns one entry per token with a leading None: the
    # first token of the sequence has no predecessor to be predicted from. The
    # completion's behavior logprobs are the tail, and taking that slice wrong
    # is exactly the off-by-one that makes a mismatch metric look plausible.
    completion_length = len(record.completion_token_ids)
    behavior = [
        0.0 if value is None else float(value)
        for value in scored[len(sequence) - completion_length :]
    ]

    policy = MismatchPolicy(
        **{
            key: value
            for key, value in (
                ("ok_abs_diff", body.ok_abs_diff),
                ("max_abs_diff", body.max_abs_diff),
                ("min_ess_ratio", body.min_ess_ratio),
            )
            if value is not None
        }
    )
    try:
        report = policy.evaluate(
            measure_mismatch(
                behavior_logprobs=behavior,
                rollout_logprobs=record.rollout_logprobs,
            )
        )
    except MismatchError as exc:
        raise http_error("mismatch_unmeasurable", str(exc), 422) from exc

    weights = None
    if report.verdict == "correct_with_tis":
        weights = [
            float(value)
            for value in compute_tis_weights(
                behavior_logprobs=behavior,
                rollout_logprobs=record.rollout_logprobs,
                clip_low=body.tis_clip_low,
                clip_high=body.tis_clip_high,
            ).weights
        ]

    return MismatchResponse(
        proxy_request_id=record.proxy_request_id,
        policy_snapshot_id=snapshot_id,
        behavior_logprobs=behavior,
        rollout_logprobs=list(record.rollout_logprobs),
        report=report.to_dict(),
        tis_weights=weights,
    )

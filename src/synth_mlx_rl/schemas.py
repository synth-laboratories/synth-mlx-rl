"""Core service schemas.

Derived from the MIT-licensed `mlx-local-rl` prototype (see NOTICE). Changed
from the prototype:

* ``Datum.old_logprobs`` is now ``behavior_logprobs``, and ``Sample`` reports
  ``rollout_logprobs``. The finalized plan's section 5.3 makes those two
  distinct populations, and one name for both is how they get conflated.
* ``loss_fn`` is an objective name from :mod:`synth_mlx_rl.objective_spec`, and
  ``ppo`` is refused with a reason rather than silently aliased to ``grpo``.
* Sampling carries a policy snapshot pin and returns a proxy request id.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .objective_spec import (
    POLICY_OBJECTIVES,
    UNAVAILABLE_OBJECTIVES,
    ObjectiveError,
    ObjectiveSpec,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class FunctionCall(StrictModel):
    name: str
    arguments: str = ""


class ToolCall(StrictModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(StrictModel):
    """One canonical message.

    Both API families normalize into this before anything is rendered, so the
    renderer has exactly one input shape to reason about.
    """

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def validate_content(self) -> "ChatMessage":
        if self.content is None and not self.tool_calls:
            raise ValueError("a message needs content or tool_calls")
        if self.role == "tool" and self.tool_call_id is None:
            raise ValueError("a tool message must carry tool_call_id")
        return self


class ToolFunction(StrictModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDefinition(StrictModel):
    """A tool in Chat Completions shape.

    The Responses family sends the flat form (``{type, name, parameters}``);
    normalization folds it into this nested form so the renderer sees one shape.
    """

    type: Literal["function"] = "function"
    function: ToolFunction


class Datum(StrictModel):
    """One pre-shifted training example.

    ``input_ids[t]`` is fed to the model and ``target_ids[t]`` is the token
    scored at that position, so no downstream code has to remember to shift.
    Policy examples additionally carry a behavior log-probability (the ratio
    denominator) and an advantage for every target position, or a scalar
    advantage broadcast across the sampled span.
    """

    input_ids: list[int] = Field(min_length=1)
    target_ids: list[int] = Field(min_length=1)
    weights: list[float] = Field(min_length=1)
    behavior_logprobs: list[float] | None = None
    advantages: list[float] | float | None = None
    reference_logprobs: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_alignment(self) -> "Datum":
        n = len(self.input_ids)
        if len(self.target_ids) != n or len(self.weights) != n:
            raise ValueError(
                "input_ids, target_ids, and weights must have equal lengths"
            )
        if any(token_id < 0 for token_id in self.input_ids + self.target_ids):
            raise ValueError("token IDs must be non-negative")
        if any(weight < 0.0 for weight in self.weights):
            raise ValueError("weights must be non-negative")
        if not any(weight > 0.0 for weight in self.weights):
            raise ValueError("weights must select at least one target token")
        if self.behavior_logprobs is not None and len(self.behavior_logprobs) != n:
            raise ValueError("behavior_logprobs must align with target_ids")
        if isinstance(self.advantages, list) and len(self.advantages) != n:
            raise ValueError("token-level advantages must align with target_ids")
        if self.reference_logprobs is not None and len(self.reference_logprobs) != n:
            raise ValueError("reference_logprobs must align with target_ids")
        return self


class ForwardBackwardRequest(StrictModel):
    data: list[Datum] = Field(min_length=1)
    loss_fn: str = "cross_entropy"
    #: How the per-token loss is reduced. This decides step size, and the two
    #: conventions differ by a factor of the token count -- hundreds, typically.
    #:
    #:   mean_tokens  sum(per-token) / unmasked tokens.  The default here, what
    #:                SLIME recommends for SFT and what TRL/HF do. Step size is
    #:                independent of sequence length.
    #:   sum          sum(per-token), no division. The Tinker convention. A
    #:                600-token trace pushes ~600x harder than a 1-token one.
    #:
    #: A ported Tinker script tuned at lr=1e-4 under `sum` will behave nothing
    #: like the same script here under `mean_tokens`, and nothing errors -- so
    #: the field is explicit rather than inferred.
    reduction: Literal["mean_tokens", "sum"] = "mean_tokens"
    clip_epsilon: float = Field(default=0.2, gt=0.0, lt=1.0)
    eps_low: float = Field(default=1.0, ge=0.0)
    eps_high: float = Field(default=4.0, gt=0.0)
    kl_beta: float = Field(default=0.0, ge=0.0)
    entropy_coef: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def validate_objective(self) -> "ForwardBackwardRequest":
        if self.loss_fn in UNAVAILABLE_OBJECTIVES:
            raise ValueError(UNAVAILABLE_OBJECTIVES[self.loss_fn])
        try:
            self.objective()
        except ObjectiveError as exc:
            raise ValueError(str(exc)) from exc
        if self.loss_fn in POLICY_OBJECTIVES:
            for index, datum in enumerate(self.data):
                if datum.behavior_logprobs is None:
                    raise ValueError(
                        f"data[{index}].behavior_logprobs is required for "
                        f"{self.loss_fn}"
                    )
                if datum.advantages is None:
                    raise ValueError(
                        f"data[{index}].advantages is required for {self.loss_fn}"
                    )
        return self

    def objective(self) -> ObjectiveSpec:
        return ObjectiveSpec(
            name=self.loss_fn,
            clip_epsilon=self.clip_epsilon,
            eps_low=self.eps_low,
            eps_high=self.eps_high,
            kl_beta=self.kl_beta,
            entropy_coef=self.entropy_coef,
        )


class ForwardBackwardResponse(StrictModel):
    loss: float
    metrics: dict[str, float]
    accumulation_count: int
    training_version: int


class AdamParams(StrictModel):
    learning_rate: float = Field(default=5e-5, gt=0.0)
    beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    beta2: float = Field(default=0.999, gt=0.0, lt=1.0)
    eps: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    bias_correction: bool = False
    max_grad_norm: float | None = Field(default=1.0, gt=0.0)


class OptimStepRequest(StrictModel):
    params: AdamParams = Field(default_factory=AdamParams)


class OptimStepResponse(StrictModel):
    step: int
    training_version: int
    grad_norm: float
    applied_accumulations: int
    learning_rate: float


class TokenizeRequest(StrictModel):
    text: str
    add_special_tokens: bool = True


class TokenizeResponse(StrictModel):
    token_ids: list[int]


class DetokenizeRequest(StrictModel):
    token_ids: list[int]
    skip_special_tokens: bool = False

    @model_validator(mode="after")
    def validate_token_ids(self) -> "DetokenizeRequest":
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be non-negative")
        return self


class DetokenizeResponse(StrictModel):
    text: str


class RenderChatRequest(StrictModel):
    messages: list[ChatMessage] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    add_generation_prompt: bool = False
    tokenize: bool = True
    enable_thinking: bool | None = None


class RenderChatResponse(StrictModel):
    token_ids: list[int] | None = None
    text: str | None = None
    tokenizer_digest: str
    template_digest: str
    render_digest: str
    enable_thinking: bool


class SamplingParamsModel(StrictModel):
    """The sampling knobs, recorded verbatim on every rollout record."""

    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    seed: int | None = Field(default=None, ge=0)
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None


class SampleRequest(StrictModel):
    prompt: str | None = None
    messages: list[ChatMessage] | None = None
    prompt_token_ids: list[int] | None = None
    tools: list[ToolDefinition] | None = None
    add_generation_prompt: bool = True
    max_tokens: int = Field(default=128, ge=1, le=32768)
    temperature: float = Field(default=0.7, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    num_samples: int = Field(default=1, ge=1, le=64)
    seed: int | None = Field(default=None, ge=0)
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    enable_thinking: bool | None = None
    #: Resolved once, at request start, and pinned for the whole completion.
    #: ``None`` means "the newest published snapshot"; it never means "whatever
    #: the trainer happens to hold when the token is generated".
    policy_snapshot_id: str | None = None
    api_family: Literal["chat_completions", "responses", "native"] = "native"

    @model_validator(mode="after")
    def validate_prompt_source(self) -> "SampleRequest":
        supplied = sum(
            value is not None
            for value in (self.prompt, self.messages, self.prompt_token_ids)
        )
        if supplied != 1:
            raise ValueError(
                "provide exactly one of prompt, messages, or prompt_token_ids"
            )
        if self.messages is not None and not self.messages:
            raise ValueError("messages cannot be empty")
        if self.prompt_token_ids is not None and not self.prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")
        if self.tools and self.messages is None:
            raise ValueError("tools require messages")
        token_lists = [
            values
            for values in (self.prompt_token_ids, self.stop_token_ids)
            if values is not None
        ]
        if any(token_id < 0 for values in token_lists for token_id in values):
            raise ValueError("token IDs must be non-negative")
        return self

    def sampling_params(self) -> SamplingParamsModel:
        stop = [self.stop] if isinstance(self.stop, str) else self.stop
        return SamplingParamsModel(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=list(stop) if stop else None,
            stop_token_ids=(
                list(self.stop_token_ids) if self.stop_token_ids else None
            ),
        )


class Sample(StrictModel):
    """A completion plus the sampler's raw, unfiltered log-probabilities.

    ``rollout_logprobs`` are read from the model's full next-token distribution
    *before* top-p / top-k / min-p truncation. They are the mismatch input, not
    the ratio denominator; see the four-population table in the finalized plan.
    """

    text: str
    prompt_token_ids: list[int]
    completion_token_ids: list[int]
    rollout_logprobs: list[float]
    finish_reason: Literal["stop", "length"]
    policy_snapshot_id: str
    training_version: int
    proxy_request_id: str


class SampleResponse(StrictModel):
    samples: list[Sample]


class LogprobsRequest(StrictModel):
    token_ids: list[int] = Field(min_length=2)
    policy_snapshot_id: str | None = None

    @model_validator(mode="after")
    def validate_token_ids(self) -> "LogprobsRequest":
        if any(token_id < 0 for token_id in self.token_ids):
            raise ValueError("token IDs must be non-negative")
        return self


class LogprobsResponse(StrictModel):
    # The first token has no predecessor inside the supplied sequence.
    logprobs: list[float | None]
    policy_snapshot_id: str | None = None


class MismatchRequest(StrictModel):
    """Ask the service to close the logprob lifecycle for one recorded call."""

    proxy_request_id: str
    #: Defaults to the snapshot the record was sampled under, which is the only
    #: snapshot the comparison is meaningful against. Overriding it measures
    #: something else, so it must be said out loud.
    policy_snapshot_id: str | None = None
    ok_abs_diff: float | None = None
    max_abs_diff: float | None = None
    min_ess_ratio: float | None = None
    tis_clip_low: float = 0.5
    tis_clip_high: float = 1.5


class MismatchResponse(StrictModel):
    proxy_request_id: str
    policy_snapshot_id: str
    #: Recomputed by the trainer under the pinned snapshot. A different
    #: population from the record's `rollout_logprobs`.
    behavior_logprobs: list[float]
    rollout_logprobs: list[float]
    report: dict[str, object]
    #: Present only when the verdict is `correct_with_tis`; a caller that gets
    #: `ok` needs no correction and one that gets `refuse` must not train.
    tis_weights: list[float] | None = None


class CheckpointRequest(StrictModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class CheckpointResponse(StrictModel):
    path: str
    step: int
    training_version: int


class StateResponse(StrictModel):
    ready: bool
    model: str
    device: str
    lora_rank: int
    lora_alpha: float
    lora_scale: float
    lora_dropout: float
    num_layers: int
    trainable_parameters: int
    total_parameters: int
    step: int
    training_version: int
    accumulation_count: int
    optimizer_initialized: bool
    max_seq_length: int
    enable_thinking: bool
    tokenizer_digest: str
    template_digest: str
    latest_policy_snapshot_id: str | None = None
    resident_snapshots: int = 0

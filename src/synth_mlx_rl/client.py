"""The native HTTP client.

Derived from the MIT-licensed `mlx-local-rl` prototype (see NOTICE). The
prototype also shipped two source-compatibility shims that squatted on the
public import names ``tinker`` and ``river_client``. Those are gone: a package
that installs a module called ``tinker`` breaks any environment that also has
the real one, and there is no version of that trade that is worth making. Only
this native client remains.

Behavioral change from the prototype:
``save_weights_and_get_sampling_client`` now publishes a real immutable snapshot
and returns a sampling client pinned to it. The prototype's version returned the
live resident policy and said so in its docstring; that is exactly the semantics
decision D2 forbids.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar, cast

import httpx
from pydantic import BaseModel

from .rollouts import RolloutRecord
from .schemas import (
    AdamParams,
    ChatMessage,
    CheckpointResponse,
    Datum,
    DetokenizeResponse,
    ForwardBackwardRequest,
    ForwardBackwardResponse,
    LogprobsResponse,
    OptimStepRequest,
    OptimStepResponse,
    RenderChatRequest,
    RenderChatResponse,
    Sample,
    SampleRequest,
    SampleResponse,
    StateResponse,
    TokenizeResponse,
    ToolDefinition,
)

T = TypeVar("T")


class APIFuture(Generic[T]):
    """A small ``.result()`` wrapper over one ordered submission queue."""

    def __init__(self, future: Future[T]):
        self._future = future

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)

    def done(self) -> bool:
        return self._future.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    async def result_async(self) -> T:
        return await asyncio.wrap_future(self._future)

    def __await__(self):  # type: ignore[no-untyped-def]
        return self.result_async().__await__()


class _MappedFuture(APIFuture[T]):
    def __init__(self, source: APIFuture[Any], transform: Any):
        self._source = source
        self._transform = transform

    def result(self, timeout: float | None = None) -> T:
        return cast(T, self._transform(self._source.result(timeout)))

    def done(self) -> bool:
        return self._source.done()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._source.exception(timeout)

    async def result_async(self) -> T:
        return cast(T, self._transform(await self._source.result_async()))


@dataclass(frozen=True, slots=True)
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    seed: int | None = None
    stop: tuple[str, ...] | None = None
    stop_token_ids: tuple[int, ...] | None = None
    enable_thinking: bool | None = None


class _Transport:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)
        # One worker is intentional: forward_backward followed immediately by
        # optim_step must preserve submission order, like a stateful training
        # queue. A pool would let a step overtake a backward.
        self.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="synth-mlx-rl-client"
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
        self.http.close()

    def get(self, path: str, response_type: type[T]) -> T:
        response = self.http.get(path)
        response.raise_for_status()
        return self._parse(response.json(), response_type)

    def post(self, path: str, payload: dict[str, Any], response_type: type[T]) -> T:
        response = self.http.post(path, json=payload)
        response.raise_for_status()
        return self._parse(response.json(), response_type)

    @staticmethod
    def _parse(payload: Any, response_type: type[T]) -> T:
        if isinstance(response_type, type) and issubclass(response_type, BaseModel):
            return cast(T, response_type.model_validate(payload))
        return cast(T, payload)

    def submit_post(
        self, path: str, payload: dict[str, Any], response_type: type[T]
    ) -> APIFuture[T]:
        return APIFuture(self.executor.submit(self.post, path, payload, response_type))


class RemoteTokenizer:
    def __init__(self, transport: _Transport):
        self._transport = transport

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self._transport.post(
            "/v1/tokenize",
            {"text": text, "add_special_tokens": add_special_tokens},
            TokenizeResponse,
        ).token_ids

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self._transport.post(
            "/v1/detokenize",
            {
                "token_ids": list(token_ids),
                "skip_special_tokens": skip_special_tokens,
            },
            DetokenizeResponse,
        ).text

    def render_chat(
        self,
        messages: Sequence[dict[str, Any] | ChatMessage],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
    ) -> RenderChatResponse:
        """Render through the server's one renderer, digests included."""

        normalized = [
            message if isinstance(message, ChatMessage) else ChatMessage.model_validate(message)
            for message in messages
        ]
        return self._transport.post(
            "/v1/render_chat",
            RenderChatRequest(
                messages=normalized,
                tools=list(tools) if tools else None,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=enable_thinking,
            ).model_dump(),
            RenderChatResponse,
        )

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, Any] | ChatMessage],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        enable_thinking: bool | None = None,
    ) -> list[int] | str:
        response = self.render_chat(
            messages,
            tools=tools,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
        if tokenize:
            assert response.token_ids is not None
            return response.token_ids
        assert response.text is not None
        return response.text


class SamplingClient:
    """A sampling handle pinned to one immutable policy snapshot.

    ``policy_snapshot_id`` is carried on every request rather than resolved per
    call, so a training step that lands between two calls cannot move the policy
    underneath a running episode.
    """

    def __init__(self, transport: _Transport, policy_snapshot_id: str | None = None):
        self._transport = transport
        self.policy_snapshot_id = policy_snapshot_id

    def sample(
        self,
        prompt: str | Sequence[int] | Sequence[dict[str, Any] | ChatMessage],
        sampling_params: SamplingParams | None = None,
        *,
        num_samples: int = 1,
        add_generation_prompt: bool = True,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> APIFuture[SampleResponse]:
        params = sampling_params or SamplingParams()
        common: dict[str, Any] = {
            "add_generation_prompt": add_generation_prompt,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "min_p": params.min_p,
            "top_k": params.top_k,
            "num_samples": num_samples,
            "seed": params.seed,
            "stop": list(params.stop) if params.stop is not None else None,
            "stop_token_ids": (
                list(params.stop_token_ids)
                if params.stop_token_ids is not None
                else None
            ),
            "enable_thinking": params.enable_thinking,
            "policy_snapshot_id": self.policy_snapshot_id,
            "api_family": "native",
        }
        if isinstance(prompt, str):
            common["prompt"] = prompt
        else:
            values = list(prompt)
            if not values:
                raise ValueError("prompt cannot be empty")
            if all(isinstance(value, int) for value in values):
                common["prompt_token_ids"] = [int(value) for value in values]
            else:
                common["messages"] = [
                    value.model_dump()
                    if isinstance(value, ChatMessage)
                    else dict(value)  # type: ignore[arg-type]
                    for value in values
                ]
                if tools:
                    common["tools"] = [tool.model_dump() for tool in tools]
        request = SampleRequest.model_validate(common)
        return self._transport.submit_post(
            "/v1/sample", request.model_dump(), SampleResponse
        )

    def compute_logprobs(self, prompt: Sequence[int]) -> APIFuture[list[float | None]]:
        """Behavior log-probabilities under this client's pinned snapshot."""

        future = self._transport.submit_post(
            "/v1/synth/logprobs",
            {
                "token_ids": [int(token_id) for token_id in prompt],
                "policy_snapshot_id": self.policy_snapshot_id,
            },
            LogprobsResponse,
        )
        return _MappedFuture(future, lambda response: response.logprobs)

    def get_rollout_record(self, proxy_request_id: str) -> RolloutRecord:
        payload = self._transport.get(
            f"/v1/synth/rollouts/{proxy_request_id}", dict  # type: ignore[arg-type]
        )
        return RolloutRecord.model_validate(payload["record"])

    def get_rollout_records(
        self, proxy_request_ids: Sequence[str]
    ) -> tuple[list[RolloutRecord], list[str]]:
        payload = self._transport.post(
            "/v1/synth/rollouts/query",
            {"proxy_request_ids": [str(x) for x in proxy_request_ids]},
            dict,  # type: ignore[arg-type]
        )
        return (
            [RolloutRecord.model_validate(item) for item in payload["records"]],
            list(payload["missing"]),
        )


class TrainingClient:
    def __init__(self, transport: _Transport):
        self._transport = transport
        self._tokenizer = RemoteTokenizer(transport)

    def get_tokenizer(self) -> RemoteTokenizer:
        return self._tokenizer

    def forward_backward(
        self,
        data: Sequence[Datum],
        loss_fn: str = "cross_entropy",
        *,
        clip_epsilon: float = 0.2,
        eps_low: float = 1.0,
        eps_high: float = 4.0,
        kl_beta: float = 0.0,
        entropy_coef: float = 0.0,
    ) -> APIFuture[ForwardBackwardResponse]:
        request = ForwardBackwardRequest(
            data=list(data),
            loss_fn=loss_fn,
            clip_epsilon=clip_epsilon,
            eps_low=eps_low,
            eps_high=eps_high,
            kl_beta=kl_beta,
            entropy_coef=entropy_coef,
        )
        return self._transport.submit_post(
            "/v1/forward_backward", request.model_dump(), ForwardBackwardResponse
        )

    def optim_step(
        self, params: AdamParams | None = None
    ) -> APIFuture[OptimStepResponse]:
        request = OptimStepRequest(params=params or AdamParams())
        return self._transport.submit_post(
            "/v1/optim_step", request.model_dump(), OptimStepResponse
        )

    def zero_grad(self) -> APIFuture[dict[str, int]]:
        return self._transport.submit_post(
            "/v1/zero_grad", {}, dict  # type: ignore[arg-type]
        )

    def save_state(self, name: str) -> APIFuture[CheckpointResponse]:
        return self._transport.submit_post(
            "/v1/checkpoints/save", {"name": name}, CheckpointResponse
        )

    def load_state(self, name: str) -> APIFuture[CheckpointResponse]:
        return self._transport.submit_post(
            "/v1/checkpoints/load", {"name": name}, CheckpointResponse
        )

    def publish_snapshot(
        self,
        *,
        snapshot_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._transport.post(
            "/v1/synth/snapshots",
            {"snapshot_id": snapshot_id, "metadata": metadata or {}},
            dict,  # type: ignore[arg-type]
        )

    def save_weights_and_get_sampling_client(
        self, name: str | None = None
    ) -> SamplingClient:
        """Optionally checkpoint, freeze a snapshot, and pin a client to it.

        The returned client keeps sampling against the frozen adapter for as
        long as it is used, whatever the trainer does next. That is what makes
        the ratio denominator well defined for the round that follows.
        """

        if name is not None:
            self.save_state(name).result()
        snapshot = self.publish_snapshot(metadata={"reason": "sampler_refresh"})
        return SamplingClient(self._transport, snapshot["policy_snapshot_id"])

    # -- datum construction ---------------------------------------------

    def datum_from_tokens(
        self,
        token_ids: Sequence[int],
        *,
        weights: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Datum:
        tokens = [int(token_id) for token_id in token_ids]
        if len(tokens) < 2:
            raise ValueError("at least two tokens are required")
        target_weights = (
            [1.0] * (len(tokens) - 1)
            if weights is None
            else [float(weight) for weight in weights]
        )
        return Datum(
            input_ids=tokens[:-1],
            target_ids=tokens[1:],
            weights=target_weights,
            metadata=metadata or {},
        )

    def datum_from_messages(
        self,
        messages: Sequence[dict[str, Any] | ChatMessage],
        *,
        assistant_only: bool = True,
        enable_thinking: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Datum:
        """Build an SFT datum, masking everything before the assistant turn.

        The prompt-prefix and full-conversation renders must use the same
        thinking mode, or the shared-prefix search below silently finds the
        wrong boundary and the loss trains on template text.
        """

        normalized = [
            message if isinstance(message, ChatMessage) else ChatMessage.model_validate(message)
            for message in messages
        ]
        if not normalized or normalized[-1].role != "assistant":
            raise ValueError("the final message must be an assistant response")

        full_ids = cast(
            list[int],
            self._tokenizer.apply_chat_template(
                normalized,
                tokenize=True,
                add_generation_prompt=False,
                enable_thinking=enable_thinking,
            ),
        )
        if len(full_ids) < 2:
            raise ValueError("rendered conversation is too short")

        if assistant_only:
            prompt_ids = cast(
                list[int],
                self._tokenizer.apply_chat_template(
                    normalized[:-1],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                ),
            )
            boundary = 0
            for left, right in zip(prompt_ids, full_ids):
                if left != right:
                    break
                boundary += 1
            if boundary < max(1, len(prompt_ids) - 4):
                raise ValueError(
                    "generation-prompt and full-conversation templates do not "
                    "share a stable prefix"
                )
            weights = [
                1.0 if target_index >= boundary else 0.0
                for target_index in range(1, len(full_ids))
            ]
        else:
            weights = [1.0] * (len(full_ids) - 1)

        return Datum(
            input_ids=full_ids[:-1],
            target_ids=full_ids[1:],
            weights=weights,
            metadata=metadata or {},
        )

    @staticmethod
    def datum_from_sample(
        sample: Sample,
        *,
        advantage: float | Sequence[float],
        behavior_logprobs: Sequence[float] | None = None,
        reference_logprobs: Sequence[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Datum:
        """Align a completion onto pre-shifted training positions.

        The easiest thing in this system to get wrong: the first completion
        token is predicted from the *last prompt* position, so the weighted span
        starts at ``len(prompt) - 1``.

        ``behavior_logprobs`` is the ratio denominator and should come from a
        trainer forward pass at the pinned snapshot. When it is omitted, the
        sampler's ``rollout_logprobs`` stand in and the datum records that in
        ``metadata['behavior_logprob_source']``, because the two are different
        populations and a run that conflated them should be able to say so.
        """

        prompt = sample.prompt_token_ids
        completion = sample.completion_token_ids
        if not prompt or not completion:
            raise ValueError("sample must contain prompt and completion tokens")

        source = "behavior"
        if behavior_logprobs is None:
            behavior_logprobs = sample.rollout_logprobs
            source = "rollout"
        if len(behavior_logprobs) != len(completion):
            raise ValueError("behavior log probabilities are misaligned")

        full = prompt + completion
        n_targets = len(full) - 1
        completion_start = len(prompt) - 1
        weights = [0.0] * n_targets
        aligned_behavior = [0.0] * n_targets
        for offset, value in enumerate(behavior_logprobs):
            weights[completion_start + offset] = 1.0
            aligned_behavior[completion_start + offset] = float(value)

        if isinstance(advantage, Sequence) and not isinstance(advantage, (str, bytes)):
            completion_advantages = [float(value) for value in advantage]
            if len(completion_advantages) != len(completion):
                raise ValueError("token advantages must align with completion")
            advantages: list[float] | float = [0.0] * n_targets
            for offset, value in enumerate(completion_advantages):
                cast(list, advantages)[completion_start + offset] = value
        else:
            advantages = float(advantage)

        aligned_reference: list[float] | None = None
        if reference_logprobs is not None:
            reference = [float(value) for value in reference_logprobs]
            if len(reference) != len(completion):
                raise ValueError("reference logprobs must align with completion")
            aligned_reference = [0.0] * n_targets
            for offset, value in enumerate(reference):
                aligned_reference[completion_start + offset] = value

        return Datum(
            input_ids=full[:-1],
            target_ids=full[1:],
            weights=weights,
            behavior_logprobs=aligned_behavior,
            advantages=advantages,
            reference_logprobs=aligned_reference,
            metadata={
                "policy_snapshot_id": sample.policy_snapshot_id,
                "training_version": sample.training_version,
                "proxy_request_id": sample.proxy_request_id,
                "behavior_logprob_source": source,
                **(metadata or {}),
            },
        )


class ServiceClient:
    def __init__(
        self, base_url: str = "http://127.0.0.1:8787", *, timeout: float = 600.0
    ):
        self._transport = _Transport(base_url, timeout)

    def __enter__(self) -> "ServiceClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._transport.close()

    def state(self) -> StateResponse:
        return self._transport.get("/v1/state", StateResponse)

    def capability(self) -> dict[str, Any]:
        return self._transport.get("/v1/synth/capability", dict)  # type: ignore[arg-type]

    def create_lora_training_client(
        self, *, base_model: str | None = None, rank: int | None = None
    ) -> TrainingClient:
        state = self.state()
        if base_model is not None and base_model != state.model:
            raise ValueError(
                f"server model is {state.model!r}, not requested {base_model!r}"
            )
        if rank is not None and rank != state.lora_rank:
            raise ValueError(
                f"server LoRA rank is {state.lora_rank}, not requested {rank}"
            )
        return TrainingClient(self._transport)

    def create_training_client_from_state(self, checkpoint_name: str) -> TrainingClient:
        client = self.create_lora_training_client()
        client.load_state(checkpoint_name).result()
        return client

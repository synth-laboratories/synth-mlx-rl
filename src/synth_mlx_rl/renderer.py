"""The one renderer.

Decision D9 makes ``/v1/chat/completions`` and ``/v1/responses`` first-class
peers rather than one wrapping the other. That is only safe if both surfaces
share a single rendering and tokenization path: if a responses-family rollout
renders a different transcript than a chat-family one, every mixed-family
comparison in a report is invalid, and nothing in the numbers says so.

So both surfaces normalize into ``list[ChatMessage]`` plus a normalized tool
list, and then exactly one function -- :meth:`Renderer.render` -- turns that into
tokens. The result carries three digests, which is what makes a divergence
*detectable* instead of silent:

``tokenizer_digest``  identity of the tokenizer (model id, vocab size, EOS ids)
``template_digest``   identity of the chat template text itself
``render_digest``     the full render: template, thinking flag, generation-prompt
                      flag, canonical messages, canonical tools, and the token
                      ids that came out

Two requests that describe the same conversation through different API families
must produce the same ``render_digest``. The test suite asserts exactly that.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .schemas import ChatMessage, ToolDefinition

RENDER_DIGEST_VERSION = "synth-mlx-rl/render/v1"
TOKENIZER_DIGEST_VERSION = "synth-mlx-rl/tokenizer/v1"
TEMPLATE_DIGEST_VERSION = "synth-mlx-rl/chat-template/v1"


class ChatTokenizer(Protocol):
    """The slice of a Hugging Face tokenizer this package depends on."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]: ...

    def decode(
        self, token_ids: Sequence[int], *, skip_special_tokens: bool = False
    ) -> str: ...

    def apply_chat_template(self, conversation: Any, **kwargs: Any) -> Any: ...


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """The single rendering result both API families receive."""

    token_ids: list[int]
    text: str | None
    enable_thinking: bool
    add_generation_prompt: bool
    tokenizer_digest: str
    template_digest: str
    render_digest: str


def _template_message(message: dict[str, Any]) -> dict[str, Any]:
    """Adapt one wire message to what a chat template expects.

    On the wire `tool_calls[].function.arguments` is a JSON *string*, because
    that is the OpenAI format this service is compatible with. Qwen's chat
    template iterates it as a mapping, so a replayed tool-calling conversation
    raises `Can only get item pairs from a mapping` from inside Jinja -- a
    message that names neither the field nor the conversation. Parse it here and
    leave the wire format alone.
    """

    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return message
    adapted = []
    for call in tool_calls:
        function = call.get("function") if isinstance(call, dict) else None
        arguments = function.get("arguments") if isinstance(function, dict) else None
        if not isinstance(arguments, str):
            adapted.append(call)
            continue
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tool call {call.get('id') or function.get('name')!r} has arguments that "
                f"are not valid JSON, and a chat template cannot render them: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ValueError(
                f"tool call {call.get('id') or function.get('name')!r} has arguments that "
                "are valid JSON but not an object; a chat template needs named arguments"
            )
        adapted.append({**call, "function": {**function, "arguments": parsed}})
    return {**message, "tool_calls": adapted}


class Renderer:
    """Renders canonical messages to tokens, once, for every surface."""

    def __init__(
        self,
        tokenizer: ChatTokenizer,
        *,
        model: str,
        default_enable_thinking: bool = False,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model
        self._default_enable_thinking = default_enable_thinking
        self._tokenizer_digest = _tokenizer_digest(tokenizer, model)
        self._template_digest = _template_digest(tokenizer)

    @property
    def model(self) -> str:
        return self._model

    @property
    def tokenizer(self) -> ChatTokenizer:
        return self._tokenizer

    @property
    def tokenizer_digest(self) -> str:
        return self._tokenizer_digest

    @property
    def template_digest(self) -> str:
        return self._template_digest

    @property
    def default_enable_thinking(self) -> bool:
        return self._default_enable_thinking

    def render(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Sequence[ToolDefinition] | None = None,
        add_generation_prompt: bool = True,
        enable_thinking: bool | None = None,
        tokenize: bool = True,
    ) -> RenderedPrompt:
        if not messages:
            raise ValueError("messages cannot be empty")
        thinking = (
            self._default_enable_thinking if enable_thinking is None else enable_thinking
        )
        conversation = [
            _template_message(message.model_dump(exclude_none=True))
            for message in messages
        ]
        tool_payload = (
            [tool.model_dump(exclude_none=True) for tool in tools] if tools else None
        )

        kwargs: dict[str, Any] = {
            "tokenize": tokenize,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": thinking,
        }
        if tool_payload is not None:
            kwargs["tools"] = tool_payload
        rendered = self._tokenizer.apply_chat_template(conversation, **kwargs)

        if tokenize:
            token_ids = _as_token_ids(rendered)
            text = None
        else:
            text = str(rendered)
            token_ids = [
                int(t) for t in self._tokenizer.encode(text, add_special_tokens=False)
            ]

        render_digest = _sha256(
            "\n".join(
                [
                    RENDER_DIGEST_VERSION,
                    self._tokenizer_digest,
                    self._template_digest,
                    f"enable_thinking={thinking}",
                    f"add_generation_prompt={add_generation_prompt}",
                    _canonical(conversation),
                    _canonical(tool_payload),
                    _canonical(token_ids),
                ]
            )
        )
        return RenderedPrompt(
            token_ids=token_ids,
            text=text,
            enable_thinking=thinking,
            add_generation_prompt=add_generation_prompt,
            tokenizer_digest=self._tokenizer_digest,
            template_digest=self._template_digest,
            render_digest=render_digest,
        )


    def render_text(self, text: str, *, add_special_tokens: bool = True) -> RenderedPrompt:
        """Render a raw completion-style prompt, bypassing the chat template.

        Recorded with an explicit ``chat_template=<bypassed>`` marker in the
        render digest, so a raw prompt can never collide with a templated one.
        """

        token_ids = [
            int(t) for t in self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ]
        return self._raw(token_ids, source="text", detail=text)

    def render_tokens(self, token_ids: Sequence[int]) -> RenderedPrompt:
        """Accept caller-supplied prompt tokens verbatim."""

        ids = [int(token_id) for token_id in token_ids]
        return self._raw(ids, source="token_ids", detail=None)

    def _raw(
        self, token_ids: list[int], *, source: str, detail: str | None
    ) -> RenderedPrompt:
        render_digest = _sha256(
            "\n".join(
                [
                    RENDER_DIGEST_VERSION,
                    self._tokenizer_digest,
                    "chat_template=<bypassed>",
                    f"source={source}",
                    _canonical(detail),
                    _canonical(token_ids),
                ]
            )
        )
        return RenderedPrompt(
            token_ids=token_ids,
            text=detail,
            enable_thinking=self._default_enable_thinking,
            add_generation_prompt=False,
            tokenizer_digest=self._tokenizer_digest,
            template_digest=self._template_digest,
            render_digest=render_digest,
        )


def _as_token_ids(rendered: Any) -> list[int]:
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if isinstance(rendered, str):
        raise TypeError(
            "apply_chat_template returned text while tokenize=True was requested"
        )
    return [int(token_id) for token_id in rendered]


def _tokenizer_digest(tokenizer: ChatTokenizer, model: str) -> str:
    parts = [TOKENIZER_DIGEST_VERSION, model]
    for attribute in ("vocab_size", "eos_token_id", "pad_token_id", "bos_token_id"):
        parts.append(f"{attribute}={getattr(tokenizer, attribute, None)!r}")
    eos_ids = getattr(tokenizer, "eos_token_ids", None)
    if eos_ids is not None:
        parts.append("eos_token_ids=" + _canonical(sorted(int(x) for x in eos_ids)))
    return _sha256("\n".join(parts))


def _template_digest(tokenizer: ChatTokenizer) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        # No template text to hash. Say so in the digest rather than emitting a
        # hash of nothing that would silently match another template-less
        # tokenizer.
        return _sha256(TEMPLATE_DIGEST_VERSION + "\nchat_template=<unavailable>")
    return _sha256(TEMPLATE_DIGEST_VERSION + "\n" + str(template))

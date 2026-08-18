"""Both API families, funnelled into one canonical conversation.

This is the narrow waist that makes decision D9 safe. ``/v1/chat/completions``
and ``/v1/responses`` accept different payload shapes; they must not produce
different transcripts. Everything below turns a request of either family into
the same ``(list[ChatMessage], list[ToolDefinition] | None)`` pair, and after
that point there is exactly one code path.

The Responses shapes implemented here follow the org's existing native
implementation (`synth-responses-gateway`): full history forwarded every turn,
tools forwarded, and ``previous_response_id`` refused unless the caller also
sent the continuation history inline, because this service holds no
conversation state either.
"""

from __future__ import annotations

from typing import Any, Sequence

from ..schemas import (
    ChatMessage,
    FunctionCall,
    ToolCall,
    ToolDefinition,
    ToolFunction,
)


class ProtocolError(ValueError):
    """A request this service refuses to guess about."""

    def __init__(self, code: str, message: str, *, status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def normalize_chat_tools(
    tools: Sequence[ToolDefinition] | None,
) -> list[ToolDefinition] | None:
    return list(tools) if tools else None


def normalize_responses_tools(
    tools: Sequence[dict[str, Any]] | None,
) -> list[ToolDefinition] | None:
    """Fold the Responses flat tool shape into the Chat Completions nested one.

    Responses sends ``{"type": "function", "name": ..., "parameters": ...}``;
    Chat Completions sends ``{"type": "function", "function": {...}}``. Chat
    templates expect the nested form, so the flat form is converted here and
    never again -- one renderer, one tool shape.
    """

    if not tools:
        return None
    normalized: list[ToolDefinition] = []
    for index, tool in enumerate(tools):
        tool_type = tool.get("type", "function")
        if tool_type != "function":
            raise ProtocolError(
                "unsupported_tool_type",
                f"tools[{index}].type={tool_type!r} is not supported; this "
                "service implements function tools only",
            )
        if "function" in tool:
            payload = dict(tool["function"])
        else:
            payload = {
                key: value
                for key, value in tool.items()
                if key in {"name", "description", "parameters", "strict"}
            }
        if not payload.get("name"):
            raise ProtocolError(
                "tool_missing_name", f"tools[{index}] has no function name"
            )
        normalized.append(ToolDefinition(function=ToolFunction(**payload)))
    return normalized


def _content_to_text(content: Any, *, where: str) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for index, part in enumerate(content):
            if isinstance(part, str):
                parts.append(part)
                continue
            if not isinstance(part, dict):
                raise ProtocolError(
                    "unsupported_content_part",
                    f"{where}.content[{index}] is not a string or an object",
                )
            part_type = part.get("type")
            if part_type in {"input_text", "output_text", "text", "summary_text"}:
                parts.append(str(part.get("text", "")))
            elif part_type == "refusal":
                parts.append(str(part.get("refusal", "")))
            else:
                raise ProtocolError(
                    "unsupported_content_part",
                    f"{where}.content[{index}].type={part_type!r} is not "
                    "supported; this service renders text only",
                )
        return "".join(parts)
    raise ProtocolError(
        "unsupported_content", f"{where}.content must be a string or a list"
    )


def normalize_responses_input(
    value: str | Sequence[dict[str, Any]],
    *,
    instructions: str | None = None,
) -> list[ChatMessage]:
    """Turn a Responses ``input`` into canonical messages."""

    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))

    if isinstance(value, str):
        messages.append(ChatMessage(role="user", content=value))
        return messages

    for index, item in enumerate(value):
        where = f"input[{index}]"
        if not isinstance(item, dict):
            raise ProtocolError(
                "unsupported_input_item", f"{where} must be an object"
            )
        item_type = item.get("type")

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id")
            if not call_id or not item.get("name"):
                raise ProtocolError(
                    "invalid_function_call",
                    f"{where} needs call_id and name",
                )
            messages.append(
                ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ToolCall(
                            id=str(call_id),
                            function=FunctionCall(
                                name=str(item["name"]),
                                arguments=str(item.get("arguments", "")),
                            ),
                        )
                    ],
                )
            )
            continue

        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not call_id:
                raise ProtocolError(
                    "invalid_function_call_output", f"{where} needs call_id"
                )
            output = item.get("output")
            messages.append(
                ChatMessage(
                    role="tool",
                    content=output if isinstance(output, str) else str(output),
                    tool_call_id=str(call_id),
                )
            )
            continue

        if item_type in {None, "message"}:
            role = item.get("role")
            if role not in {"system", "developer", "user", "assistant", "tool"}:
                raise ProtocolError(
                    "unsupported_input_item",
                    f"{where}.role={role!r} is not a supported role",
                )
            text = _content_to_text(item.get("content"), where=where)
            messages.append(
                ChatMessage(
                    role=role,  # type: ignore[arg-type]
                    content=text,
                    tool_call_id=item.get("call_id"),
                )
            )
            continue

        raise ProtocolError(
            "unsupported_input_item",
            f"{where}.type={item_type!r} is not supported; supported items are "
            "message, function_call, and function_call_output",
        )

    if not messages:
        raise ProtocolError(
            "empty_input",
            "input produced no messages; the Responses family here is "
            "stateless and needs the full history on every turn",
        )
    return messages


def validate_previous_response_id(
    previous_response_id: str | None, value: str | Sequence[dict[str, Any]]
) -> None:
    """Refuse a continuation this service cannot honestly reconstruct.

    Same rule as the org's native Responses gateway: an id alone is refused,
    because there is no server-side conversation to resume. An id accompanied by
    the inline continuation history is accepted and ignored.
    """

    if not previous_response_id:
        return
    items = value if isinstance(value, (list, tuple)) else []
    if any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in items
    ):
        return
    raise ProtocolError(
        "previous_response_id_requires_full_history",
        "previous_response_id requires the complete continuation history "
        "(including a function_call_output item) inline in `input`; this "
        "service holds no server-side conversation state",
    )

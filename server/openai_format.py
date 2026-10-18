"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's web protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array and tool definitions into one prompt,
and wrap DeepSeek's text output back into standard OpenAI response/stream objects,
with native support for `reasoning_content` (DeepSeek-R1 / Hermes).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Iterable, List, Optional

from .schemas import ChatMessage

_ROLE_LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}


def _text_of(content) -> str:
    """Extract plain text from a message's content (string or list-of-parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for p in content:
        if isinstance(p, dict) and p.get("type") == "text":
            parts.append(p.get("text", ""))
    return "\n".join(parts)


def messages_to_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[dict]] = None,
) -> str:
    """Flatten a chat history and tool definitions into a single prompt DeepSeek can answer.

    - Supports Hermes / OpenAI tool schemas and tool calls
    - Retains tool responses and assistant tool_calls across multi-turn loops
    - Preserves system and user turns cleanly
    """
    lines = []

    # If tools are defined, inject a Hermes-compatible tool definition header
    if tools:
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
        tools_system = (
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "To use a tool, output a tool call block formatted as:\n"
            "<tool_call>\n"
            '{"name": "<function_name>", "arguments": <args_dict>}\n'
            "</tool_call>"
        )
        lines.append(f"System: {tools_system}")

    for m in messages:
        role = m.role.lower()
        content = _text_of(m.content)

        if role == "system":
            if content:
                lines.append(f"System: {content}")
        elif role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            parts = []
            if getattr(m, "reasoning_content", None):
                parts.append(f"<think>\n{m.reasoning_content}\n</think>")
            if content:
                parts.append(content)
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    try:
                        args_obj = json.loads(args) if isinstance(args, str) else args
                    except Exception:
                        args_obj = args
                    call_str = json.dumps({"name": name, "arguments": args_obj}, ensure_ascii=False)
                    parts.append(f"<tool_call>\n{call_str}\n</tool_call>")
            asst_text = "\n".join(parts)
            lines.append(f"Assistant: {asst_text}")
        elif role in ("tool", "function"):
            name = getattr(m, "name", None) or getattr(m, "tool_call_id", None) or "tool"
            lines.append(f"Tool ({name}): {content}")
        else:
            label = _ROLE_LABELS.get(role, role.capitalize())
            lines.append(f"{label}: {content}")

    if len(messages) == 1 and messages[0].role == "user" and not tools:
        return _text_of(messages[0].content)

    lines.append("Assistant:")
    return "\n\n".join(lines)


def _now() -> int:
    return int(time.time())


def _id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _est_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) — DeepSeek's web API gives us no count."""
    return max(1, len(text) // 4)


def completion_response(
    model: str,
    content: str,
    prompt: str,
    conversation_id: str = None,
    reasoning: str = None,
) -> dict:
    """A full (non-streaming) OpenAI chat.completion object.

    Includes `reasoning_content` if reasoning/thinking output is present.
    """
    pt = _est_tokens(prompt)
    ct = _est_tokens(content) + (_est_tokens(reasoning) if reasoning else 0)

    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning

    return {
        "id": _id(),
        "object": "chat.completion",
        "created": _now(),
        "model": model,
        "conversation_id": conversation_id,
        "choices": [
            {
                "index": 0,
                "message": msg,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": pt + ct,
        },
    }


def stream_chunks(model: str, stream: Iterable[Any]) -> Iterable[str]:
    """Yield OpenAI SSE lines (`data: {...}\\n\\n`) for a streamed completion.

    Separates reasoning tokens (`delta.reasoning_content`) from answer tokens
    (`delta.content`) for compatibility with Hermes, Open WebUI, and standard LRMs.
    """
    cid, created = _id(), _now()

    def frame(delta: dict, finish=None, extra: dict = None) -> str:
        obj = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if extra:
            obj.update(extra)
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    # First frame announces the assistant role.
    yield frame({"role": "assistant", "content": ""})

    for d in stream:
        if not d:
            continue
        if getattr(d, "is_thinking", False):
            yield frame({"reasoning_content": str(d)})
        else:
            yield frame({"content": str(d)})

    conversation_id = getattr(stream, "conversation_id", None)
    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"

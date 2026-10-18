"""Translate between OpenAI's chat-completions shapes and our DeepSeek client.

DeepSeek's web protocol has no system/role channel — just a single `prompt` string.
So we flatten the OpenAI `messages` array and tool definitions into one prompt,
and wrap DeepSeek's text output back into standard OpenAI response/stream objects,
with native support for `reasoning_content` and DeepSeek DSML tool calling.
"""

from __future__ import annotations

import html
import json
import re
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


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Parse DeepSeek DSML tool calls and <tool_call> blocks into OpenAI tool_calls.

    Handles:
      < | | DSML | | calls>
      < | | DSML | | invoke name="...">
      < | | DSML | | parameter name="..." string="true">...</ | | DSML | | parameter>
      </ | | DSML | | invoke>
      </ | | DSML | | calls>

    As well as fullwidth pipe (<｜DSML｜calls>) and standard <tool_call> JSON blocks.
    """
    if not text:
        return text, []

    tool_calls = []
    clean_text = text

    # Pattern for DSML calls blocks:
    # Delimiters can be: < | | DSML | | calls>, <｜DSML｜calls>, <|DSML|calls>, etc.
    dsml_block_re = re.compile(
        r"<[|\s\uff5c]*DSML[|\s\uff5c]*(?:calls|tool_calls)>([\s\S]*?)(?:</[|\s\uff5c]*DSML[|\s\uff5c]*(?:calls|tool_calls)>|$)",
        re.IGNORECASE,
    )

    invoke_re = re.compile(
        r"<[|\s\uff5c]*DSML[|\s\uff5c]*invoke\s+name=[\"']([^\"']+)[\"']>([\s\S]*?)(?:</[|\s\uff5c]*DSML[|\s\uff5c]*invoke>|$)",
        re.IGNORECASE,
    )

    param_re = re.compile(
        r"<[|\s\uff5c]*DSML[|\s\uff5c]*parameter\s+name=[\"']([^\"']+)[\"'](?:\s+string=[\"']([^\"']+)[\"'])?>([\s\S]*?)</[|\s\uff5c]*DSML[|\s\uff5c]*parameter>",
        re.IGNORECASE,
    )

    for block_match in dsml_block_re.finditer(text):
        block_content = block_match.group(1)
        for inv_match in invoke_re.finditer(block_content):
            tool_name = inv_match.group(1).strip()
            inv_body = inv_match.group(2)
            args = {}
            for p_match in param_re.finditer(inv_body):
                p_name = p_match.group(1).strip()
                p_is_str = (p_match.group(2) or "true").lower() == "true"
                raw_val = p_match.group(3).strip()
                p_val = html.unescape(raw_val)
                if not p_is_str:
                    try:
                        p_val = json.loads(p_val)
                    except Exception:
                        pass
                args[p_name] = p_val

            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False) if args else "{}",
                },
            })

    # Also handle standard <tool_call> JSON format
    tool_call_xml_re = re.compile(r"<tool_call>([\s\S]*?)</tool_call>", re.IGNORECASE)
    for tc_match in tool_call_xml_re.finditer(text):
        raw_json = tc_match.group(1).strip()
        try:
            parsed = json.loads(raw_json)
            name = parsed.get("name") or parsed.get("tool")
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            if name:
                args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args_str,
                    },
                })
        except Exception:
            pass

    if tool_calls:
        # Strip DSML and tool_call markup from clean_text
        clean_text = dsml_block_re.sub("", clean_text)
        clean_text = tool_call_xml_re.sub("", clean_text)
        clean_text = clean_text.strip()

    return clean_text, tool_calls


def messages_to_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[dict]] = None,
) -> str:
    """Flatten a chat history and tool definitions into a single prompt DeepSeek can answer.

    - Supports Hermes / OpenAI tool schemas and tool calls
    - Reconstructs native DeepSeek DSML tool calls and results across turns
    - Preserves system and user turns cleanly
    """
    lines = []

    # If tools are defined, inject tool definitions
    if tools:
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
        tools_system = (
            "You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>"
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
                parts.append("<｜DSML｜calls>")
                for tc in m.tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    raw_args = fn.get("arguments", "{}")
                    try:
                        args_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except Exception:
                        args_dict = {"command": raw_args}
                    parts.append(f'<｜DSML｜invoke name="{name}">')
                    if isinstance(args_dict, dict):
                        for k, v in args_dict.items():
                            val_str = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                            is_str = "true" if isinstance(v, str) else "false"
                            parts.append(f'<｜DSML｜parameter name="{k}" string="{is_str}">{val_str}</｜DSML｜parameter>')
                    else:
                        parts.append(f'<｜DSML｜parameter name="command" string="true">{args_dict}</｜DSML｜parameter>')
                    parts.append("</｜DSML｜invoke>")
                parts.append("</｜DSML｜calls>")
            asst_text = "\n".join(parts)
            lines.append(f"Assistant: {asst_text}")
        elif role in ("tool", "function"):
            name = getattr(m, "name", None) or getattr(m, "tool_call_id", None) or "tool"
            lines.append(f"<｜DSML｜result invoke_name=\"{name}\">\n{content}\n</｜DSML｜result>")
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

    Includes parsed `tool_calls` and `reasoning_content` if present.
    """
    clean_content, tool_calls = parse_tool_calls(content)

    pt = _est_tokens(prompt)
    ct = _est_tokens(content) + (_est_tokens(reasoning) if reasoning else 0)

    finish_reason = "tool_calls" if tool_calls else "stop"

    msg = {
        "role": "assistant",
        "content": clean_content if clean_content else None,
    }
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = tool_calls

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
                "finish_reason": finish_reason,
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

    - Separates reasoning tokens (`delta.reasoning_content`) from answer tokens.
    - Buffers all content text and only emits it after the full response is
      complete, so that DSML / <tool_call> blocks can be detected and translated
      into proper OpenAI `tool_calls` deltas instead of leaking as raw text.
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

    # Collect all content chunks, emitting reasoning immediately but buffering
    # regular content so we can detect tool calls at the end.
    content_buffer = []

    for d in stream:
        if not d:
            continue
        if getattr(d, "is_thinking", False):
            yield frame({"reasoning_content": str(d)})
            continue
        content_buffer.append(str(d))

    conversation_id = getattr(stream, "conversation_id", None)
    full_text = "".join(content_buffer)

    # Check if the accumulated text contains tool calls
    clean_text, tool_calls = parse_tool_calls(full_text)

    if tool_calls:
        # Emit any leading text before the tool calls
        if clean_text:
            yield frame({"content": clean_text})
        # Emit each tool call as a proper delta
        for idx, tc in enumerate(tool_calls):
            yield frame({
                "tool_calls": [
                    {
                        "index": idx,
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"],
                        },
                    }
                ]
            })
        yield frame({}, finish="tool_calls", extra={"conversation_id": conversation_id})
        yield "data: [DONE]\n\n"
        return

    # No tool calls — emit all content chunks as normal streaming deltas
    for chunk_text in content_buffer:
        if chunk_text:
            yield frame({"content": chunk_text})

    yield frame({}, finish="stop", extra={"conversation_id": conversation_id})
    yield "data: [DONE]\n\n"


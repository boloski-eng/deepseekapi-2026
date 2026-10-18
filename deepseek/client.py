"""
Pure-HTTP DeepSeek chat client.

Speaks chat.deepseek.com's internal API directly using a captured signed-in
session (see `deepseek.auth`). For each message it:

    1. creates a chat session   (POST /api/v0/chat_session/create)
    2. fetches a PoW challenge   (POST /api/v0/chat/create_pow_challenge)
    3. solves it via the WASM    (deepseek.pow.DeepSeekPow)
    4. POSTs the completion       with the x-ds-pow-response header
    5. parses the SSE stream      into text

    from deepseek.auth import get_session
    from deepseek.client import DeepSeekClient

    client = DeepSeekClient(get_session())
    print(client.chat("Hello!"))                 # full reply
    for chunk in client.stream("Tell a joke"):   # streamed
        print(chunk, end="", flush=True)
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from typing import Iterator, Optional

import httpx

from .auth import Session, get_session
from .pow import DeepSeekPow

BASE = "https://chat.deepseek.com"
COMPLETION_PATH = "/api/v0/chat/completion"

# DeepSeek's mode pill, sent as `model_type` in the completion body. "default" is
# Instant (the fast model); "expert" is the stronger, slower model. Omitting the
# field lets the backend pick, so we always send one explicitly.
DEFAULT_MODEL_TYPE = "default"

# A conversation_id is an opaque "<chat_session_id>:<last_message_id>" token. It
# carries everything needed to resume a thread, so the client stays stateless.
_CID_SEP = ":"


def _encode_cid(session_id: str, message_id: Optional[int]) -> str:
    if message_id is None:
        return session_id
    return f"{session_id}{_CID_SEP}{message_id}"


def _decode_cid(conversation_id: Optional[str]) -> tuple[Optional[str], Optional[int]]:
    """Split a conversation_id back into (chat_session_id, parent_message_id)."""
    if not conversation_id:
        return None, None
    session_id, _, msg = conversation_id.partition(_CID_SEP)
    parent = int(msg) if msg.isdigit() else None
    return (session_id or None), parent


@dataclass
class Reply:
    """A completed chat reply plus the id to resume the conversation."""

    text: str
    conversation_id: str
    reasoning: Optional[str] = None

    def __str__(self) -> str:  # so print(reply) shows the text
        return self.text


def _biz(data: dict) -> dict:
    """Unwrap DeepSeek's `data.biz_data` envelope, raising on API-level errors."""
    if data.get("code") != 0:
        raise RuntimeError(f"DeepSeek API error: {data.get('msg') or data}")
    biz = data.get("data", {}).get("biz_data")
    if biz is None:
        raise RuntimeError(f"Unexpected response shape: {data}")
    return biz


class DeepSeekClient:
    def __init__(
        self,
        session: Optional[Session] = None,
        allow_interactive: bool = True,
    ):
        # `allow_interactive=False` makes session resolution non-blocking: it
        # uses a cached/headless session and raises LoginRequired instead of
        # opening a browser window. The server passes False (see server/api.py).
        self.session = session or get_session(allow_interactive=allow_interactive)
        self._pow = DeepSeekPow()
        # The wasmtime Store behind the PoW solver is not reentrant; serialise
        # access so concurrent server requests don't corrupt it.
        self._pow_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=BASE,
            headers=self._base_headers(),
            cookies=self.session.cookies,
            timeout=httpx.Timeout(120.0, read=300.0),
        )

    def _base_headers(self) -> dict:
        return {
            "authorization": f"Bearer {self.session.token}",
            "accept": "*/*",
            "content-type": "application/json",
            "user-agent": self.session.user_agent,
            "origin": BASE,
            "referer": f"{BASE}/",
            "x-app-version": "2.0.0",
            "x-client-version": "2.0.0",
            "x-client-platform": "web",
            "x-client-locale": "en_US",
            "x-client-bundle-id": "com.deepseek.chat",
            "x-client-timezone-offset": "19800",
        }

    # --- protocol steps -----------------------------------------------------

    def create_chat_session(self) -> str:
        r = self._http.post("/api/v0/chat_session/create", json={})
        r.raise_for_status()
        return _biz(r.json())["chat_session"]["id"]

    def _pow_header(self, target_path: str = COMPLETION_PATH) -> str:
        r = self._http.post(
            "/api/v0/chat/create_pow_challenge", json={"target_path": target_path}
        )
        r.raise_for_status()
        challenge = _biz(r.json())["challenge"]
        with self._pow_lock:
            return self._pow.make_header(challenge)

    # --- public API ---------------------------------------------------------

    def stream(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> "_Stream":
        """Stream a reply. Iterate it for text chunks; read `.conversation_id`
        afterwards to resume the thread. Pass an existing `conversation_id` to
        continue a previous conversation.

        `model` is DeepSeek's model_type wire value: "default" (Instant) or
        "expert"; it defaults to "default" on a NEW thread. It cannot be combined
        with `conversation_id` — a thread's model is fixed when it's created, so
        resuming keeps the original model. `thinking` enables DeepThink reasoning
        and `search` enables web search; both are independent of the model.
        """
        if conversation_id and model is not None:
            raise ValueError(
                "`model` cannot be set together with `conversation_id`; a thread's "
                "model is fixed when it is created. Pass `model` only on the first turn."
            )
        session_id, parent_id = _decode_cid(conversation_id)
        if session_id is None:
            # New thread: select the model (default when unspecified).
            session_id = self.create_chat_session()
            model_type: Optional[str] = model or DEFAULT_MODEL_TYPE
        else:
            # Resuming: let the existing thread's model stand (send no model_type).
            model_type = None
        return _Stream(self, prompt, session_id, parent_id, model_type, thinking, search)

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        model: Optional[str] = None,
        thinking: bool = False,
        search: bool = False,
    ) -> Reply:
        """Return the complete reply (`.text`) plus its `.conversation_id` and optional `.reasoning`."""
        s = self.stream(prompt, conversation_id=conversation_id,
                        model=model, thinking=thinking, search=search)
        content_parts = []
        reasoning_parts = []
        for chunk in s:
            if getattr(chunk, "is_thinking", False):
                reasoning_parts.append(str(chunk))
            else:
                content_parts.append(str(chunk))
        text = "".join(content_parts)
        reasoning = "".join(reasoning_parts) if reasoning_parts else None
        return Reply(text=text, conversation_id=s.conversation_id, reasoning=reasoning)

    def close(self) -> None:
        self._http.close()


class _Stream:
    """Iterable of reply-text chunks. After it's consumed, `.conversation_id`
    holds the token for resuming the conversation."""

    def __init__(self, client: "DeepSeekClient", prompt: str, session_id: str,
                 parent_id: Optional[int], model: str,
                 thinking: bool, search: bool):
        self._client = client
        self._prompt = prompt
        self._session_id = session_id
        self._parent_id = parent_id
        self._model = model
        self._thinking = thinking
        self._search = search
        self._message_id: Optional[int] = None

    def __iter__(self) -> Iterator[str]:
        body = {
            "chat_session_id": self._session_id,
            "parent_message_id": self._parent_id,
            "prompt": self._prompt,
            "ref_file_ids": [],
            "thinking_enabled": self._thinking,
            "search_enabled": self._search,
            "action": None,
            "preempt": False,
        }
        # Only select a model on a new thread; on resume the thread keeps its own.
        if self._model is not None:
            body["model_type"] = self._model
        # PoW challenges are short-lived, so solve right before the request.
        headers = {"x-ds-pow-response": self._client._pow_header()}
        meta: dict = {}
        with self._client._http.stream(
            "POST", COMPLETION_PATH, json=body, headers=headers
        ) as resp:
            resp.raise_for_status()
            yield from _parse_sse(resp.iter_lines(), meta)
        if meta.get("message_id") is not None:
            self._message_id = meta["message_id"]

    @property
    def conversation_id(self) -> str:
        return _encode_cid(self._session_id, self._message_id)


class StreamChunk(str):
    """A string chunk that carries an `is_thinking` boolean attribute."""

    is_thinking: bool = False

    def __new__(cls, content: str, is_thinking: bool = False):
        obj = super().__new__(cls, content)
        obj.is_thinking = is_thinking
        return obj


def _parse_sse(lines, meta: Optional[dict] = None) -> Iterator[StreamChunk]:
    """Turn DeepSeek's SSE completion stream into reply-text deltas.

    Handles both THINKING (reasoning) and RESPONSE fragments accurately.
    Preserves active fragment pointers across metadata patches (token counts,
    elapsed seconds, status updates), preventing dropped chunks and truncated
    responses.
    """
    fragment_types: dict[int, str] = {}
    emitted_lengths: dict[int, int] = {}
    active_frag_idx: Optional[int] = None
    active_prop: Optional[str] = None
    buffer = ""

    for line in lines:
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8", errors="replace")

        line = line.strip()
        if not line:
            continue

        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
        else:
            payload = line

        if not payload or payload == "[DONE]":
            continue

        try:
            obj = json.loads(payload)
            buffer = ""
        except json.JSONDecodeError:
            buffer += payload
            try:
                obj = json.loads(buffer)
                buffer = ""
            except json.JSONDecodeError:
                continue

        v = obj.get("v")

        # Snapshot frame: contains full response object or updated fragments
        if isinstance(v, dict) and "response" in v:
            resp_obj = v["response"]
            if meta is not None:
                _capture_message_id(meta, v)

            fragments = resp_obj.get("fragments", [])
            for idx, frag in enumerate(fragments):
                frag_type = frag.get("type", "RESPONSE")
                fragment_types[idx] = frag_type
                content = frag.get("content", "")
                if isinstance(content, str) and content:
                    prev_len = emitted_lengths.get(idx, 0)
                    if len(content) > prev_len:
                        delta = content[prev_len:]
                        emitted_lengths[idx] = len(content)
                        active_frag_idx = idx
                        active_prop = "content"
                        yield StreamChunk(delta, is_thinking=(frag_type == "THINKING"))
            continue

        # Patch frame with path "p"
        if "p" in obj:
            p = str(obj["p"])
            if meta is not None and (p.endswith("message_id") or p.endswith("/id")) and isinstance(v, int):
                meta["message_id"] = v

            frag_match = re.match(r"^response/fragments/(-?\d+)(?:/(.*))?$", p)
            if frag_match:
                raw_idx = int(frag_match.group(1))
                if raw_idx < 0:
                    raw_idx = max(fragment_types.keys(), default=0)
                prop = frag_match.group(2) or ""

                if prop == "type" and isinstance(v, str):
                    fragment_types[raw_idx] = v

                if prop == "content" or prop.endswith("/content") or prop == "":
                    active_frag_idx = raw_idx
                    active_prop = "content"

                    if isinstance(v, str) and v:
                        frag_type = fragment_types.get(active_frag_idx, "RESPONSE")
                        emitted_lengths[active_frag_idx] = emitted_lengths.get(active_frag_idx, 0) + len(v)
                        yield StreamChunk(v, is_thinking=(frag_type == "THINKING"))
            continue

        # Bare append frame: {"v": "..."} without "p"
        if isinstance(v, str) and v and active_prop == "content" and active_frag_idx is not None:
            frag_type = fragment_types.get(active_frag_idx, "RESPONSE")
            emitted_lengths[active_frag_idx] = emitted_lengths.get(active_frag_idx, 0) + len(v)
            yield StreamChunk(v, is_thinking=(frag_type == "THINKING"))


def _capture_message_id(meta: dict, snapshot: dict) -> None:
    """Best-effort: pull the assistant message_id out of a snapshot frame.

    DeepSeek nests the assistant message under `response`; we check there first,
    then the snapshot root, accepting `message_id` or `id`.
    """
    for container in (snapshot.get("response"), snapshot):
        if isinstance(container, dict):
            mid = container.get("message_id", container.get("id"))
            if isinstance(mid, int):
                meta["message_id"] = mid
                return

"""Unofficial OpenAI-compatible client for chat.deepseek.com."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auth import Session, get_session, login
    from .client import DeepSeekClient, Reply
    from .pow import DeepSeekPow

def __getattr__(name: str):
    if name in ("Session", "get_session", "login"):
        from . import auth
        return getattr(auth, name)
    if name in ("DeepSeekClient", "Reply"):
        from . import client
        return getattr(client, name)
    if name == "DeepSeekPow":
        from . import pow
        return getattr(pow, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["Session", "get_session", "login", "DeepSeekClient", "Reply", "DeepSeekPow"]


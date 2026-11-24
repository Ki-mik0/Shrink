"""Find the text an agent's API request lets shrink change, and nothing else.

Only two kinds of strings are rewritten, always in place, so the request keeps
its exact structure (block lists stay lists, ``cache_control``, thinking
blocks, ``system`` and tool definitions are never touched):

* ``tool``: what a tool returned (command output, search results);
* ``user``: what the user typed (only translated, with ``--translate``).

Results of tools that read or edit files are left alone: the agent edits a
file by quoting what it saw, and a shortened copy would make that fail.

Supported request bodies: Anthropic Messages (``/v1/messages``, also
``count_tokens``), OpenAI Responses (``/responses``) and Chat Completions
(``/chat/completions``). Anything else is forwarded untouched.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

#: rewrite(text, kind, tool_name) -> new text (the same object when unchanged)
Rewrite = Callable[[str, str, str | None], str]

READ_LIKE = re.compile(r"read|view|open|cat\b|edit|write|notebook|str_replace|patch", re.I)


def api_of(path: str) -> str | None:
    route = path.split("?", 1)[0].rstrip("/")
    if route.endswith(("/v1/messages", "/v1/messages/count_tokens")):
        return "anthropic"
    if route.endswith("/responses"):
        return "openai-responses"
    if route.endswith("/chat/completions"):
        return "openai-chat"
    return None


def rewrite_request(api: str, body: dict[str, Any], rewrite: Rewrite) -> bool:
    """Rewrite ``body`` in place; True if anything changed."""
    walkers = {
        "anthropic": _anthropic,
        "openai-responses": _openai_responses,
        "openai-chat": _openai_chat,
    }
    return walkers[api](body, rewrite)


def _skipped(tool: str | None) -> bool:
    return tool is not None and READ_LIKE.search(tool) is not None


def _set(holder: dict[str, Any], key: str, kind: str, tool: str | None, rewrite: Rewrite) -> bool:
    value = holder.get(key)
    if not isinstance(value, str) or (kind == "tool" and _skipped(tool)):
        return False
    new = rewrite(value, kind, tool)
    if new == value:
        return False
    holder[key] = new
    return True


def _text_blocks(
    holder: dict[str, Any],
    key: str,
    kind: str,
    tool: str | None,
    rewrite: Rewrite,
    *,
    types: set[str],
) -> bool:
    """``holder[key]`` is a string or a list of ``{"type": ..., "text": ...}`` blocks."""
    value = holder.get(key)
    if isinstance(value, str):
        return _set(holder, key, kind, tool, rewrite)
    changed = False
    if isinstance(value, list):
        for block in value:
            if isinstance(block, dict) and block.get("type") in types:
                changed |= _set(block, "text", kind, tool, rewrite)
    return changed


# ------------------------------------------------------------------ Anthropic


def _anthropic(body: dict[str, Any], rewrite: Rewrite) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    tools: dict[str, str] = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") in ("tool_use", "server_tool_use"):
                    tools[str(block.get("id"))] = str(block.get("name"))
    changed = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            changed |= _set(message, "content", "user", None, rewrite)
            continue
        for block in content if isinstance(content, list) else []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                changed |= _set(block, "text", "user", None, rewrite)
            elif block.get("type") == "tool_result":
                tool = tools.get(str(block.get("tool_use_id")))
                changed |= _text_blocks(block, "content", "tool", tool, rewrite, types={"text"})
    return changed


# ------------------------------------------------------------------ OpenAI


def _openai_responses(body: dict[str, Any], rewrite: Rewrite) -> bool:
    items = body.get("input")
    if isinstance(items, str):
        return _set(body, "input", "user", None, rewrite)
    if not isinstance(items, list):
        return False
    tools = {
        str(item.get("call_id")): str(item.get("name") or item.get("type"))
        for item in items
        if isinstance(item, dict) and str(item.get("type", "")).endswith("_call")
    }
    changed = False
    for item in items:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("type", "message"))
        if kind == "message" and item.get("role") == "user":
            changed |= _text_blocks(item, "content", "user", None, rewrite, types={"input_text"})
        elif kind.endswith("_call_output"):
            tool = tools.get(str(item.get("call_id")))
            changed |= _set(item, "output", "tool", tool, rewrite)
    return changed


def _openai_chat(body: dict[str, Any], rewrite: Rewrite) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return False
    tools: dict[str, str] = {}
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                if isinstance(call, dict):
                    name = (call.get("function") or {}).get("name")
                    tools[str(call.get("id"))] = str(name)
    changed = False
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "user":
            changed |= _text_blocks(message, "content", "user", None, rewrite, types={"text"})
        elif role == "tool":
            tool = tools.get(str(message.get("tool_call_id")))
            changed |= _text_blocks(message, "content", "tool", tool, rewrite, types={"text"})
    return changed

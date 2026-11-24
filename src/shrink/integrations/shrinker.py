"""Compress the parts of an agent request that may change, and measure the saving.

Agents send the whole conversation again on every turn, so the same tool
output is seen many times. Results are cached by content: repeated text costs
nothing, and, because the compressors are deterministic, earlier turns come out
byte-identical every time, which keeps the provider's prompt cache working.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from shrink.compressors.base import Compressor
from shrink.document import Document
from shrink.integrations.adapters import api_of, rewrite_request

CACHE_SIZE = 4096


@dataclass(frozen=True)
class Saving:
    api: str
    blocks: int  # strings that changed
    tokens_before: int
    tokens_after: int


class RequestShrinker:
    def __init__(
        self,
        *,
        level: int = 1,
        translate: Compressor | None = None,
        count: Callable[[str], int] | None = None,
    ) -> None:
        self._level = level
        self._translate = translate
        self._count = count or (lambda text: (len(text) + 3) // 4)
        self._cache: OrderedDict[str, tuple[str, int, int]] = OrderedDict()
        self._lock = threading.Lock()

    def process(
        self, path: str, body: bytes, content_encoding: str = ""
    ) -> tuple[bytes, Saving | None]:
        """The body to forward (the original bytes when nothing changed) and the saving."""
        api = api_of(path)
        if api is None or not body or content_encoding.strip().lower() not in ("", "identity"):
            return body, None
        try:
            data = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return body, None
        if not isinstance(data, dict):
            return body, None
        totals = [0, 0, 0]  # blocks, before, after

        def rewrite(text: str, kind: str, tool: str | None) -> str:
            new, before, after = self._rewrite(text, kind)
            if new != text:
                totals[0] += 1
                totals[1] += before
                totals[2] += after
            return new

        if not rewrite_request(api, data, rewrite):
            return body, None
        out = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return out, Saving(api, totals[0], totals[1], totals[2])

    def _rewrite(self, text: str, kind: str) -> tuple[str, int, int]:
        key = hashlib.sha1(f"{kind}\0{text}".encode("utf-8", "surrogatepass")).hexdigest()
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
            new = self._compress(text, kind)
            result = (new, self._count(text), self._count(new)) if new != text else (text, 0, 0)
            self._cache[key] = result
            if len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
            return result

    def _compress(self, text: str, kind: str) -> str:
        try:
            if kind == "tool":
                return _compress_tool_output(text, self._level)
            if kind == "user" and self._translate is not None:
                return self._translate.compress(Document(text), self._level).text
        except Exception:  # fail-open: the agent gets the original text
            return text
        return text


def _compress_tool_output(text: str, level: int) -> str:
    from shrink.compressors.json_mode import compress_json, looks_like_json
    from shrink.compressors.log import compress_log

    new = compress_json(text, level) if looks_like_json(text) else compress_log(text, level)
    return new if len(new) < len(text) else text


def describe(saving: Saving) -> dict[str, Any]:
    return {
        "api": saving.api,
        "blocks": saving.blocks,
        "tokens_before": saving.tokens_before,
        "tokens_after": saving.tokens_after,
    }

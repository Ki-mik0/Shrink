"""Token counting and the ``--stats`` report on stderr.

The exact counter (tiktoken) is optional: ``pip install shrink[tokens]``. Its
vocabulary is fetched once, on request, with ``shrink --download tokenizer``
and kept in shrink's cache directory; after that counting works offline. A
normal run never touches the network: without the vocabulary, or without
tiktoken, counting falls back to the ``chars / 4`` estimate, marked ``~``.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TextIO

from shrink.compressors.base import Step
from shrink.document import Document
from shrink.io import ENCODING, ERRORS
from shrink.observability import DEFAULT_ENCODING
from shrink.paths import cache_dir

CHARS_PER_TOKEN = 4


class TokenCounter(Protocol):
    name: str
    approximate: bool

    def count(self, text: str) -> int: ...


class ApproxCounter:
    """About four characters per token for English text and code."""

    name = "approx"
    approximate = True

    def __init__(self, hint: str | None = None) -> None:
        self.hint = hint  # how to get exact counts, shown under the report

    def count(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)


class TokenizerMissingError(RuntimeError):
    pass


def tokenizer_dir() -> Path:
    return cache_dir() / "tiktoken"


def _ready_marker(encoding: str) -> Path:
    return tokenizer_dir() / f"{encoding}.ready"


@contextlib.contextmanager
def _tiktoken_cache() -> Iterator[None]:
    """Point tiktoken at shrink's cache for the duration of a load."""
    old = os.environ.get("TIKTOKEN_CACHE_DIR")
    os.environ["TIKTOKEN_CACHE_DIR"] = str(tokenizer_dir())
    try:
        yield
    finally:
        if old is None:
            del os.environ["TIKTOKEN_CACHE_DIR"]
        else:
            os.environ["TIKTOKEN_CACHE_DIR"] = old


class TiktokenCounter:
    approximate = False
    hint = None

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        import tiktoken

        if not _ready_marker(encoding).is_file():
            raise TokenizerMissingError(f"{encoding} vocabulary not downloaded")
        with _tiktoken_cache():
            self._encoding = tiktoken.get_encoding(encoding)
        self.name = encoding

    def count(self, text: str) -> int:
        # Lone surrogates (undecodable input bytes) cannot be encoded by tiktoken.
        clean = text.encode(ENCODING, ERRORS).decode(ENCODING, "replace")
        # encode_ordinary: "<|endoftext|>" in the input is text, not a special token.
        return len(self._encoding.encode_ordinary(clean))


def download_tokenizer(encoding: str = DEFAULT_ENCODING) -> Path:
    """Fetch and verify the vocabulary (tiktoken checks its hash); return the folder."""
    try:
        import tiktoken
    except ImportError as exc:
        raise RuntimeError("tiktoken is not installed: pip install 'shrink[tokens]'") from exc
    folder = tokenizer_dir()
    folder.mkdir(parents=True, exist_ok=True)
    with _tiktoken_cache():
        tiktoken.get_encoding(encoding).encode_ordinary("check")
    _ready_marker(encoding).write_text("ok\n", encoding="utf-8")
    return folder


def make_counter(encoding: str = DEFAULT_ENCODING) -> TokenCounter:
    """The exact counter when available, otherwise the estimate. Never raises."""
    try:
        return TiktokenCounter(encoding)
    except ImportError:
        return ApproxCounter("for exact counts: pip install 'shrink[tokens]'")
    except TokenizerMissingError:
        option = "" if encoding == DEFAULT_ENCODING else f" --encoding {encoding}"
        return ApproxCounter(f"for exact counts, once: shrink --download tokenizer{option}")
    except Exception:  # damaged cache, unknown encoding
        return ApproxCounter()


def format_change(before: int, after: int) -> str:
    if before == 0:
        return "0.0%"
    return f"{(after - before) / before * 100:+.1f}%"


@dataclass(frozen=True)
class Price:
    """USD per million input tokens; ``cached`` for cache reads, ``hit`` their share."""

    input: float
    cached: float | None = None
    hit: float = 0.0

    @classmethod
    def parse(cls, spec: str, hit: float = 0.0) -> Price:
        """``"3"`` or ``"3/0.3"`` (input / cached input), in USD per million tokens."""
        fresh, _, cached = spec.partition("/")
        price = cls(float(fresh), float(cached) if cached else None, hit)
        if price.input < 0 or (price.cached is not None and price.cached < 0):
            raise ValueError("prices cannot be negative")
        if not 0.0 <= hit <= 1.0:
            raise ValueError("the cache hit rate must be between 0 and 1")
        if hit > 0 and price.cached is None:
            raise ValueError("a cache hit rate needs a cached price: --price INPUT/CACHED")
        return price

    def cost(self, tokens: int) -> float:
        cached = self.cached if self.cached is not None else self.input
        per_million = (1 - self.hit) * self.input + self.hit * cached
        return tokens * per_million / 1_000_000

    def describe(self) -> str:
        text = f"${self.input:g}/Mtok"
        if self.cached is not None:
            text += f", cached ${self.cached:g}/Mtok, {self.hit:.0%} cache hits"
        return text


def _money(value: float) -> str:
    if abs(value) >= 1:
        return f"${value:.2f}"
    return f"${value:.4f}" if abs(value) >= 0.01 else f"${value:.6f}"


class StatsReporter:
    """Accumulates per-document counts and prints a summary, then one line per step."""

    wants_steps = True

    def __init__(
        self,
        counter: TokenCounter,
        stream: TextIO | None = None,
        *,
        fmt: str = "text",
        price: Price | None = None,
    ) -> None:
        self._counter = counter
        self._stream = stream
        self._fmt = fmt
        self._price = price
        self.documents = 0
        self.tokens_before = 0
        self.tokens_after = 0
        # step name -> [tokens in, tokens out], summed over documents, in first-seen order
        self.steps: dict[str, list[int]] = {}

    def record(self, before: Document, after: Document, steps: Sequence[Step] = ()) -> None:
        self.documents += 1
        count_before = self._counter.count(before.full_text)
        self.tokens_before += count_before
        if not steps:
            self.tokens_after += self._counter.count(after.full_text)
            return
        previous = count_before
        for step in steps:
            current = self._counter.count(step.after.full_text)
            totals = self.steps.setdefault(step.name, [0, 0])
            totals[0] += previous
            totals[1] += current
            previous = current
        self.tokens_after += previous

    def _mark(self) -> str:
        return "~" if self._counter.approximate else ""

    def summary(self) -> str:
        mark = self._mark()
        docs = "document" if self.documents == 1 else "documents"
        return (
            f"shrink: stats: {self.documents} {docs}, "
            f"tokens {mark}{self.tokens_before} -> {mark}{self.tokens_after} "
            f"({format_change(self.tokens_before, self.tokens_after)}) "
            f"[{self._counter.name}]"
        )

    def step_lines(self) -> list[str]:
        """What each step saved, in tokens and as a share of the whole input."""
        mark = self._mark()
        width = max((len(name) for name in self.steps), default=0)
        lines = []
        for name, (tokens_in, tokens_out) in self.steps.items():
            share = format_change(self.tokens_before, self.tokens_before - (tokens_in - tokens_out))
            lines.append(
                f"shrink:   {name:<{width}}  {mark}{tokens_in} -> {mark}{tokens_out} "
                f"({format_change(tokens_in, tokens_out)}, {share} of input)"
            )
        return lines

    def cost_line(self) -> str | None:
        if self._price is None:
            return None
        before = self._price.cost(self.tokens_before)
        after = self._price.cost(self.tokens_after)
        return (
            f"shrink: cost: {_money(before)} -> {_money(after)} per call, "
            f"saves {_money(before - after)} ({_money((before - after) * 1000)} per 1000 calls) "
            f"at {self._price.describe()}"
        )

    def as_dict(self) -> dict[str, object]:
        saved = self.tokens_before - self.tokens_after
        data: dict[str, object] = {
            "documents": self.documents,
            "encoding": self._counter.name,
            "approximate": self._counter.approximate,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": saved,
            "saved_percent": round(saved / self.tokens_before * 100, 2)
            if self.tokens_before
            else 0.0,
            "steps": [
                {"name": name, "tokens_in": tokens_in, "tokens_out": tokens_out}
                for name, (tokens_in, tokens_out) in self.steps.items()
            ],
        }
        if self._price is not None:
            before = self._price.cost(self.tokens_before)
            after = self._price.cost(self.tokens_after)
            data["cost_usd"] = {
                "before": round(before, 8),
                "after": round(after, 8),
                "saved": round(before - after, 8),
                "price_per_mtok": self._price.input,
                "cached_price_per_mtok": self._price.cached,
                "cache_hit_rate": self._price.hit,
            }
        return data

    def finish(self) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        if self._fmt == "json":
            stream.write(json.dumps(self.as_dict(), ensure_ascii=False) + "\n")
        else:
            lines = [self.summary(), *self.step_lines()]
            cost = self.cost_line()
            if cost is not None:
                lines.append(cost)
            hint = getattr(self._counter, "hint", None)
            if hint and self._counter.approximate:
                lines.append(f"shrink: {hint}")
            stream.write("\n".join(lines) + "\n")
        stream.flush()

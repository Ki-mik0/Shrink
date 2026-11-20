"""Token counting and the ``--stats`` report on stderr.

The exact counter (tiktoken) is optional: ``pip install shrink[tokens]``. If it
is missing or cannot load its vocabulary (no network on first run), counting
falls back to the ``chars / 4`` estimate and the report marks numbers with ``~``.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from typing import Protocol, TextIO

from shrink.compressors.base import Step
from shrink.document import Document
from shrink.io import ENCODING, ERRORS

DEFAULT_ENCODING = "o200k_base"
CHARS_PER_TOKEN = 4


class TokenCounter(Protocol):
    name: str
    approximate: bool

    def count(self, text: str) -> int: ...


class ApproxCounter:
    """About four characters per token for English text and code."""

    name = "approx"
    approximate = True

    def count(self, text: str) -> int:
        return math.ceil(len(text) / CHARS_PER_TOKEN)


class TiktokenCounter:
    approximate = False

    def __init__(self, encoding: str = DEFAULT_ENCODING) -> None:
        import tiktoken

        self._encoding = tiktoken.get_encoding(encoding)
        self.name = encoding

    def count(self, text: str) -> int:
        # Lone surrogates (undecodable input bytes) cannot be encoded by tiktoken.
        clean = text.encode(ENCODING, ERRORS).decode(ENCODING, "replace")
        return len(self._encoding.encode(clean, disallowed_special=()))


def make_counter(encoding: str = DEFAULT_ENCODING) -> TokenCounter:
    """The exact counter when available, otherwise the estimate. Never raises."""
    try:
        return TiktokenCounter(encoding)
    except Exception:  # missing package, no network, unknown encoding
        return ApproxCounter()


def format_change(before: int, after: int) -> str:
    if before == 0:
        return "0.0%"
    return f"{(after - before) / before * 100:+.1f}%"


class StatsReporter:
    """Accumulates per-document counts and prints a summary, then one line per step."""

    wants_steps = True

    def __init__(self, counter: TokenCounter, stream: TextIO | None = None) -> None:
        self._counter = counter
        self._stream = stream
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

    def finish(self) -> None:
        stream = self._stream if self._stream is not None else sys.stderr
        stream.write("\n".join([self.summary(), *self.step_lines()]) + "\n")
        stream.flush()
